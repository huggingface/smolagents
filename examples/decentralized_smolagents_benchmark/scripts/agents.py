"""Agent definitions and utilities for decentralized team."""

import importlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from scripts.text_inspector_tool import TextInspectorTool
from scripts.text_web_browser import (
    ArchiveSearchTool,
    FinderTool,
    FindNextTool,
    PageDownTool,
    PageUpTool,
    SimpleTextBrowser,
    VisitTool,
)
from scripts.visual_qa import visualizer
from smolagents import GoogleSearchTool, ToolCallingAgent
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool

from .message_store import MessageStore


# Browser configuration
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}


@dataclass
class AgentConfig:
    name: str
    role: str
    tools: List[Tool]
    system_prompt: str
    model: Any  # Model instance
    max_turns: int = 10
    concurrency_limit: int = 3


class DecentralizedAgent:
    def __init__(
        self,
        config: AgentConfig,
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
    ):
        self.config = config
        self.message_store = message_store
        self.last_seen_msg = None
        self.turn_count = 0

        self.agent = ToolCallingAgent(
            tools=config.tools, model=self.config.model, instructions=config.system_prompt, max_steps=20
        )

        # Thread pool for parallel tool execution
        self.executor = ThreadPoolExecutor(max_workers=config.concurrency_limit)

    def check_messages(self) -> List[Dict]:
        """Check for new messages."""
        messages = self.message_store.get_messages(agent_id=self.config.name, last_seen=self.last_seen_msg)
        if messages:
            self.last_seen_msg = messages[-1]["id"]
        return messages

    def post_message(self, content: str, **kwargs) -> str:
        """Post a new message to the store."""
        message = {"sender": self.config.name, "content": content, **kwargs}
        return self.message_store.post_message(message)

    def vote_on_proposal(self, proposal_id: str, content: str, vote: str):
        """Vote yes/no on a proposal."""
        self.post_message(content=content, reply_to=proposal_id, vote=vote)

    def run_step(self) -> bool:
        """Run one step/turn of the agent's loop."""
        if self.turn_count >= self.config.max_turns:
            return False

        # Check messages
        messages = self.check_messages()
        if not messages:
            time.sleep(0.1)  # Avoid tight polling
            return True

        # Process messages
        for msg in messages:
            if msg.get("type") == "final_answer_proposal":
                # Handle voting on proposals
                result = self.agent.run(
                    f"Please evaluate this final answer proposal and vote yes/no:\n{msg['content']}"
                )
                result_text = str(result) if result else ""
                vote = "yes" if "agree" in result_text.lower() else "no"
                self.vote_on_proposal(msg["id"], result_text, vote)

            else:
                # Regular message processing
                result = self.agent.run(
                    f"New message from {msg['sender']}: {msg['content']}\n"
                    "You can reply or take any appropriate action."
                )
                if result:
                    self.post_message(str(result))

        self.turn_count += 1
        return True

    def run(self):
        """Run the agent's main loop."""
        while self.run_step():
            pass


def create_team(
    message_store: MessageStore, model_type: str, model_id: str, provider: Optional[str] = None
) -> List[DecentralizedAgent]:
    """Create the team of specialized agents."""

    base_prompt = """You are a specialized agent working as part of a decentralized team.
Your responsibilities:
- Check messages frequently
- Prioritize messages tagged as FinalAnswer proposals
- Keep messages concise and cite sources where relevant
- Vote honestly on FinalAnswer proposals based on your expertise
"""
    # Initialize shared components
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    # Create directory for browser downloads if needed
    os.makedirs(BROWSER_CONFIG["downloads_folder"], exist_ok=True)

    # Initialize model
    model_cls = importlib.import_module("smolagents.models").LiteLLMModel
    model = model_cls(model_name=model_id, provider_name=provider)

    # Create tool instances for each agent type
    code_tools = [Tool(PythonInterpreterTool())]

    # Full web tools suite
    web_tools = [
        Tool(GoogleSearchTool(provider="serper")),
        Tool(VisitTool(browser)),
        Tool(PageUpTool(browser)),
        Tool(PageDownTool(browser)),
        Tool(FinderTool(browser)),
        Tool(FindNextTool(browser)),
        Tool(ArchiveSearchTool(browser)),
        Tool(TextInspectorTool(model, text_limit)),
        Tool(visualizer),
    ]

    reader_tools = [
        Tool.from_code("""
from smolagents.tools import Tool
class FileReaderTool(Tool):
    '''Tool for reading file contents'''
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        }
    }
    description = "Read contents of a file"
    def __call__(self, file_path: str) -> str:
        '''Read contents of a file'''
        with open(file_path) as f:
            return f.read()
    """)
    ]

    configs = [
        AgentConfig(
            name="CodeAgent",
            role="Python code execution specialist",
            tools=code_tools,
            model=model,
            system_prompt=base_prompt + "\nYou specialize in writing and executing Python code.",
        ),
        AgentConfig(
            name="WebSearchAgent",
            role="Fast web research specialist",
            tools=web_tools,
            model=model,
            system_prompt=base_prompt + "\nYou specialize in quick web searches and shallow synthesis.",
        ),
        AgentConfig(
            name="DeepResearchAgent",
            role="Deep analysis specialist",
            tools=web_tools + code_tools,
            model=model,
            system_prompt=base_prompt + "\nYou specialize in deep research and exploring multiple hypotheses.",
        ),
        AgentConfig(
            name="DocumentReaderAgent",
            role="Document analysis specialist",
            tools=reader_tools,
            model=model,
            system_prompt=base_prompt + "\nYou specialize in reading and analyzing documents.",
        ),
    ]

    return [
        DecentralizedAgent(
            config=config, message_store=message_store, model_type=model_type, model_id=model_id, provider=provider
        )
        for config in configs
    ]
