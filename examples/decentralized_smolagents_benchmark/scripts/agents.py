"""Agent definitions and utilities for decentralized team."""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
from smolagents import GoogleSearchTool, LiteLLMModel, ToolCallingAgent
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool

from .message_store import MessageStore


# evaluation roles
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

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

# ------------------------ Prompts (Team Charter) ---------------------------

TEAM_CHARTER = """
You are a specialized agent working as part of a decentralized team composed of 4 agents working through a shared message bus.
Decision rule: A final answer is accepted when a strict majority of all agents vote YES
(required_yes = floor(N/2)+1 = 3).

Your responsibilities:
- Check messages frequently
- Prioritize messages tagged as FinalAnswer proposals
- Keep messages concise and cite sources where relevant
- Vote honestly on FinalAnswer proposals based on your expertise

Protocol
1) Read only messages addressed to you (@<name>), @all, or in threads you follow.
   Mentions generate a notification ping; do not auto-follow — decide yourself.
2) Prefer working inside threads (default #main). Create new threads for subtopics (#<topic>).
3) Keep messages concise and actionable.
4) To propose a solution, follow FORMATTING_RULES exactly:
   - Answer must be exactly "NUMBER unit" (e.g. "29 km/h")
   - No explanatory text or period at end
   - Keep same units as question
   - Be consistent with other agents
5) When you see a proposal, evaluate it and reply with strict JSON:
   {"vote":"yes|no","confidence":0.0-1.0,"rationale":"...","edits":"<optional>"}
6) Use tools deliberately; include only minimal outputs and sources.
7) Never leak secrets or tokens. Sandbox untrusted code/data.
""".strip()

CODE_AGENT_ADDON = """
Role: write, run, and test Python.
- Write small, testable functions with docstrings.
- Produce a 30–60s smoke test after nontrivial changes and show output.
- List any files you create/modify.
""".strip()

RESEARCH_AGENT_ADDON = """
Role: fast reconnaissance and careful synthesis.
- Do fast triage first (3–5 bullets), then deepen where it matters.
- Prefer primary sources; note contradictions explicitly.
- Don’t propose a final answer until two independent supports per key claim.
""".strip()

DOC_AGENT_ADDON = """
Role: document analysis specialist.
- Extract structure (TOC), key passages with offsets, and any formulas or values.
- Be precise and keep citations with page/section references when possible.
""".strip()

DEEP_RESEARCH_AGENT_ADDON = """
Role: deep research and exploration of hypotheses.
- Conduct thorough investigations and synthesize findings.
- Explore multiple hypotheses and perspectives.
- Collaborate with other agents to refine and validate claims.
""".strip()


# ---------------End of prompts----------------


@dataclass
class AgentConfig:
    name: str
    role: str
    tools: List[Tool]
    system_prompt: str
    model: Any  # Model instance
    max_turns: int = 20
    concurrency_limit: int = 1
    keywords: List[str] = field(default_factory=list)
    message_history: set = field(default_factory=set)  # Track previously seen messages


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
        self.last_seen_ts: Optional[str] = None
        self.subscriptions = {"threads": ["main"], "auto_follow_on_mention": False}

        self.agent = ToolCallingAgent(
            tools=config.tools, model=self.config.model, instructions=config.system_prompt, max_steps=20
        )

        # Thread pool for parallel tool execution
        self.executor = ThreadPoolExecutor(max_workers=config.concurrency_limit)

    def check_messages(self) -> List[Dict]:
        """Check for new messages."""
        messages = self.message_store.get_messages(agent_id=self.config.name, last_seen_ts=self.last_seen_ts)
        if messages:
            self.last_seen_ts = messages[-1]["timestamp"]
        return messages

    def post_message(
        self,
        content: Any,
        recipients: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        msg_type: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = self.message_store.append_message(
            sender=self.config.name,
            content=content,
            recipients=recipients,
            thread_id=thread_id,
            msg_type=msg_type,
            reply_to=reply_to,
        )
        emit_mentions(
            self.message_store, content if isinstance(content, str) else json.dumps(content), msg.get("thread_id")
        )
        return msg

    def vote_on_proposal(self, proposal_id: str, proposal_msg: Dict[str, Any]) -> None:
        """Vote on a final answer proposal."""
        verdict = self.evaluate_proposal(proposal_msg)

        # Properly associate the vote with the proposal_id
        self.message_store.record_vote(
            proposal_id=proposal_id,
            voter=self.config.name,
            vote=verdict["vote"],
            confidence=verdict["confidence"],
            rationale=verdict.get("rationale", ""),
            thread_id=proposal_msg.get("thread_id"),
        )

    def run_step(self) -> bool:
        """Run one step/turn of the agent's loop."""
        if self.turn_count >= self.config.max_turns:
            return False

        # Check messages
        messages = self.check_messages()
        if not messages:
            # If no new messages but we haven't voted on all proposals, find and vote on them
            all_messages = self.message_store.get_messages(agent_id=self.config.name)
            proposals = [m for m in all_messages if m.get("type") == "final_answer_proposal"]
            for proposal in proposals:
                # Check if we already voted
                votes = self.message_store.count_votes(proposal["id"])
                if self.config.name not in votes["voters"]:
                    self.vote_on_proposal(proposal_id=proposal["id"], proposal_msg=proposal)
            time.sleep(0.1)  # Avoid tight polling
            return True

        # Process messages
        for msg in messages:
            msg_content = str(msg.get("content", ""))
            if msg_content in self.config.message_history:
                continue  # Skip repeated messages
            self.config.message_history.add(msg_content)

            if msg.get("type") == "final_answer_proposal":
                # Handle voting on proposals
                votes = self.message_store.count_votes(msg["id"])
                if self.config.name not in votes["voters"]:
                    self.vote_on_proposal(proposal_id=msg["id"], proposal_msg=msg)
            elif msg.get("type") == "vote" and msg.get("content", {}).get("vote") == "no":
                # If a proposal was rejected, consider making a new one
                prompt = (
                    f"A proposal was rejected with this rationale: {msg['content'].get('rationale', '')}\n"
                    "Based on all available information, either:\n"
                    "1. Make a new proposal with stronger evidence if you disagree with the rejection\n"
                    "2. Provide additional information to help reach consensus\n\n"
                    "If making a new proposal, format as JSON:\n"
                    '{"type": "final_answer_proposal", "content": "answer", '
                    '"confidence": 0.0-1.0, "evidence": "sources/reasoning"}'
                    "3. If you think the proposal is still valid, explain why it should be reconsidered."
                )
                result = self.agent.run(prompt)
                if result:
                    self.process_agent_response(result)
            else:
                # Regular message processing
                prompt = (
                    f"Message from {msg['sender']}: {msg['content']}\n\n"
                    "Options:\n"
                    "1. Make a final answer proposal if confident (with evidence)\n"
                    "2. Point out conflicting information\n"
                    "3. Ask specific questions\n"
                    "4. Share relevant new information\n\n"
                    "For proposals use JSON format:\n"
                    '{"type": "final_answer_proposal", "content": "answer", '
                    '"confidence": 0.0-1.0, "evidence": "sources/reasoning"}'
                )
                result = self.agent.run(prompt)
                if result:
                    self.process_agent_response(result)

        self.turn_count += 1
        return True

    def process_agent_response(self, result: Any):
        """Process an agent's response and determine if it's a proposal or regular message."""
        try:
            result_str = str(result)
            data = json.loads(result_str)
            if isinstance(data, dict) and data.get("type") == "final_answer_proposal":
                self.post_message(content=data, msg_type="final_answer_proposal")
            else:
                self.post_message(result_str)
        except json.JSONDecodeError:
            self.post_message(str(result))

        self.turn_count += 1
        return True

    def run(self):
        """Run the agent's main loop."""
        while self.run_step():
            pass

    # --- polling & processing ---

    def poll(self) -> List[Dict[str, Any]]:
        msgs = self.message_store.get_messages(agent_id=self.config.name, last_seen_ts=self.last_seen_ts)
        if msgs:
            self.last_seen_ts = msgs[-1]["timestamp"]
        return msgs

    def step(self) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for msg in self.poll():
            out = self.on_message(msg)
            if isinstance(out, dict):
                outputs.append(out)
            elif isinstance(out, list):
                outputs.extend(out)
        return outputs

    # --- handlers ---

    def on_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        t = msg.get("type")
        if t == "final_answer_proposal":
            verdict = self.evaluate_proposal(msg)
            self.message_store.record_vote(
                proposal_id=msg["id"],
                voter=self.config.name,
                vote=verdict["vote"],
                confidence=verdict["confidence"],
                rationale=verdict.get("rationale", ""),
                thread_id=msg.get("thread_id"),
            )
        elif t == "mention":
            self.handle_mention(msg)
        # Optionally implement respond_if_useful for other messages
        return None

    def handle_mention(self, msg: Dict[str, Any]):
        thread = msg.get("thread_id")
        if thread and self.should_follow_thread(thread, msg):
            if thread not in self.subscriptions["threads"]:
                self.subscriptions["threads"].append(thread)
                self.post_message(f"{self.config.name} is now following #{thread}.", thread_id=thread)

    def should_follow_thread(self, thread: str, msg: Dict[str, Any]) -> bool:
        text = json.dumps(msg, ensure_ascii=False)
        interest = any(k.lower() in text.lower() for k in self.config.keywords)
        explicit = f"@{self.config.name}".lower() in text.lower()
        return explicit or interest

    # --- proposal evaluation ---

    def evaluate_proposal(self, proposal_msg: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a final answer proposal and return a vote with confidence and rationale."""
        # Extract the actual proposal content, handling both direct and nested formats
        proposal_content = proposal_msg.get("content", {})
        if isinstance(proposal_content, str):
            try:
                proposal_content = json.loads(proposal_content)
            except json.JSONDecodeError:
                return {
                    "vote": "no",
                    "confidence": 0.0,
                    "rationale": "Invalid proposal format: content is not valid JSON",
                }

        prompt = (
            "Evaluate this FINAL_ANSWER_PROPOSAL based on:\n"
            "1. Accuracy and completeness of the answer\n"
            "2. Quality and reliability of evidence\n"
            "3. Alignment with known facts\n"
            "4. Clarity and precision\n\n"
            f"Proposal: {json.dumps(proposal_content, ensure_ascii=False)}\n\n"
            "Return strictly formatted JSON only:\n"
            '{"vote":"yes|no", "confidence":0.0-1.0, "rationale":"detailed reason for your vote"}'
        )

        raw = self.agent.run(prompt)
        try:
            data = json.loads(str(raw))
            vote = data.get("vote", "no")
            if vote not in ("yes", "no"):
                vote = "no"

            try:
                conf = float(data.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            rationale = str(data.get("rationale", "No rationale provided"))

            return {"vote": vote, "confidence": conf, "rationale": rationale}
        except Exception as e:
            return {"vote": "no", "confidence": 0.0, "rationale": f"Failed to evaluate proposal: {str(e)}"}


def create_team(
    message_store: MessageStore, model_type: str, model_id: str, provider: Optional[str] = None
) -> List[DecentralizedAgent]:
    """Create the team of specialized agents."""

    base_prompt = TEAM_CHARTER

    # Initialize shared components
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    # Initialize model
    model_params: dict[str, Any] = {
        "model_id": model_id,
        "custom_role_conversions": custom_role_conversions,
    }
    if model_id == "o1":
        model_params["reasoning_effort"] = "high"
        model_params["max_completion_tokens"] = 8192
    else:
        model_params["max_tokens"] = 4096

    model = LiteLLMModel(**model_params)

    ti_tool = TextInspectorTool(model, text_limit)

    # Create directory for browser downloads if needed
    os.makedirs(BROWSER_CONFIG["downloads_folder"], exist_ok=True)

    # Create base tools that are shared between code and web tools
    shared_tools = [visualizer, ti_tool]

    # Base web tools
    web_tools = [
        GoogleSearchTool(provider="serper"),
        VisitTool(browser),
        PageUpTool(browser),
        PageDownTool(browser),
        FinderTool(browser),
        FindNextTool(browser),
        ArchiveSearchTool(browser),
    ]

    # Base code tools
    code_tools: List[Tool] = [PythonInterpreterTool()]

    # Add shared tools to both collections
    code_tools.extend(shared_tools)
    web_tools.extend(shared_tools)

    reader_tools = [
        Tool.from_code("""
from smolagents.tools import Tool
class FileReaderTool(Tool):
    '''Tool for reading file contents'''
    name = "file_reader"
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Path to the file to read"
        }
    }
    output_type = "string"
    description = "Read contents of a file"
    def forward(self, file_path: str) -> str:
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
            system_prompt=base_prompt + "\n" + CODE_AGENT_ADDON,
            keywords=["code", "python", "execution"],
        ),
        AgentConfig(
            name="WebSearchAgent",
            role="Fast web research specialist",
            tools=web_tools,
            model=model,
            system_prompt=base_prompt + "\n" + RESEARCH_AGENT_ADDON,
        ),
        AgentConfig(
            name="DeepResearchAgent",
            role="Deep analysis specialist",
            tools=list(set(web_tools + code_tools)),
            model=model,
            system_prompt=base_prompt + "\n" + DEEP_RESEARCH_AGENT_ADDON,
            keywords=["research", "deep", "analysis", "hypothesis"],
        ),
        AgentConfig(
            name="DocumentReaderAgent",
            role="Document analysis specialist",
            tools=reader_tools,
            model=model,
            system_prompt=base_prompt + "\n" + DOC_AGENT_ADDON,
            keywords=["document", "pdf", "extract", "page"],
        ),
    ]

    return [
        DecentralizedAgent(
            config=config, message_store=message_store, model_type=model_type, model_id=model_id, provider=provider
        )
        for config in configs
    ]


# --------------------------- Mentions & parsing -----------------------------


def emit_mentions(message_store: MessageStore, text: str, thread_id: Optional[str]):
    for m in re.findall(r"@([A-Za-z0-9_-]+)", text or ""):
        message_store.append_message(
            sender="system",
            recipients=[m],
            thread_id=thread_id or "main",
            msg_type="mention",
            content={"note": f"You were mentioned in #{thread_id or 'main'}.", "original_text": text},
        )
