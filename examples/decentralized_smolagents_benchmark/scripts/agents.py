"""Agent definitions and utilities for decentralized team."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


try:
    from PIL import Image as PIL
except ImportError:
    PIL = None

from scripts.text_inspector_tool import FileReaderTool, TextInspectorTool
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
from smolagents import CodeAgent, GoogleSearchTool, LiteLLMModel, ToolCallingAgent
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import Tool

from .decentralized_tools import create_decentralized_tools
from .message_store import MessageStore


# evaluation roles
custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}

script_dir = Path(__file__).parent.parent

# Browser configuration
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
BROWSER_CONFIG = {
    "viewport_size": 1024 * 5,
    "downloads_folder": script_dir / "downloads_folder",
    "request_kwargs": {
        "headers": {"User-Agent": user_agent},
        "timeout": 300,
    },
    "serpapi_key": os.getenv("SERPAPI_API_KEY"),
}

# ------------------------ Prompts (Team Charter) ---------------------------

TEAM_CHARTER = """
You are a specialized agent working as part of a decentralized team composed of 4 agents collaborating to solve problems.
CRITICAL: You must regularly use the communication tools provided to stay connected with your team. Check messages and notifications frequently!

COMMUNICATION TOOLS AVAILABLE:
1. send_message_to_agent: Send private messages to specific agents
2. send_message_to_channel: Post messages to team channels/threads
3. read_messages: Check all your messages and mentions
4. read_notifications: Check notifications including polls needing votes
5. search_messages: Search through message history
6. create_general_poll: Create polls for team decisions
7. create_final_answer_poll: Propose final answers for consensus
8. vote_on_poll: Vote on active polls
9. view_active_polls: See what polls are currently active

COMMUNICATION CHANNELS:
1. Public Messages (Message Store):
   - Post in the main thread or create topic-specific threads (#topic)
   - Mention others with @name to get their attention
   - Use @all for team-wide announcements

2. Direct Messages:
   - Use send_message tool for private agent-to-agent communication. You are encouraged to communicate with other agents to get help from them.
   - Valid target_ids: 0=CodeAgent, 1=WebSearchAgent, 2=DeepResearchAgent, 3=DocumentReaderAgent
   - Use receive_messages tool to check your private messages
   - Useful for detailed discussions or sharing sensitive information

3. Threads:
   - Default thread is #main
   - Create new threads for subtopics (#analysis, #research, etc.)
   - Follow relevant threads to stay updated

COLLABORATION PROTOCOL:
1. When receiving a task:
   - FIRST: Check notifications and messages to see team status
   - Share your initial thoughts in #main channel
   - Break down complex problems into subtasks
   - Create relevant threads for different aspects

2. During Investigation:
   - Check messages frequently to see team updates
   - Share findings and progress regularly via messaging
   - Ask questions and request help when needed
   - Combine private and public messages strategically
   - Offer assistance to teammates
   - Respond to messages and mentions promptly

3. For Proposals:
   - Use create_general_poll to propose intermediate solutions
   - Use create_final_answer_poll to propose final answers that would be sent to the user
   - CRITICAL: Check view_active_polls first - only one active poll allowed at a time!
   - If a poll exists, vote on it using vote_on_poll - DO NOT create competing polls
   - Polls with 2+ NO votes are automatically deleted to allow new proposals
   - Always include evidence and reasoning in proposals
   - Required votes for consensus: floor(N/2)+1 = 3

4. Voting Guidelines:
   - Check notifications regularly for polls needing votes
   - Vote honestly based on your expertise using vote_on_poll
   - Include confidence level (0.0-1.0) and rationale
   - Suggest improvements if voting NO

5. Best Practices:
   - Keep messages concise and cite sources
   - Use thread-specific discussions
   - Be proactive in seeking consensus. Therefore, answer to polls promptly.
   - Use tools deliberately; include only minimal outputs and sources.
   - Use messaging tools proactively - the team relies on communication!
   - Combine expertise with other agents through discussion
   - Be responsive to team communications
   - Never leak secrets or tokens. Sandbox untrusted code/data.

COMMUNICATION APPROACH:
- Use send_message_to_channel to share deep analysis in #analysis threads
- Use send_message_to_agent for complex hypothesis discussions
- Collaborate with all agents to develop comprehensive understanding
- Lead consensus-building through thoughtful poll creation and voting
- Start each task by checking your notifications with read_notifications
- Use read_messages regularly to stay updated on team discussions, after having looked at notifications
- Send messages to collaborate: use send_message_to_agent for private discussions, send_message_to_channel for team communication (with context and more than one agent, but not cessarely all the agents)
- When you have findings or need input, share them through messaging
- Before making decisions, you can create polls using create_general_poll
- Vote on polls promptly when you see them in notifications
- For final answers, use create_final_answer_poll to get team consensus

KEY COMMUNICATION PATTERNS:
1. Use read_notifications to gather context from other agents' work
2. Share comprehensive analysis with send_message_to_channel in #analysis
3. Use send_message_to_agent for deep technical and hypothesis discussions
4. Create well-reasoned polls with create_general_poll for major findings
5. Vote thoughtfully on polls using vote_on_poll with detailed rationale

Remember: A solution is only accepted when a majority agrees (3/4 agents). Work together to reach consensus!

Finally, make sure that you respect the required format of answer, when you propose one.

Your success depends on active communication. Use the messaging and notification tools regularly to coordinate with your team!
""".strip()

CODE_AGENT_ADDON = """
ROLE: Python Code Execution Specialist
Primary Responsibilities:
- Write, test, and execute Python code
- Create small, testable functions with docstrings
- Run smoke tests and validate changes
- Handle code-related questions and implementation

COLLABORATION PATTERNS:
1. With Research Agent (@WebSearchAgent):
   - Receive algorithm suggestions and requirements
   - Implement solutions based on their research
   - Share code execution results

2. With Doc Agent (@DocumentReaderAgent):
   - Get specifications and requirements
   - Implement documented patterns
   - Help validate technical documentation

3. With Deep Research Agent (@DeepResearchAgent):
   - Implement complex algorithms
   - Run experiments and benchmarks
   - Validate hypotheses through code

COMMUNICATION:
- Create #implementation threads for coding tasks
- Use private messages for detailed technical discussions
- Share code outputs in public threads for review
- Tag @all for major implementation decisions
""".strip()


RESEARCH_AGENT_ADDON = """
ROLE: Fast Web Research Specialist
Primary Responsibilities:
- Quick information gathering and triage
- Web search and source evaluation
- Initial hypothesis formation
- Fact verification and cross-referencing

COLLABORATION PATTERNS:
1. With Code Agent (@CodeAgent):
   - Share algorithm ideas and implementations
   - Provide real-world examples and use cases
   - Verify technical information

2. With Doc Agent (@DocumentReaderAgent):
   - Cross-reference findings with documentation
   - Validate technical specifications
   - Share relevant external resources

3. With Deep Research Agent (@DeepResearchAgent):
   - Share initial findings for deeper analysis
   - Collaborate on hypothesis validation
   - Cross-verify conclusions

COMMUNICATION:
- Create #research threads for new topics
- Share quick findings in #main
- Use private messages for hypothesis discussion
- Tag @all for significant discoveries
""".strip()

DOC_AGENT_ADDON = """
ROLE: Document Analysis Specialist
Primary Responsibilities:
- Analyze and structure documentation
- Extract key information and references
- Maintain precise citations and sources
- Track technical specifications

COLLABORATION PATTERNS:
1. With Code Agent (@CodeAgent):
   - Share technical specifications
   - Validate implementation against documentation
   - Track API changes and requirements

2. With Research Agent (@WebSearchAgent):
   - Compare documentation with external sources
   - Validate technical claims
   - Share relevant documentation sections

3. With Deep Research Agent (@DeepResearchAgent):
   - Provide detailed technical background
   - Support hypothesis validation
   - Share comprehensive documentation analysis

COMMUNICATION:
- Maintain #documentation thread
- Create topic-specific threads for major docs
- Use private messages for detailed analysis
- Tag @all for critical documentation updates
""".strip()  # Dire que c'est l'id. Etre plus clair dans la description. Voir les agents dans les propriétés l'agent qu'on réinjecte dans le system prompt.

DEEP_RESEARCH_AGENT_ADDON = """
ROLE: Deep Analysis and Research Specialist
Primary Responsibilities:
- Conduct thorough investigations
- Develop and test complex hypotheses
- Synthesize information from multiple sources
- Validate conclusions rigorously

COLLABORATION PATTERNS:
1. With Code Agent (@CodeAgent):
   - Design complex experiments
   - Analyze implementation approaches
   - Validate results mathematically

2. With Research Agent (@WebSearchAgent):
   - Expand on initial research findings
   - Develop comprehensive analysis
   - Cross-validate information sources

3. With Doc Agent (@DocumentReaderAgent):
   - Deep dive into technical documentation
   - Analyze architectural decisions
   - Validate against specifications

COMMUNICATION:
- Maintain #analysis thread for deep dives
- Create hypothesis-specific threads
- Use private messages for complex discussions
- Tag @all for major research findings
""".strip()


# TODO: mettre les descriptions des tools comme dans chacun des autres tools dans default_tools.
# Mettre les notifications
# Post and send messages sous la forme de tools.
# Ou créer des conversations ids. Et envoyer à la conversation, avec SendMessageTools. 2 tools: SendMessageToChannel (liste agent id + main channel + titre channel) et SendMessageToAgent.
# Utiliser un poll différent pour les pols et ne pas hériter de FinalAnswerTool. Ne plus l'utiliser et arrêter les agents lorsque c'est fini. Le but est aussi d'avoir une trace complète.

# ---------------End of prompts----------------


class DecentralizedToolCallingAgent(ToolCallingAgent):
    """ToolCallingAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name


class DecentralizedCodeAgent(CodeAgent):
    """CodeAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name

@dataclass
class AgentConfig:
    name: str
    role: str
    tools: List[Tool]
    system_prompt: str
    model: Any  # Model instance
    max_turns: int = 20
    keywords: List[str] = field(default_factory=list)
    agent_type: str = "tool_calling"  # "tool_calling" or "code"


class DecentralizedAgent:
    """Minimal wrapper around vanilla smolagents that provides decentralized tools."""

    def __init__(
        self,
        config: "AgentConfig",
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
    ):
        self.config = config
        self.message_store = message_store

        # Create the appropriate vanilla smolagents agent
        # The LLM will decide when and how to use the communication tools
        if config.agent_type == "code":
            self.agent = CodeAgent(
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )
        else:
            self.agent = ToolCallingAgent(
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )

    def run(self, task: str, **kwargs):
        """Run the agent on a task - delegate to the vanilla smolagents."""
        return self.agent.run(task, **kwargs)

    def step(self) -> List[Dict[str, Any]]:
        """Single step interface for compatibility."""
        return []


class DecentralizedAgents:
    """Main orchestrator class that combines all the created agents of the team."""

    def __init__(
        self,
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
        run_id: Optional[str] = None
    ):
        """Initialize the decentralized agent team.

        Args:
            message_store: The message store for agent communication
            model_type: Type of model to use (e.g., "LiteLLMModel")
            model_id: Model ID to use (e.g., "gpt-4o")
            provider: Model provider (e.g., "openai")
            run_id: Optional run ID for tracking
        """
        self.message_store = message_store
        self.model_type = model_type
        self.model_id = model_id
        self.provider = provider
        self.run_id = run_id or "default"

        # Create the team of agents
        self.agents = self._create_decentralized_team()

        # Track execution state
        self.results = []
        self.consensus_reached = False
        self.final_answer = None

        # Log team creation
        agent_info = []
        for agent in self.agents:
            agent_details = {"name": agent.config.name, "role": agent.config.role}
            agent_info.append(agent_details)

        logging.info(json.dumps({
            "event": "decentralized_team_created",
            "run_id": self.run_id,
            "agent_count": len(self.agents),
            "agents": agent_info
        }))

    def _create_decentralized_team(self) -> List[DecentralizedAgent]:
        """Create the team of specialized agents."""
        base_prompt = TEAM_CHARTER

        # Initialize shared components
        text_limit = 100000
        browser = SimpleTextBrowser(**BROWSER_CONFIG)

        # Initialize model
        model_params: dict[str, Any] = {
            "model_id": self.model_id,
            "custom_role_conversions": custom_role_conversions,
        }

        model_params["max_tokens"] = 8192

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

        # Use simple file reader tool
        reader_tools: List[Tool] = [FileReaderTool()]

        # Add shared tools to both collections
        code_tools.extend(shared_tools)
        web_tools.extend(shared_tools)
        reader_tools.extend(shared_tools)

        # Add decentralized tools to each agent's tool set
        code_agent_tools = code_tools + create_decentralized_tools(self.message_store, "CodeAgent")
        web_agent_tools = web_tools + create_decentralized_tools(self.message_store, "WebSearchAgent")
        deep_agent_tools = list(set(web_tools + code_tools)) + create_decentralized_tools(
            self.message_store, "DeepResearchAgent"
        )
        doc_agent_tools = reader_tools + create_decentralized_tools(self.message_store, "DocumentReaderAgent")

        configs = [
            AgentConfig(
                name="CodeAgent",
                role="Python code execution specialist",
                tools=code_agent_tools,
                model=model,
                system_prompt=base_prompt + "\n" + CODE_AGENT_ADDON,
                keywords=["code", "python", "execution"],
                agent_type="code",
            ),
            AgentConfig(
                name="WebSearchAgent",
                role="Fast web research specialist",
                tools=web_agent_tools,
                model=model,
                system_prompt=base_prompt + "\n" + RESEARCH_AGENT_ADDON,
                agent_type="tool_calling",
            ),
            AgentConfig(
                name="DeepResearchAgent",
                role="Deep analysis specialist",
                tools=deep_agent_tools,
                model=model,
                system_prompt=base_prompt + "\n" + DEEP_RESEARCH_AGENT_ADDON,
                keywords=["research", "deep", "analysis", "hypothesis"],
                agent_type="code",
            ),
            AgentConfig(
                name="DocumentReaderAgent",
                role="Document analysis specialist",
                tools=doc_agent_tools,
                model=model,
                system_prompt=base_prompt + "\n" + DOC_AGENT_ADDON,
                keywords=["document", "pdf", "extract", "page"],
                agent_type="tool_calling",
            ),
        ]

        return [
            DecentralizedAgent(
                config=config,
                message_store=self.message_store,
                model_type=self.model_type,
                model_id=self.model_id,
                provider=self.provider,
            )
            for config in configs
        ]


    def run(self, task: str) -> Dict[str, Any]:
        """Run the entire decentralized agent team on a task.

        Args:
            task: The task/question to solve

        Returns:
            Dict containing the result with status, answer, and metadata
        """
        print(f"🚀 Starting decentralized agent team for: {task}")
        logging.info(json.dumps({
            "event": "team_execution_started",
            "run_id": self.run_id,
            "task": task
        }))

        # Post initial task to message store
        initial_msg = self.message_store.append_message(
            sender="system", content=task, thread_id="main", msg_type="task"
        )
        print("📝 Posted initial task to message store")
        logging.info(json.dumps({
            "event": "task_posted",
            "run_id": self.run_id,
            "message_id": initial_msg.get("id"),
            "task": task
        }))

        # Print team info
        print(f"👥 Created team of {len(self.agents)} agents:")
        for agent in self.agents:
            print(f"  - {agent.config.name} ({agent.config.role})")

        print("\n💬 Running agents in parallel with collaborative approach...")

        # Prepare collaborative task prompt with enhanced instructions
        QUESTION_ADDON = """
IMPORTANT: Before answering, please:
1. Use read_notifications to check if there are any ongoing team discussions or polls
2. Use view_active_polls to see if there are any polls you should vote on
3. Use read_messages to see what other agents have contributed
4. If there's an active poll about the final answer, vote on it using vote_on_poll
5. If no poll exists yet and you're confident in an answer, create a final answer poll using create_final_answer_poll

Work collaboratively with your team!"""

        collaborative_task = f"{task}\n\n{QUESTION_ADDON}"

        # Run all agents in parallel
        self.results = []
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self._run_single_agent, agent, collaborative_task, i): agent
                for i, agent in enumerate(self.agents)
            }

            # Process completed agents and check for early consensus
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    agent_result = future.result()
                    self.results.append(agent_result)

                    print(f"✅ {agent.config.name} completed ({len(self.results)}/{len(self.agents)} done)")

                    # Check for early consensus after each completion
                    if len(self.results) >= 2:  # Need at least 2 agents for meaningful consensus
                        consensus_result = self._check_for_consensus()
                        if consensus_result:
                            print(f"\n🎯 Consensus reached early after {agent.config.name}: {consensus_result}")
                            logging.info(json.dumps({
                                "event": "consensus_reached_early",
                                "run_id": self.run_id,
                                "after_agent": agent.config.name,
                                "completed_agents": len(self.results),
                                "final_answer": str(consensus_result)[:200]
                            }))

                            # Cancel remaining tasks (best effort)
                            for remaining_future in future_to_agent:
                                if not remaining_future.done():
                                    remaining_future.cancel()

                            self.consensus_reached = True
                            self.final_answer = consensus_result
                            total_duration = time.time() - start_time
                            return self._create_result_summary("success_early", total_duration)

                except Exception as e:
                    print(f"❌ {agent.config.name} failed: {e}")
                    logging.error(json.dumps({
                        "event": "agent_failed_in_parallel",
                        "run_id": self.run_id,
                        "agent_name": agent.config.name,
                        "error": str(e)
                    }))
                    # Continue with other agents

        # Final consensus check
        logging.info(json.dumps({"event": "final_consensus_check_started", "run_id": self.run_id}))
        final_consensus = self._check_for_consensus()

        total_duration = time.time() - start_time

        if final_consensus:
            print(f"\n🎯 Final consensus reached: {final_consensus}")
            self.consensus_reached = True
            self.final_answer = final_consensus
            return self._create_result_summary("success", total_duration)
        else:
            # Fallback to last valid result
            return self._create_fallback_result(total_duration)

    def _run_single_agent(self, agent: DecentralizedAgent, task: str, index: int) -> Dict[str, Any]:
        """Run a single agent and return result metadata."""
        print(f"\n🤖 Running {agent.config.name}...")
        logging.info(json.dumps({
            "event": "agent_started",
            "run_id": self.run_id,
            "agent_name": agent.config.name,
            "agent_index": index
        }))

        start_time = time.time()

        try:
            result = agent.run(task)
            end_time = time.time()
            duration = end_time - start_time

            print(f"✅ {agent.config.name} completed")
            logging.info(json.dumps({
                "event": "agent_completed",
                "run_id": self.run_id,
                "agent_name": agent.config.name,
                "duration_seconds": round(duration, 2),
                "result_type": type(result).__name__,
                "result_preview": str(result)[:200] if result else None
            }))

            return {
                "agent_name": agent.config.name,
                "status": "success",
                "result": result,
                "duration": duration,
                "error": None
            }

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            print(f"❌ {agent.config.name} failed: {e}")
            logging.error(json.dumps({
                "event": "agent_failed",
                "run_id": self.run_id,
                "agent_name": agent.config.name,
                "error": str(e),
                "error_type": type(e).__name__,
                "duration_seconds": round(duration, 2)
            }))

            return {
                "agent_name": agent.config.name,
                "status": "error",
                "result": f"Error: {e}",
                "duration": duration,
                "error": str(e)
            }

    def _check_for_consensus(self) -> Optional[str]:
        """Check if agents reached consensus through polling."""
        logging.info(json.dumps({"event": "consensus_check_started", "run_id": self.run_id}))

        try:
            # Look for finalized polls with final answers
            all_messages = list(self.message_store._iter_messages())
            logging.info(json.dumps({
                "event": "messages_retrieved",
                "run_id": self.run_id,
                "message_count": len(all_messages)
            }))

            for msg in all_messages:
                if isinstance(msg, dict):
                    content = msg.get("content", {})
                    if isinstance(content, dict) and content.get("type") == "final_answer":
                        answer = content.get("answer", "")
                        logging.info(json.dumps({
                            "event": "final_answer_found",
                            "run_id": self.run_id,
                            "message_id": msg.get("id"),
                            "answer": str(answer)[:200]
                        }))
                        return answer

            # Check for active polls that might need finalization
            active_polls = self.message_store.get_active_polls()
            print(f"🔍 Found {len(active_polls)} active polls")
            logging.info(json.dumps({
                "event": "active_polls_check",
                "run_id": self.run_id,
                "poll_count": len(active_polls)
            }))

            for poll in active_polls:
                if isinstance(poll, dict):
                    poll_id = poll.get("poll_id")
                    print(f"🗳️ Checking poll {poll_id}")
                    logging.info(json.dumps({
                        "event": "poll_check_started",
                        "run_id": self.run_id,
                        "poll_id": poll_id,
                        "poll_question": poll.get("question", "")
                    }))

                    if poll_id:
                        # Try to finalize the poll
                        result = self.message_store.finalize_poll_if_ready(poll_id)
                        print(f"📊 Poll finalization result: {result}")
                        logging.info(json.dumps({
                            "event": "poll_finalization_attempted",
                            "run_id": self.run_id,
                            "poll_id": poll_id,
                            "result": str(result)[:200] if result else None
                        }))

                        if result and not result.get("deleted"):
                            answer_content = result.get("content", {})
                            if isinstance(answer_content, dict):
                                answer = answer_content.get("answer", "")
                                print(f"✅ Consensus reached: {answer}")
                                logging.info(json.dumps({
                                    "event": "consensus_reached",
                                    "run_id": self.run_id,
                                    "poll_id": poll_id,
                                    "answer": str(answer)[:200]
                                }))
                                return answer
                        elif result and result.get("deleted"):
                            print(f"🗑️ Poll {poll_id} was deleted due to insufficient support")
                            logging.info(json.dumps({
                                "event": "poll_deleted",
                                "run_id": self.run_id,
                                "poll_id": poll_id,
                                "reason": "insufficient_support"
                            }))

        except Exception as e:
            print(f"⚠️ Error checking consensus: {e}")
            logging.error(json.dumps({
                "event": "consensus_check_error",
                "run_id": self.run_id,
                "error": str(e),
                "error_type": type(e).__name__
            }))

        logging.info(json.dumps({
            "event": "consensus_check_completed",
            "run_id": self.run_id,
            "result": "no_consensus"
        }))
        return None

    def _create_result_summary(self, status: str, duration: float) -> Dict[str, Any]:
        """Create a comprehensive result summary."""
        logging.info(json.dumps({
            "event": "run_completed",
            "run_id": self.run_id,
            "status": status,
            "duration_seconds": round(duration, 2)
        }))

        return {
            "status": status,
            "answer": self.final_answer,
            "consensus_reached": self.consensus_reached,
            "total_duration": round(duration, 2),
            "agent_results": self.results,
            "run_id": self.run_id,
            "agent_count": len(self.agents),
            "successful_agents": len([r for r in self.results if r.get("status") == "success"]),
            "failed_agents": len([r for r in self.results if r.get("status") == "error"])
        }

    def _create_fallback_result(self, duration: float) -> Dict[str, Any]:
        """Create result when no consensus is reached."""
        # Fallback to last valid result
        valid_results = [r for r in self.results if r.get("status") == "success"]

        if valid_results:
            answer = valid_results[-1]["result"]  # Use the last valid result
            print(f"\n📝 Final result (fallback): {answer}")
            logging.info(json.dumps({
                "event": "fallback_result_used",
                "run_id": self.run_id,
                "result": str(answer)[:200],
                "valid_result_count": len(valid_results)
            }))

            self.final_answer = str(answer)
            return self._create_result_summary("success_fallback", duration)
        else:
            print("\n❌ No valid results obtained")
            logging.error(json.dumps({
                "event": "no_valid_results",
                "run_id": self.run_id,
                "total_results": len(self.results),
                "error_results": len([r for r in self.results if r.get("status") == "error"])
            }))

            self.final_answer = None
            return self._create_result_summary("failure", duration)

    def get_agent_by_name(self, name: str) -> Optional[DecentralizedAgent]:
        """Get an agent by name."""
        for agent in self.agents:
            if agent.config.name == name:
                return agent
        return None

    def get_results(self) -> Dict[str, Any]:
        """Get the current execution results."""
        return {
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "results": self.results,
            "run_id": self.run_id
        }

    @property
    def agent_names(self) -> List[str]:
        """Get list of agent names."""
        return [agent.config.name for agent in self.agents]

    @property
    def agent_count(self) -> int:
        """Get number of agents in the team."""
        return len(self.agents)
