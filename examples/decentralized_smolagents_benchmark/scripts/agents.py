"""Agent definitions and utilities for decentralized team."""

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

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
from smolagents import CodeAgent, GoogleSearchTool, LiteLLMModel, ToolCallingAgent
from smolagents.default_tools import FinalAnswerTool, PythonInterpreterTool, ReceiveMessagesTool, SendMessageTool
from smolagents.tools import Tool

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
   - Share your initial thoughts in #main
   - Break down complex problems into subtasks
   - Create relevant threads for different aspects

2. During Investigation:
   - Share findings and progress regularly
   - Ask questions and request help when needed
   - Combine private and public messages strategically
   - Check your messages often

3. For Proposals:
   - Use the poll mechanism to propose solutions
   - CRITICAL: Only one active poll is allowed at a time. Check for existing polls first!
   - If a poll exists, vote on it or wait - DO NOT create competing polls
   - Polls with 2+ NO votes are automatically deleted to allow new proposals
   - Format: Use create_poll(question="...", proposal="answer")
   - Always include evidence and reasoning
   - Other agents must vote using record_vote
   - Required votes for consensus: floor(N/2)+1 = 3

4. Voting:
   - Vote honestly based on your expertise
   - Include confidence level (0.0-1.0) and rationale
   - Suggest improvements if voting NO
   - Format: {"vote":"YES/NO", "confidence":0.0-1.0, "rationale":"..."}

5. Best Practices:
   - Keep messages concise and cite sources
   - Use thread-specific discussions
   - Combine expertise with other agents
   - Be proactive in seeking consensus. Therefore, answer to polls promptly.
   - Use tools deliberately; include only minimal outputs and sources.
   - Never leak secrets or tokens. Sandbox untrusted code/data.

Remember: A solution is only accepted when a majority agrees (3/4 agents). Work together to reach consensus!
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
""".strip()

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


# ---------------End of prompts----------------


class PollCreatingFinalAnswerTool(FinalAnswerTool):
    """Custom FinalAnswerTool that creates polls instead of just returning the answer."""

    def __init__(self, agent_name: str, message_store: "MessageStore"):
        super().__init__()
        self.agent_name = agent_name
        self.message_store = message_store

    def forward(self, answer: str) -> str:
        """Create a poll for the final answer instead of returning it directly."""
        # Check for existing active polls first
        active_polls = self.message_store.get_active_polls()
        if active_polls:
            existing_poll = active_polls[0]
            print(f"🚫 {self.agent_name}: Cannot create poll - active poll exists by {existing_poll.get('proposer')}")
            print(f"   Please vote on existing poll ID: {existing_poll.get('poll_id')}")
            return f"Cannot create new proposal - there's already an active poll by {existing_poll.get('proposer')}. Please vote on the existing poll first."

        # Create poll for the final answer
        question = "Do you approve this final answer?"

        print(f"🗳️ Creating poll: {self.agent_name} proposes: {answer[:50]}...")
        result = self.message_store.create_poll(
            question=question, proposal=answer, proposer=self.agent_name, thread_id="main"
        )

        if isinstance(result, dict) and "error" in result:
            return f"Cannot create poll: {result.get('error')}"

        # Return the answer (this won't be the final output since we're creating a poll)
        return answer


class DecentralizedToolCallingAgent(ToolCallingAgent):
    """ToolCallingAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name

        # Replace the default final_answer tool with our poll-creating version
        self.tools["final_answer"] = PollCreatingFinalAnswerTool(agent_name, message_store)


class DecentralizedCodeAgent(CodeAgent):
    """CodeAgent that creates polls for final answers."""

    def __init__(self, message_store: MessageStore, agent_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_store = message_store
        self.agent_name = agent_name

        # Replace the default final_answer tool with our poll-creating version
        self.tools["final_answer"] = PollCreatingFinalAnswerTool(agent_name, message_store)


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
    agent_id: int = 0  # Remove auto-increment, will be set explicitly
    agent_type: str = "tool_calling"  # "tool_calling" or "code"


class DecentralizedAgent:
    """Simplified wrapper around vanilla smolagents that handles messaging and polling."""

    def __init__(
        self,
        config: AgentConfig,
        message_store: MessageStore,
        model_type: str,
        model_id: str,
        provider: Optional[str] = None,
        shared_message_queues: Optional[dict] = None,
    ):
        self.config = config
        self.message_store = message_store
        self.last_seen_msg = None
        self.turn_count = 0
        self.last_seen_ts: Optional[str] = None
        self.subscriptions = {"threads": ["main"], "auto_follow_on_mention": True}
        self._processed_message_ids: set = set()  # Track processed messages by ID instead of content
        self._lock = threading.Lock()  # For thread safety

        # Use shared message queues or create own
        if shared_message_queues:
            self.message_queues = shared_message_queues
        else:
            from queue import Queue

            self.message_queues = {}
            self.message_queue = Queue()
            self.message_queues[config.agent_id] = self.message_queue

        # Add messaging tools
        messaging_tools = [
            SendMessageTool(self.message_queues, config.agent_id),
            ReceiveMessagesTool(self.message_queues, config.agent_id),
        ]
        config.tools.extend(messaging_tools)

        # Create the appropriate vanilla smolagents agent
        if config.agent_type == "code":
            self.agent = DecentralizedCodeAgent(
                message_store=message_store,
                agent_name=config.name,
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )
        else:
            self.agent = DecentralizedToolCallingAgent(
                message_store=message_store,
                agent_name=config.name,
                tools=config.tools,
                model=config.model,
                instructions=config.system_prompt,
                max_steps=20,
            )

        # Thread pool for parallel tool execution
        self.executor = ThreadPoolExecutor(max_workers=config.concurrency_limit)

    def check_messages(self) -> List[Dict]:
        """Check for new messages with thread safety."""
        with self._lock:
            messages = self.message_store.get_messages(
                agent_id=self.config.name, last_seen_ts=self.last_seen_ts, include_mentions=True, include_private=True
            )
            if messages:
                self.last_seen_ts = messages[-1].get("timestamp")
                # Filter out already processed messages
                new_messages = [msg for msg in messages if msg.get("id") not in self._processed_message_ids]
                return new_messages
            return []

    def check_notifications(self) -> Dict[str, List[Dict]]:
        """Check for notifications including mentions, DMs, and polls."""
        return self.message_store.get_notifications(agent_id=self.config.name, since_ts=self.last_seen_ts)

    def search_messages(self, query: str, thread_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Search messages visible to this agent."""
        return self.message_store.search_messages(
            query=query, agent_id=self.config.name, thread_id=thread_id, limit=limit
        )

    def post_message(
        self,
        content: Any,
        recipients: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        msg_type: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Post a message with deduplication."""
        # Use deduplication for regular messages
        if msg_type in ["vote", "final_answer_proposal", "poll"]:
            msg = self.message_store.append_message(
                sender=self.config.name,
                content=content,
                recipients=recipients,
                thread_id=thread_id,
                msg_type=msg_type,
                reply_to=reply_to,
            )
        else:
            msg = self.message_store.append_message_with_dedup(
                sender=self.config.name,
                content=content,
                recipients=recipients,
                thread_id=thread_id,
                msg_type=msg_type,
                reply_to=reply_to,
            )

        if msg:
            # Handle mentions
            content_str = content if isinstance(content, str) else json.dumps(content)
            self.emit_mentions(content_str, msg.get("thread_id"))

        return msg

    def emit_mentions(self, text: str, thread_id: Optional[str]):
        """Handle @mentions in messages."""
        import re

        for match in re.findall(r"@([A-Za-z0-9_-]+)", text or ""):
            # Create a mention notification message
            self.message_store.append_message(
                sender="system",
                content={
                    "type": "mention",
                    "mentioned_agent": match,
                    "original_sender": self.config.name,
                    "original_content": text,
                    "thread_id": thread_id,
                },
                recipients=[match],
                thread_id=thread_id,
                msg_type="mention",
            )

    def vote_on_proposal(self, proposal_id: str, proposal_msg: Dict[str, Any]) -> None:
        """Vote on a final answer proposal."""
        verdict = self.evaluate_proposal(proposal_msg)

        # Properly associate the vote with the proposal_id
        self.message_store.record_vote(
            poll_id=proposal_id,
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

        try:
            # Check for new messages
            messages = self.check_messages()

            # Check for active polls that need votes
            active_polls = self.message_store.get_active_polls()
            print(f"DEBUG {self.config.name}: Found {len(active_polls)} active polls")
            for poll in active_polls:
                poll_id = poll.get("poll_id") if isinstance(poll, dict) else None
                print(f"DEBUG {self.config.name}: Poll type: {type(poll)}")
                print(f"DEBUG {self.config.name}: Poll data: {poll}")
                if poll_id:
                    # Check if we already voted on this poll
                    vote_info = self.message_store.count_votes(poll_id)
                    voters = vote_info.get("votes_by_voter", {})
                    print(f"DEBUG {self.config.name}: Poll {poll_id} has votes from: {list(voters.keys())}")
                    if self.config.name not in voters:
                        print(f"DEBUG {self.config.name}: Found unvoted poll {poll_id} - voting now")
                        # Create a mock poll message for voting
                        mock_poll_msg = {
                            "content": poll,
                            "thread_id": "main",
                            "type": "poll",
                            "sender": poll.get("proposer", "unknown"),
                        }
                        # try:
                        self._handle_poll_vote(mock_poll_msg)
                        # except Exception as e:
                        #     print(f"ERROR {self.config.name}: Failed to vote on poll {poll_id}: {e}")
                        #     print(f"ERROR {self.config.name}: Poll data type: {type(poll)}")
                        #     print(f"ERROR {self.config.name}: Poll data: {poll}")
                        #     import traceback
                        #     traceback.print_exc()
                    else:
                        print(f"DEBUG {self.config.name}: Already voted on poll {poll_id}")

            if not messages:
                # Check for unvoted polls and notifications
                notifications = self.check_notifications()

                # Handle polls needing votes
                for poll_msg in notifications.get("polls_needing_votes", []):
                    self._handle_poll_vote(poll_msg)

                # Brief sleep to avoid tight polling
                time.sleep(0.1)
                self.turn_count += 1
                return True

            # Process new messages
            for msg in messages:
                msg_id = msg.get("id")
                if msg_id and msg_id not in self._processed_message_ids:
                    self._process_message(msg)
                    with self._lock:
                        self._processed_message_ids.add(msg_id)

            self.turn_count += 1
            return True

        except Exception as e:
            print(f"Error in agent {self.config.name}: {e}")
            import traceback

            traceback.print_exc()
            self.turn_count += 1
            return True  # Continue despite errors

    def _process_message(self, msg: Dict[str, Any]) -> None:
        """Process a single message."""
        msg_type = msg.get("type", "message")
        sender = msg.get("sender", "")
        content = msg.get("content", "")

        # Skip own messages
        if sender == self.config.name:
            return

        # Log message processing
        print(f"DEBUG {self.config.name}: Processing {msg_type} from {sender}: {str(content)[:100]}...")

        try:
            if msg_type == "poll":
                self._handle_poll_vote(msg)
            elif msg_type == "mention":
                self._handle_mention(msg)
            elif msg_type == "task":
                # Handle task messages same as regular messages
                self._handle_regular_message(msg)
            elif msg_type in ["message", None]:
                self._handle_regular_message(msg)
        except Exception as e:
            print(f"Error processing message in {self.config.name}: {e}")

    def _handle_poll_vote(self, poll_msg: Dict[str, Any]) -> None:
        """Handle voting on a poll."""
        try:
            poll_content = poll_msg.get("content", {})

            # Handle case where content might be a string (JSON)
            if isinstance(poll_content, str):
                try:
                    poll_content = json.loads(poll_content)
                except Exception as e:
                    print(f"ERROR {self.config.name}: Cannot parse poll content: {poll_content[:100]}... - {e}")
                    return

            poll_id = poll_content.get("poll_id") if isinstance(poll_content, dict) else None

            if not poll_id:
                print(f"ERROR {self.config.name}: No poll_id in poll message: {type(poll_content)} - {poll_content}")
                return

            # Check if we already voted
            vote_info = self.message_store.count_votes(poll_id)
            if self.config.name in vote_info.get("votes_by_voter", {}):
                print(f"DEBUG {self.config.name}: Already voted on poll {poll_id}")
                return  # Already voted

            # Check if poll is still open
            if vote_info.get("closed") or vote_info.get("deleted"):
                print(f"DEBUG {self.config.name}: Poll {poll_id} is closed/deleted")
                return

            # Evaluate and vote
            print(f"DEBUG {self.config.name}: Evaluating poll {poll_id}")
            verdict = self.evaluate_poll(poll_msg)

            self.message_store.record_vote(
                poll_id=poll_id,
                voter=self.config.name,
                vote=verdict["vote"],
                confidence=verdict.get("confidence", 0.5),
                rationale=verdict.get("rationale", ""),
                thread_id=poll_msg.get("thread_id"),
            )

            # Log the vote
            print(
                f"VOTE LOGGED: {self.config.name} voted {verdict['vote']} on poll {poll_id} (confidence: {verdict.get('confidence', 0.5)})"
            )

            # Check if this poll should now be finalized or deleted
            result = self.message_store.finalize_poll_if_ready(poll_id)
            if result:
                if result.get("deleted"):
                    print(f"POLL DELETED: Poll {poll_id} was deleted due to {result.get('reason')}")
                else:
                    # Poll passed
                    print(f"CONSENSUS REACHED: Poll {poll_id} has been finalized!")
        except Exception as e:
            print(f"ERROR {self.config.name}: Exception in _handle_poll_vote: {e}")
            import traceback

            traceback.print_exc()

        # Check vote counts for logging
        updated_vote_info = self.message_store.count_votes(poll_id)
        yes_count = updated_vote_info.get("tally", {}).get("YES", 0)
        no_count = updated_vote_info.get("tally", {}).get("NO", 0)
        total_count = updated_vote_info.get("tally", {}).get("eligible", 4)
        threshold = poll_content.get("threshold", 3)

        print(
            f"POLL STATUS: Poll {poll_id} has {yes_count} YES / {no_count} NO votes (threshold: {threshold} YES needed), total count: {total_count}"
        )

    def _handle_mention(self, msg: Dict[str, Any]) -> None:
        """Handle mention notifications."""
        content = msg.get("content", {})
        thread_id = content.get("thread_id", "main")

        # Follow the thread if auto-follow is enabled
        if self.subscriptions.get("auto_follow_on_mention", True):
            if thread_id not in self.subscriptions.get("threads", []):
                self.subscriptions["threads"].append(thread_id)

        # Generate a response to the mention
        original_content = content.get("original_content", "")
        original_sender = content.get("original_sender", "")

        prompt = (
            f"You were mentioned by {original_sender} in thread '{thread_id}': {original_content}\n\n"
            "Respond appropriately based on your role and expertise. You can:\n"
            "1. Provide relevant information or assistance\n"
            "2. Ask clarifying questions\n"
            "3. Collaborate with other agents\n"
            "4. Make a proposal if you have a solution\n\n"
            "Keep your response focused and helpful."
        )

        result = self.agent.run(prompt)
        if result:
            self.post_message(content=str(result), thread_id=thread_id, reply_to=msg.get("id"))

    def _handle_regular_message(self, msg: Dict[str, Any]) -> None:
        """Handle regular messages."""
        sender = msg.get("sender", "")
        content = msg.get("content", "")
        thread_id = msg.get("thread_id", "main")

        # Check if this message is relevant based on keywords or mentions
        content_str = str(content).lower()
        is_relevant = (
            any(keyword.lower() in content_str for keyword in self.config.keywords)
            or f"@{self.config.name.lower()}" in content_str
            or thread_id in self.subscriptions.get("threads", [])
        )

        if not is_relevant:
            return

        # Generate response using vanilla smolagents - it will handle final_answer tool calls automatically
        prompt = (
            f"Message from {sender} in thread '{thread_id}': {content}\n\n"
            "Based on your expertise and role, decide how to respond:\n"
            "1. Provide helpful information or analysis\n"
            "2. Ask questions to clarify requirements\n"
            "3. Collaborate with other agents using @mentions\n"
            "4. Use the final_answer tool when you have a confident final answer\n\n"
            "Otherwise, provide a helpful response."
        )

        result = self.agent.run(prompt)
        if result:
            # If it's a final answer, the custom tool already created a poll
            # Otherwise, just post the message normally
            self.post_message(str(result), thread_id=thread_id)

    def run(self):
        """Run the agent's main loop."""
        while self.run_step():
            pass

    # --- Enhanced polling & processing with better error handling ---

    def poll(self) -> List[Dict[str, Any]]:
        """Alternative polling method for backward compatibility."""
        return self.check_messages()

    def step(self) -> List[Dict[str, Any]]:
        """Process messages and return outputs."""
        outputs: List[Dict[str, Any]] = []
        messages = self.poll()

        for msg in messages:
            try:
                out = self.on_message(msg)
                if isinstance(out, dict):
                    outputs.append(out)
                elif isinstance(out, list):
                    outputs.extend(out)
            except Exception as e:
                print(f"Error in step processing for {self.config.name}: {e}")

        return outputs

    # --- handlers ---

    def on_message(self, msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle individual messages - simplified version."""
        msg_type = msg.get("type", "message")

        if msg_type == "poll":
            self._handle_poll_vote(msg)
        elif msg_type == "mention":
            self._handle_mention(msg)

        return None  # Most processing is done in the handlers

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

    # === Poll helpers (GENERAL, domain‑agnostic) ===

    def propose_poll(self, proposal: Any, question: str, thread_id: Optional[str] = None):
        return self.message_store.create_poll(
            question=question,
            proposal=proposal,
            proposer=self.config.name,
            thread_id=thread_id or getattr(self, "thread_id", None),
        )

    def evaluate_poll(self, poll_msg: dict) -> dict:
        """Evaluate a poll using the agent's expertise."""

        try:
            c = poll_msg.get("content", {})

            # Handle case where content might be a string (JSON)
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except Exception:
                    print(f"ERROR: Cannot parse poll content in evaluate_poll: {c}")
                    return {"vote": "NO", "confidence": 0.1, "rationale": "Invalid poll format"}

            proposal = c.get("proposal", "")
            question = c.get("question", "")

            # Use the same evaluation logic as evaluate_proposal but simplified
            prompt = (
                f"POLL EVALUATION:\n"
                f"Question: {question}\n"
                f"Proposal: {proposal}\n\n"
                "Based on your expertise, evaluate this proposal:\n"
                "- Is it accurate and complete?\n"
                "- Is it well-reasoned?\n"
                "- Would you approve it?\n\n"
                "Respond with a simple JSON:\n"
                '{"vote":"YES","confidence":0.8,"rationale":"brief reason"}'
            )

            raw = self.agent.run(prompt)
            result_str = str(raw).strip()

            # Try to extract JSON
            import re

            json_match = re.search(r'\{[^}]*"vote"[^}]*\}', result_str, re.IGNORECASE | re.DOTALL)
            if json_match:
                result_str = json_match.group()

            try:
                result = json.loads(result_str)
                vote = str(result.get("vote", "NO")).upper()
                if vote not in ["YES", "NO"]:
                    vote = "NO"

                return {
                    "vote": vote,
                    "confidence": float(result.get("confidence", 0.5)),
                    "rationale": str(result.get("rationale", "Generic evaluation")),
                }
            except (json.JSONDecodeError, ValueError):
                pass

        except Exception as e:
            print(f"Error in evaluate_poll for {self.config.name}: {e}")

        # Fallback: simple heuristic
        c = poll_msg.get("content", {})
        if isinstance(c, str):
            try:
                c = json.loads(c)
            except json.JSONDecodeError:
                c = {}
        prop = c.get("proposal", "")
        ok = prop and str(prop).strip() != ""
        return {
            "vote": "YES" if ok else "NO",
            "confidence": 0.6 if ok else 0.4,
            "rationale": "Fallback evaluation - proposal is non-empty" if ok else "Empty or invalid proposal",
        }

    def vote_on_poll(self, poll_msg: dict):
        c = poll_msg.get("content", {})
        poll_id = c.get("poll_id")
        if not poll_id:
            return
        verdict = self.evaluate_poll(poll_msg)
        self.message_store.record_vote(
            poll_id=poll_id,
            voter=self.config.name,
            vote=verdict["vote"],
            confidence=verdict.get("confidence", 0.5),
            rationale=verdict.get("rationale", ""),
            thread_id=poll_msg.get("thread_id"),
        )

    def _extract_proposal_from_text(self, s: str) -> Optional[str]:
        """Heuristic: squeeze plain text into a proposal (short answers)."""
        if not s:
            return None
        text = str(s).strip()
        # common LLM phrasing
        m = __import__("re").search(r"final answer:\s*(.+)$", text, flags=__import__("re").I)
        if m:
            text = m.group(1).strip()
        text = __import__("re").sub(r"\s+", " ", text)
        return text if 0 < len(text) <= 400 else None


def create_team(
    message_store: MessageStore, model_type: str, model_id: str, provider: Optional[str] = None
) -> List[DecentralizedAgent]:
    """Create the team of specialized agents."""

    base_prompt = TEAM_CHARTER

    # Initialize shared components
    text_limit = 100000
    browser = SimpleTextBrowser(**BROWSER_CONFIG)

    # Create shared message queues for all agents
    from queue import Queue

    shared_message_queues = {}
    agent_names = ["CodeAgent", "WebSearchAgent", "DeepResearchAgent", "DocumentReaderAgent"]
    for i, name in enumerate(agent_names):
        shared_message_queues[i] = Queue()  # Use integer IDs for SendMessageTool compatibility

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
            agent_id=0,  # Assign integer IDs
            agent_type="code",  # Use CodeAgent for code execution
        ),
        AgentConfig(
            name="WebSearchAgent",
            role="Fast web research specialist",
            tools=web_tools,
            model=model,
            system_prompt=base_prompt + "\n" + RESEARCH_AGENT_ADDON,
            agent_id=1,
            agent_type="tool_calling",
        ),
        AgentConfig(
            name="DeepResearchAgent",
            role="Deep analysis specialist",
            tools=list(set(web_tools + code_tools)),
            model=model,
            system_prompt=base_prompt + "\n" + DEEP_RESEARCH_AGENT_ADDON,
            keywords=["research", "deep", "analysis", "hypothesis"],
            agent_id=2,
            agent_type="code",  # Use CodeAgent since it has coding tools
        ),
        AgentConfig(
            name="DocumentReaderAgent",
            role="Document analysis specialist",
            tools=reader_tools,
            model=model,
            system_prompt=base_prompt + "\n" + DOC_AGENT_ADDON,
            keywords=["document", "pdf", "extract", "page"],
            agent_id=3,
            agent_type="tool_calling",
        ),
    ]

    return [
        DecentralizedAgent(
            config=config,
            message_store=message_store,
            model_type=model_type,
            model_id=model_id,
            provider=provider,
            shared_message_queues=shared_message_queues,
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
