"""
Decentralized communication and polling tools for multi-agent collaboration.

This module provides tools for agents to:
1. Send private messages to specific agents
2. Send messages to channels/threads for group discussion
3. Read messages and notifications
4. Create and vote on polls for decision making
5. Propose final answers through consensus mechanism
"""

import re
from typing import Any, Dict, List, Optional

from smolagents.default_tools import FinalAnswerTool
from smolagents.tools import Tool

from .message_store import MessageStore


# =============================================================================
# MESSAGING TOOLS
# =============================================================================


class SendMessageToAgent(Tool):
    """Tool for sending private messages directly to specific agents."""

    name = "send_message_to_agent"
    description = """Send a private message directly to a specific agent for one-on-one communication.
    Use this for sharing sensitive information, detailed technical discussions, or coordination that
    doesn't need to involve the whole team. The target agent will receive a notification."""
    inputs = {
        "target_agent": {
            "type": "string",
            "description": "Name of the recipient agent (CodeAgent, WebSearchAgent, DeepResearchAgent, DocumentReaderAgent)",
        },
        "message": {"type": "string", "description": "The message content to send to the target agent"},
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, target_agent: str, message: str) -> str:
        """Send a private message to the specified agent."""
        try:
            result = self.message_store.append_message(
                sender=self.agent_name,
                content=message,
                recipients=[target_agent],
                thread_id=None,  # Private messages don't use threads
                msg_type="private_message",
            )
            return f"✉️ Private message sent to {target_agent}: {message[:50]}, result: {result}..."
        except Exception as e:
            return f"❌ Failed to send message to {target_agent}: {str(e)}"

class CreateChannel(Tool):
    """Tool for creating a new channel for group discussions."""
    name = "create_channel"
    description = """Create a new channel for team discussions on specific topics.
    Channels help organize conversations by topic or participant group.
    Use this to establish dedicated spaces for focused discussions."""
    inputs = {
        "channel_id": {
            "type": "string",
            "description": "Unique identifier for the new channel (e.g., 'research', 'analysis', 'implementation')",
        },
        "channel_description": {
            "type": "string",
            "description": "Brief description of the channel's purpose and topic",
        },
        "initial_members": {
            "type": "string",
            "description": "Optional comma-separated list of agent names to initially notify about this channel",
            "nullable": True,
        }
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, channel_id: str, channel_description: str, initial_members: Optional[str] = None) -> str:
        """Create a new channel for team discussions."""
        try:
            # Parse initial members
            member_list = None
            if initial_members:
                member_list = [m.strip() for m in initial_members.split(",") if m.strip()]

            # Create channel by posting a channel creation message
            channel_message = {
                "type": "channel_created",
                "channel_id": channel_id,
                "description": channel_description,
                "creator": self.agent_name,
                "initial_members": member_list or []
            }

            self.message_store.append_message(
                sender=self.agent_name,
                content=channel_message,
                recipients=member_list or ["@all"],
                thread_id=channel_id,
                msg_type="channel_created",
            )

            member_info = f" with members: {', '.join(member_list)}" if member_list else ""
            return f"📢 Channel created: '{channel_id}' - {channel_description}{member_info}"

        except Exception as e:
            return f"❌ Failed to create channel {channel_id}: {str(e)}"


class SendMessageToChannel(Tool):
    """Tool for sending messages to channels/threads for group discussion."""

    name = "send_message_to_channel"
    description = """Send a message to a channel/thread where multiple agents can participate.
    Channels are automatically created if they don't exist. You can:

    - Use existing channels: 'main', 'research', 'analysis', 'implementation'
    - Create topic-based channels: 'hypothesis-testing', 'data-analysis', 'code-review'
    - Create agent-group channels by listing agents: 'CodeAgent,WebSearchAgent'

    Use @AgentName to mention specific agents and get their attention."""
    inputs = {
        "thread_id": {
            "type": "string",
            "description": """Channel/thread ID or specification:
            - Topic-based: 'research', 'analysis', 'implementation', 'hypothesis-testing'
            - Agent-based: 'CodeAgent,WebSearchAgent' (comma-separated agent names)
            - General: 'main' for general discussion
            Channels will be auto-created if they don't exist.""",
        },
        "message": {
            "type": "string",
            "description": "The message content to send. Use @AgentName to mention specific agents.",
        },
        "recipients": {
            "type": "string",
            "description": "Optional comma-separated list of specific agent names to notify. If not provided, uses channel members or mentioned agents.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, thread_id: str, message: str, recipients: Optional[str] = None) -> str:
        """Send a message to the specified channel/thread with auto-creation."""
        try:
            # Ensure thread_id is a string to prevent type errors
            if not isinstance(thread_id, str):
                thread_id = str(thread_id)

            # Process the thread_id to determine channel type and auto-create if needed
            processed_channel_id, auto_recipients = self._process_channel_id(thread_id)

            # Process mentions in the message
            mentioned_agents = self._extract_mentions(message)

            # Determine final recipient list
            recipient_list = None
            if recipients:
                recipient_list = [r.strip() for r in recipients.split(",")]
            elif mentioned_agents:
                recipient_list = mentioned_agents
            elif auto_recipients:
                recipient_list = auto_recipients
            # If no specific recipients, use @all for public channels

            self.message_store.append_message(
                sender=self.agent_name,
                content=message,
                recipients=recipient_list,
                thread_id=processed_channel_id,
                msg_type="channel_message",
            )

            # Create info message about recipients/mentions
            recipient_info = ""
            if mentioned_agents:
                recipient_info += f" (mentioning: {', '.join(mentioned_agents)})"
            if recipient_list and recipient_list != mentioned_agents:
                recipient_info += f" (recipients: {', '.join(recipient_list)})"

            return f"📢 Message sent to #{processed_channel_id}{recipient_info}: {message[:50]}..."

        except Exception as e:
            return f"❌ Failed to send message to #{thread_id}: {str(e)}"

    def _process_channel_id(self, thread_id: str) -> tuple[str, Optional[List[str]]]:
        """Process thread_id to determine channel and auto-create if needed."""

        # Ensure thread_id is a string to avoid 'int is not iterable' errors
        if not isinstance(thread_id, str):
            thread_id = str(thread_id)

        # If thread_id contains agent names (has comma or ends with 'Agent'), it's an agent-based channel
        if ',' in thread_id or thread_id.endswith('Agent'):
            # Extract agent names
            if ',' in thread_id:
                agent_names = [name.strip() for name in thread_id.split(',') if name.strip()]
            else:
                agent_names = [thread_id.strip()]

            # Create a channel ID from agent names
            channel_id = f"group-{'-'.join(sorted(agent_names)).lower()}"

            # Check if this agent-based channel exists
            if not self._channel_exists(channel_id):
                self._auto_create_channel(
                    channel_id=channel_id,
                    description=f"Private discussion group for: {', '.join(agent_names)}",
                    members=agent_names
                )

            return channel_id, agent_names

        # Topic-based or standard channels
        else:
            # Check if the channel exists
            if not self._channel_exists(thread_id):
                # Auto-create topic-based channel
                description = self._generate_topic_description(thread_id)
                self._auto_create_channel(
                    channel_id=thread_id,
                    description=description,
                    members=None  # Public channel
                )

            return thread_id, None

    def _channel_exists(self, channel_id: str) -> bool:
        """Check if a channel already exists by looking for channel creation messages."""
        try:
            # Look for existing messages in this thread
            existing_messages = self.message_store.get_thread_messages(channel_id)
            return len(existing_messages) > 0
        except Exception:
            return False

    def _auto_create_channel(self, channel_id: str, description: str, members: Optional[List[str]]) -> None:
        """Auto-create a channel with given parameters."""
        try:
            channel_message = {
                "type": "channel_created",
                "channel_id": channel_id,
                "description": description,
                "creator": self.agent_name,
                "auto_created": True,
                "initial_members": members or []
            }

            self.message_store.append_message(
                sender="system",
                content=channel_message,
                recipients=members or ["@all"],
                thread_id=channel_id,
                msg_type="channel_created",
            )

            print(f"🆕 Auto-created channel: #{channel_id} - {description}")

        except Exception as e:
            print(f"⚠️ Failed to auto-create channel #{channel_id}: {e}")

    def _generate_topic_description(self, topic: str) -> str:
        """Generate a description for topic-based channels."""
        # Ensure topic is a string to prevent type errors
        if not isinstance(topic, str):
            topic = str(topic)

        topic_descriptions = {
            "research": "Web research and information gathering",
            "analysis": "Deep analysis and data examination",
            "implementation": "Code development and implementation",
            "hypothesis": "Hypothesis development and testing",
            "documentation": "Document analysis and review",
            "planning": "Project planning and coordination",
            "testing": "Testing and validation discussions",
            "review": "Code and content review sessions",
            "brainstorm": "Brainstorming and ideation",
            "debug": "Debugging and troubleshooting"
        }

        # Try to match common topics
        for key, desc in topic_descriptions.items():
            if key in topic.lower():
                return desc

        # Default description
        return f"Discussion channel for {topic.replace('-', ' ').replace('_', ' ')}"

    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from message text."""
        return re.findall(r"@([A-Za-z0-9_-]+)", text or "")


# =============================================================================
# MESSAGE READING TOOLS
# =============================================================================


class ReadMessagesTool(Tool):
    """Tool for reading messages addressed to the current agent."""

    name = "read_messages"
    description = """Read all new messages addressed to this agent including:
    - Private messages sent directly to you
    - Channel/thread messages where you were mentioned or that match your interests
    - Poll notifications where your vote is needed
    Returns a list of message objects with sender, content, thread, and type information."""
    inputs = {
        "since_timestamp": {
            "type": "string",
            "description": "Optional timestamp to get only messages after this time (ISO format). If not provided, gets recent unread messages.",
            "nullable": True,
        }
    }
    output_type = "array"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, since_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read all messages for this agent."""
        try:
            messages = self.message_store.get_messages(
                agent_id=self.agent_name, last_seen_ts=since_timestamp, include_mentions=True, include_private=True
            )

            if not messages:
                return []

            # Format messages for display
            formatted_messages = []
            for msg in messages:
                formatted_msg = {
                    "id": msg.get("id"),
                    "sender": msg.get("sender"),
                    "type": msg.get("type", "message"),
                    "content": msg.get("content"),
                    "thread_id": msg.get("thread_id"),
                    "timestamp": msg.get("timestamp"),
                    "recipients": msg.get("recipients"),
                }
                formatted_messages.append(formatted_msg)

            return formatted_messages
        except Exception as e:
            return [{"error": f"Failed to read messages: {str(e)}"}]


class ReadNotificationsTool(Tool):
    """Tool for checking notifications including mentions, direct messages, and polls."""

    name = "read_notifications"
    description = """Check all notifications for this agent. Returns categorized notifications:
    - mentions: Messages where you were mentioned with @YourName
    - direct_messages: Private messages sent directly to you
    - polls_needing_votes: Active polls where you haven't voted yet
    - thread_updates: New activity in threads you're following
    Use this to stay updated on important communications and required actions."""
    inputs = {
        "since_timestamp": {
            "type": "string",
            "description": "Optional timestamp to get notifications since a specific time (ISO format)",
            "nullable": True,
        }
    }
    output_type = "object"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, since_timestamp: Optional[str] = None) -> Dict[str, Any]:
        """Get categorized notifications for this agent."""
        try:
            notifications = self.message_store.get_notifications(agent_id=self.agent_name, since_ts=since_timestamp)

            # Add polls needing votes
            active_polls = self.message_store.get_active_polls()
            polls_needing_votes = []

            for poll in active_polls:
                poll_id = poll.get("poll_id")
                if poll_id:
                    vote_info = self.message_store.count_votes(poll_id)
                    voters = vote_info.get("votes_by_voter", {})
                    if self.agent_name not in voters:
                        polls_needing_votes.append(
                            {
                                "poll_id": poll_id,
                                "question": poll.get("question"),
                                "proposal": poll.get("proposal"),
                                "proposer": poll.get("proposer"),
                                "thread_id": poll.get("thread_id", "main"),
                            }
                        )

            notifications["polls_needing_votes"] = polls_needing_votes
            return notifications

        except Exception as e:
            return {"error": f"Failed to get notifications: {str(e)}"}


class SearchMessagesTool(Tool):
    """Tool for searching through message history."""

    name = "search_messages"
    description = """Search through the message history to find relevant information.
    Use this to find previous discussions, research findings, or decisions made by the team.
    You can search by keywords, filter by thread, or limit results by time period."""
    inputs = {
        "query": {
            "type": "string",
            "description": "Search query - keywords or phrases to search for in message content",
        },
        "thread_id": {
            "type": "string",
            "description": "Optional thread ID to search within (e.g., 'main', 'research', 'implementation')",
            "nullable": True,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 20)",
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, query: str, thread_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Search messages visible to this agent."""
        try:
            messages = self.message_store.search_messages(
                query=query, agent_id=self.agent_name, thread_id=thread_id, limit=limit
            )
            return messages
        except Exception as e:
            return [{"error": f"Failed to search messages: {str(e)}"}]


# =============================================================================
# POLLING TOOLS
# =============================================================================


class CreateGeneralPollTool(Tool):
    """Tool for creating general polls for team decision making."""

    name = "create_general_poll"
    description = """Create a general poll to gather team consensus on intermediate decisions, approaches, or strategies.
    Use this for collaborative decision-making when you need team input on research directions, implementation approaches,
    or other non-final decisions."""
    inputs = {
        "question": {"type": "string", "description": "The question or decision to vote on"},
        "proposal": {"type": "string", "description": "Your proposed answer or approach"},
        "thread_id": {"type": "string", "description": "Thread ID for the poll (default: 'main')", "nullable": True},
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, question: str, proposal: str, thread_id: str = "main") -> str:
        """Create a general poll for team decision-making."""
        try:
            # Check if there are any active polls
            #active_polls = self.message_store.get_active_polls()
            #if active_polls:
            #    return f"🚫 Cannot create poll: There is already an active poll in progress (ID: {active_polls[0].get('poll_id')})"

            result = self.message_store.create_poll(
                question=question, proposal=proposal, proposer=self.agent_name, thread_id=thread_id
            )

            poll_id = result.get("content", {}).get("poll_id", "unknown")
            return f"🗳️ General poll created (ID: {poll_id}): {question[:50]}..."

        except Exception as e:
            return f"❌ Failed to create poll: {str(e)}"


class CreateFinalAnswerPollTool(Tool):
    """Tool for creating polls specifically for proposing final answers."""

    name = "create_final_answer_poll"
    description = """Create a poll to propose a final answer to the user's original question.
    Use this when you have a confident, complete answer that should be presented to the user.
    The proposal will be voted on by all agents, and if it reaches majority consensus (N//2 + 1 votes),
    it will be returned as the final answer to the user.
    
    CRITICAL FORMAT INSTRUCTIONS:
    Always carefully follow the format required by the question. This could be for instance:
    - For math problems: final_answer should be ONLY the number, expression, or mathematical result (e.g., "7", "3.14", "$50")
    - For factual questions: final_answer should be ONLY the specific fact requested (e.g., "1925", "John Smith", "Paris")
    - For yes/no questions: final_answer should be ONLY "Yes" or "No"
    - Do NOT include explanations, reasoning, or phrases like "The answer is..." in final_answer
    - Put all explanations in supporting_evidence, not in final_answer
    """
    inputs = {
        "final_answer": {
            "type": "string",
            "description": "ONLY the core answer - no explanations or reasoning. For math: just the number/result. For facts: just the specific requested information. For yes/no: just 'Yes' or 'No'.",
        },
        "supporting_evidence": {
            "type": "string",
            "description": "Supporting evidence, reasoning, calculations, or sources for your answer. Include all explanations HERE, not in final_answer.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, final_answer: str, supporting_evidence: str = "") -> str:
        """Create a poll for a final answer proposal."""
        try:
            # Check if there are any active polls
            #active_polls = self.message_store.get_active_polls()
            #if active_polls:
            #    return f"🚫 Cannot create final answer poll: There is already an active poll in progress (ID: {active_polls[0].get('poll_id')})"

            # Create question and proposal for final answer
            question = "Should this be our final answer to the user?"
            full_proposal = (
                f"{final_answer}\n\n**Supporting Evidence:**\n{supporting_evidence}"
                if supporting_evidence
                else final_answer
            )

            # Store the clean final answer separately in the poll for proper extraction
            result = self.message_store.create_poll(
                question=question,
                proposal=full_proposal,
                proposer=self.agent_name,
                thread_id="main",
                final_answer=final_answer  # Store clean answer separately
            )

            poll_id = result.get("content", {}).get("poll_id", "unknown")
            return f"🗳️ Final answer poll created (ID: {poll_id}): {final_answer[:50]}..."

        except Exception as e:
            return f"❌ Failed to create final answer poll: {str(e)}"


class VoteOnPollTool(Tool):
    """Tool for voting on active polls with detailed evaluation."""

    name = "vote_on_poll"
    description = """Vote on a specific active poll. Provide the poll ID, your vote (YES/NO), confidence level,
    and detailed rationale based on your expertise. Your vote helps reach team consensus on important decisions."""
    inputs = {
        "poll_id": {"type": "string", "description": "ID of the poll you want to vote on", "nullable": True},
        "vote": {"type": "string", "description": "Your vote: 'YES' to approve or 'NO' to reject the proposal"},
        "confidence": {
            "type": "number",
            "description": "Confidence level in your vote (0.0 to 1.0, where 1.0 = completely confident)",
        },
        "rationale": {
            "type": "string",
            "description": "Detailed explanation for your vote, including your reasoning and any suggestions for improvement",
        },
    }
    output_type = "string"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self, vote: str, confidence: float, rationale: str, poll_id: Optional[str] = None) -> str:
        """Vote on an active poll."""
        try:
            # Validate vote
            vote = vote.upper()
            if vote not in ["YES", "NO"]:
                return "❌ Invalid vote. Must be 'YES' or 'NO'."

            # Get active polls
            active_polls = self.message_store.get_active_polls()
            if not active_polls:
                return "❌ No active polls to vote on."

            # Find the specific poll to vote on
            target_poll = None
            if poll_id:
                # Vote on specific poll ID
                for poll in active_polls:
                    if poll.get("poll_id") == poll_id:
                        target_poll = poll
                        break
                if not target_poll:
                    return f"❌ Poll with ID {poll_id} not found or is not active."
            else:
                # If no poll_id specified and only one active poll, use it
                if len(active_polls) == 1:
                    target_poll = active_polls[0]
                else:
                    poll_list = "\n".join([f"- {p.get('poll_id', 'unknown')}: {p.get('question', 'Unknown question')[:60]}..." for p in active_polls])
                    return f"❌ Multiple active polls found. Please specify poll_id parameter:\n{poll_list}"

            poll_id = target_poll.get("poll_id")
            if not poll_id:
                return "❌ Invalid poll ID."

            # Check if already voted
            vote_info = self.message_store.count_votes(poll_id)
            voters = vote_info.get("votes_by_voter", {})
            if self.agent_name in voters:
                return f"❌ You have already voted on poll {poll_id}. Current vote: {voters[self.agent_name].get('vote')}"

            # Record vote
            result = self.message_store.record_vote(
                poll_id=poll_id,
                voter=self.agent_name,
                vote=vote,
                confidence=confidence,
                rationale=rationale,
                thread_id=target_poll.get("thread_id", "main"),
            )

            return f"✅ Vote recorded on poll {poll_id}: {vote} (confidence: {confidence:.1f}) - {rationale[:50]}..."

        except Exception as e:
            return f"❌ Failed to record vote: {str(e)}"


class ViewActivePollsTool(Tool):
    """Tool for viewing currently active polls and their details."""

    name = "view_active_polls"
    description = """View details of currently active polls including the question, proposal, current vote counts,
    and which agents have voted. Use this to see what decisions are pending and check voting progress."""
    inputs = {}
    output_type = "array"

    def __init__(self, message_store: MessageStore, agent_name: str):
        super().__init__()
        self.message_store = message_store
        self.agent_name = agent_name

    def forward(self) -> List[Dict[str, Any]]:
        """View active polls with voting status."""
        try:
            active_polls = self.message_store.get_active_polls()

            if not active_polls:
                return [{"message": "No active polls"}]

            poll_details = []
            for poll in active_polls:
                poll_id = poll.get("poll_id")
                if poll_id:
                    vote_info = self.message_store.count_votes(poll_id)
                    poll_details.append(
                        {
                            "poll_id": poll_id,
                            "question": poll.get("question"),
                            "proposal": poll.get("proposal"),
                            "proposer": poll.get("proposer"),
                            "thread_id": poll.get("thread_id"),
                            "vote_counts": vote_info.get("tally", {}),
                            "voters": list(vote_info.get("votes_by_voter", {}).keys()),
                            "has_voted": self.agent_name in vote_info.get("votes_by_voter", {}),
                        }
                    )

            return poll_details

        except Exception as e:
            return [{"error": f"Failed to view polls: {str(e)}"}]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_decentralized_tools(message_store: MessageStore, agent_name: str) -> List[Tool]:
    """Create all decentralized communication and polling tools for an agent."""
    return [
        # Communication tools
        SendMessageToAgent(message_store, agent_name),
        SendMessageToChannel(message_store, agent_name),
        # Reading tools
        ReadMessagesTool(message_store, agent_name),
        ReadNotificationsTool(message_store, agent_name),
        SearchMessagesTool(message_store, agent_name),
        # Polling tools
        CreateGeneralPollTool(message_store, agent_name),
        CreateFinalAnswerPollTool(message_store, agent_name),
        VoteOnPollTool(message_store, agent_name),
        ViewActivePollsTool(message_store, agent_name),
        # Final answer tool (decentralized version)
        CreateChannel(message_store, agent_name),
    ]
