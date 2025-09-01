"""Message store implementation for decentralized agents."""

import json
import logging
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def now_ts() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def majority_plus_one(n_agents: int) -> int:
    return (n_agents // 2) + 1


class MessageStore:
    def __init__(self, run_id: str, agent_names: Optional[List[str]] = None):
        self.run_id = run_id
        script_dir = Path(__file__).parent.parent
        self.run_dir = script_dir / "runs" / run_id
        self.messages_file = self.run_dir / "messages.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._configured_agents = list(agent_names or [])

        logging.info(
            json.dumps(
                {
                    "event": "message_store_initialized",
                    "run_id": run_id,
                    "agent_count": len(self._configured_agents),
                    "messages_file": str(self.messages_file),
                }
            )
        )

    def get_messages(
        self,
        agent_id: str,
        last_seen: Optional[str] = None,
        last_seen_ts: Optional[str] = None,
        thread_id: Optional[str] = None,
        include_mentions: bool = True,
        include_private: bool = True,
    ) -> List[Dict]:
        """Get messages visible to the given agent with enhanced filtering."""
        if not self.messages_file.exists():
            return []

        messages = []
        with self._lock:
            for msg in self._iter_messages():
                # Additional safety check
                if not isinstance(msg, dict):
                    continue

                # Skip messages before last_seen timestamp
                if last_seen_ts and msg.get("timestamp", "") <= last_seen_ts:
                    continue

                # Skip if message is before last_seen ID (legacy support)
                if last_seen and msg.get("id", "") <= last_seen:
                    continue

                # Filter by thread if specified
                if thread_id and msg.get("thread_id") != thread_id:
                    continue

                # Check message visibility with enhanced error logging
                recipients = msg.get("recipients", [])

                # Enhanced type safety check with detailed logging - BULLETPROOF VERSION
                if not isinstance(recipients, (list, tuple)):
                    logging.warning(
                        json.dumps(
                            {
                                "event": "type_safety_fix_applied",
                                "location": "get_messages",
                                "message_id": msg.get("id", "unknown"),
                                "recipients_type": type(recipients).__name__,
                                "recipients_value": str(recipients),
                                "sender": msg.get("sender", "unknown"),
                                "timestamp": msg.get("timestamp", "unknown"),
                            }
                        )
                    )
                    if recipients is None:
                        recipients = []
                    elif isinstance(recipients, (int, float, bool)):
                        # Convert problematic types to empty list for safety
                        recipients = []
                    else:
                        recipients = [str(recipients)]  # Convert single value to list

                # Ensure recipients is a list of strings
                recipients = [str(r) for r in recipients if r is not None]

                sender = msg.get("sender", "")
                content_str = str(msg.get("content", ""))

                # Message visibility logic:
                # 1. Public messages (empty recipients or @all)
                # 2. Direct messages to this agent
                # 3. Messages mentioning this agent (@agent_id)
                # 4. Messages from this agent (own messages)
                visible = False

                if not recipients or "@all" in recipients:
                    visible = True  # Public message
                elif agent_id in recipients:
                    visible = include_private  # Direct message
                elif include_mentions and f"@{agent_id}" in content_str:
                    visible = True  # Mentioned in message
                elif sender == agent_id:
                    visible = True  # Own message

                if visible:
                    messages.append(msg)

        return sorted(messages, key=lambda m: m.get("timestamp", ""))

    def get_thread_messages(self, thread_id: str, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all messages from a specific thread, filtered by agent visibility if specified."""
        return self.get_messages(
            agent_id=agent_id or "system",  # Default to system if no agent specified
            thread_id=thread_id,
        )

    def search_messages(
        self,
        query: str,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        limit: int = 50,
        after_ts: Optional[str] = None,
        before_ts: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with agent visibility and better filtering."""
        q = query.lower().strip()
        if not q:
            return []

        out: List[Dict[str, Any]] = []
        with self._lock:
            for msg in self._iter_messages():
                ts = msg.get("timestamp", "")

                # Time filtering
                if after_ts and ts <= after_ts:
                    continue
                if before_ts and ts >= before_ts:
                    continue

                # Thread filtering
                if thread_id and msg.get("thread_id") != thread_id:
                    continue

                # Agent visibility check
                if agent_id:
                    recipients = msg.get("recipients", [])

                    # Enhanced type safety check with detailed logging - BULLETPROOF VERSION
                    if not isinstance(recipients, (list, tuple)):
                        logging.warning(
                            json.dumps(
                                {
                                    "event": "type_safety_fix_applied",
                                    "location": "search_messages",
                                    "message_id": msg.get("id", "unknown"),
                                    "recipients_type": type(recipients).__name__,
                                    "recipients_value": str(recipients),
                                    "sender": msg.get("sender", "unknown"),
                                    "query": query,
                                    "timestamp": msg.get("timestamp", "unknown"),
                                }
                            )
                        )
                        if recipients is None:
                            recipients = []
                        elif isinstance(recipients, (int, float, bool)):
                            # Convert problematic types to empty list for safety
                            recipients = []
                        else:
                            recipients = [str(recipients)]  # Convert single value to list

                    # Ensure recipients is a list of strings
                    recipients = [str(r) for r in recipients if r is not None]

                    sender = msg.get("sender", "")
                    content_str = str(msg.get("content", ""))

                    # Additional safety check for the actual operations
                    try:
                        visible = (
                            not recipients
                            or "@all" in recipients  # Public
                            or agent_id in recipients  # Direct message
                            or f"@{agent_id}" in content_str  # Mentioned
                            or sender == agent_id  # Own message
                        )
                    except TypeError as e:
                        # Log detailed error information if the type check above missed something
                        logging.error(
                            json.dumps(
                                {
                                    "event": "type_error_caught",
                                    "location": "search_messages_visibility_check",
                                    "error": str(e),
                                    "message_id": msg.get("id", "unknown"),
                                    "recipients": recipients,
                                    "recipients_type": type(recipients).__name__,
                                    "agent_id": agent_id,
                                    "sender": sender,
                                    "query": query,
                                    "run_id": self.run_id,
                                }
                            )
                        )
                        # Default to not visible if we can't determine visibility
                        continue
                    if not visible:
                        continue

                # Content search
                blob = json.dumps(msg, ensure_ascii=False).lower()
                if q in blob:
                    out.append(msg)
                    if len(out) >= limit:
                        break

        out.sort(key=lambda m: m.get("timestamp", ""))
        return out

    # ------------------------------ notifications & agent tools -----------
    def get_notifications(self, agent_id: str, since_ts: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Get notifications for an agent including mentions, direct messages, and thread activity."""
        notifications = {"mentions": [], "direct_messages": [], "thread_updates": {}, "polls_needing_votes": []}

        with self._lock:
            for msg in self._iter_messages():
                if since_ts and msg.get("timestamp", "") <= since_ts:
                    continue

                msg_type = msg.get("type", "message")
                sender = msg.get("sender", "")
                recipients = msg.get("recipients", [])

                # Enhanced type safety check with detailed logging - BULLETPROOF VERSION
                if not isinstance(recipients, (list, tuple)):
                    logging.warning(
                        json.dumps(
                            {
                                "event": "type_safety_fix_applied",
                                "location": "get_notifications",
                                "message_id": msg.get("id", "unknown"),
                                "recipients_type": type(recipients).__name__,
                                "recipients_value": str(recipients),
                                "sender": sender,
                                "agent_id": agent_id,
                                "timestamp": msg.get("timestamp", "unknown"),
                            }
                        )
                    )
                    if recipients is None:
                        recipients = []
                    elif isinstance(recipients, (int, float, bool)):
                        # Convert problematic types to empty list for safety
                        recipients = []
                    else:
                        recipients = [str(recipients)]  # Convert single value to list

                # Ensure recipients is a list of strings
                recipients = [str(r) for r in recipients if r is not None]

                content_str = str(msg.get("content", ""))
                thread_id = msg.get("thread_id", "main")

                # Skip own messages
                if sender == agent_id:
                    continue

                # Additional safety check for the actual operations
                try:
                    # Direct messages
                    if agent_id in recipients:
                        notifications["direct_messages"].append(msg)

                    # Mentions
                    if f"@{agent_id}" in content_str:
                        notifications["mentions"].append(msg)

                    # Thread updates (public messages in threads agent is following)
                    if not recipients or "@all" in recipients:
                        if thread_id not in notifications["thread_updates"]:
                            notifications["thread_updates"][thread_id] = []
                        notifications["thread_updates"][thread_id].append(msg)

                except TypeError as e:
                    # Log detailed error information if the type check above missed something
                    logging.error(
                        json.dumps(
                            {
                                "event": "type_error_caught",
                                "location": "get_notifications_processing",
                                "error": str(e),
                                "message_id": msg.get("id", "unknown"),
                                "recipients": recipients,
                                "recipients_type": type(recipients).__name__,
                                "agent_id": agent_id,
                                "sender": sender,
                                "run_id": self.run_id,
                            }
                        )
                    )
                    continue  # Skip this message if we can't process it safely

                # Polls needing votes
                if msg_type == "poll":
                    poll_content = msg.get("content", {})
                    if poll_content.get("status") == "open":
                        # Check if agent hasn't voted yet
                        poll_id = poll_content.get("poll_id")
                        if poll_id:
                            vote_info = self.count_votes(poll_id)
                            if agent_id not in vote_info.get("votes_by_voter", {}):
                                notifications["polls_needing_votes"].append(msg)

        # Sort all notification lists by timestamp
        for key, value in notifications.items():
            if isinstance(value, list):
                notifications[key] = sorted(value, key=lambda m: m.get("timestamp", ""))
            elif isinstance(value, dict):
                for thread, msgs in value.items():
                    notifications[key][thread] = sorted(msgs, key=lambda m: m.get("timestamp", ""))

        return notifications

    def get_active_threads(self, agent_id: Optional[str] = None, since_ts: Optional[str] = None) -> List[str]:
        """Get list of active thread IDs, optionally filtered by agent visibility."""
        threads = set()

        with self._lock:
            for msg in self._iter_messages():
                if since_ts and msg.get("timestamp", "") <= since_ts:
                    continue

                if agent_id:
                    # Apply agent visibility filtering
                    recipients = msg.get("recipients", [])

                    # Enhanced type safety check with detailed logging - BULLETPROOF VERSION
                    if not isinstance(recipients, (list, tuple)):
                        logging.warning(
                            json.dumps(
                                {
                                    "event": "type_safety_fix_applied",
                                    "location": "get_active_threads",
                                    "message_id": msg.get("id", "unknown"),
                                    "recipients_type": type(recipients).__name__,
                                    "recipients_value": str(recipients),
                                    "sender": msg.get("sender", "unknown"),
                                    "agent_id": agent_id,
                                    "timestamp": msg.get("timestamp", "unknown"),
                                }
                            )
                        )
                        if recipients is None:
                            recipients = []
                        elif isinstance(recipients, (int, float, bool)):
                            # Convert problematic types to empty list for safety
                            recipients = []
                        else:
                            recipients = [str(recipients)]  # Convert single value to list

                    # Ensure recipients is a list of strings
                    recipients = [str(r) for r in recipients if r is not None]

                    sender = msg.get("sender", "")
                    content_str = str(msg.get("content", ""))

                    # Additional safety check for the actual operations
                    try:
                        visible = (
                            not recipients
                            or "@all" in recipients  # Public
                            or agent_id in recipients  # Direct message
                            or f"@{agent_id}" in content_str  # Mentioned
                            or sender == agent_id  # Own message
                        )
                    except TypeError as e:
                        # Log detailed error information if the type check above missed something
                        logging.error(
                            json.dumps(
                                {
                                    "event": "type_error_caught",
                                    "location": "get_active_threads_visibility_check",
                                    "error": str(e),
                                    "message_id": msg.get("id", "unknown"),
                                    "recipients": recipients,
                                    "recipients_type": type(recipients).__name__,
                                    "agent_id": agent_id,
                                    "sender": sender,
                                    "run_id": self.run_id,
                                }
                            )
                        )
                        # Skip this message if we can't process it safely
                        continue
                    if not visible:
                        continue

                thread_id = msg.get("thread_id", "main")
                threads.add(thread_id)

        return sorted(list(threads))

    def get_channels_info(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about all channels visible to the agent."""
        channels = {}

        with self._lock:
            for msg in self._iter_messages():
                msg_type = msg.get("type", "message")
                thread_id = msg.get("thread_id", "main")

                # Skip if agent filtering is enabled and message not visible to agent
                if agent_id:
                    recipients = msg.get("recipients", [])

                    # Enhanced type safety check - BULLETPROOF VERSION
                    if not isinstance(recipients, (list, tuple)):
                        if recipients is None:
                            recipients = []
                        elif isinstance(recipients, (int, float, bool)):
                            # Convert problematic types to empty list for safety
                            recipients = []
                        else:
                            recipients = [str(recipients)]

                    # Ensure recipients is a list of strings
                    recipients = [str(r) for r in recipients if r is not None]

                    sender = msg.get("sender", "")
                    content_str = str(msg.get("content", ""))

                    try:
                        visible = (
                            not recipients
                            or "@all" in recipients  # Public
                            or agent_id in recipients  # Direct message
                            or f"@{agent_id}" in content_str  # Mentioned
                            or sender == agent_id  # Own message
                        )
                    except TypeError:
                        continue  # Skip problematic messages

                    if not visible:
                        continue

                # Initialize channel info if not exists
                if thread_id not in channels:
                    channels[thread_id] = {
                        "channel_id": thread_id,
                        "subject": None,
                        "description": None,
                        "creator": None,
                        "created_at": None,
                        "members": set(),
                        "last_activity": None,
                        "message_count": 0,
                        "is_created_channel": False,
                    }

                # Update channel info
                channels[thread_id]["message_count"] += 1
                channels[thread_id]["last_activity"] = msg.get("timestamp", "")

                # Extract channel creation info
                if msg_type == "channel_created":
                    content = msg.get("content", {})
                    if isinstance(content, dict):
                        channels[thread_id].update(
                            {
                                "subject": content.get("subject") or content.get("channel_id", thread_id),
                                "description": content.get("description", ""),
                                "creator": content.get("creator") or msg.get("sender", ""),
                                "created_at": msg.get("timestamp", ""),
                                "is_created_channel": True,
                            }
                        )
                        initial_members = content.get("initial_members", [])
                        if initial_members:
                            channels[thread_id]["members"].update(initial_members)

                # Add message sender to members
                sender = msg.get("sender", "")
                if sender and sender not in ["system", "Coordinator"]:
                    channels[thread_id]["members"].add(sender)

        # Convert sets to lists for JSON serialization
        for channel in channels.values():
            channel["members"] = sorted(list(channel["members"]))

        return channels

    def _append_line(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self.messages_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _iter_messages(self) -> Iterable[Dict[str, Any]]:
        if not self.messages_file.exists():
            return
        with self.messages_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    # Ensure we only yield dictionary objects
                    if isinstance(parsed, dict):
                        yield parsed
                except json.JSONDecodeError as e:
                    # Log JSON parsing errors but continue
                    print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    # Handle other parsing errors
                    print(f"Warning: Unexpected error parsing line {line_num}: {e}")
                    continue

    # ------------------------------ core API ---------------------------------
    def append_message(
        self,
        *,
        sender: str,
        content: Any,
        recipients: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        msg_type: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a message to the log and return the full record."""
        with self._lock:
            # Ensure recipients is always properly defined and type-safe
            if recipients is None:
                # Default behavior: most messages should be visible to all agents
                if msg_type in ["task", "poll", "final_answer", "channel_message"]:
                    recipients = ["@all"]
                elif msg_type == "private_message":
                    # This should have been specified, but default to empty for safety
                    recipients = []
                else:
                    # Default to public for unknown message types
                    recipients = ["@all"]
            elif not isinstance(recipients, (list, tuple)):
                # Convert non-list types to proper format
                if isinstance(recipients, (int, float, bool)):
                    # These types are problematic - convert to empty list for safety
                    logging.warning(
                        json.dumps(
                            {
                                "event": "problematic_recipients_type_fixed",
                                "location": "append_message",
                                "recipients_type": type(recipients).__name__,
                                "recipients_value": str(recipients),
                                "sender": sender,
                                "msg_type": msg_type,
                            }
                        )
                    )
                    recipients = []
                else:
                    # Convert to list
                    recipients = [str(recipients)]

            # Ensure all recipients are strings
            recipients = [str(r) for r in recipients if r is not None]

            msg_id = str(uuid.uuid4())
            record = {
                "id": msg_id,
                "timestamp": now_ts(),
                "sender": sender,
                "type": msg_type or (content.get("type") if isinstance(content, dict) else None),
                "content": content,
                "recipients": recipients,  # Now guaranteed to be a list
                "thread_id": thread_id,
                "reply_to": reply_to,
            }

            # Log message creation with proper recipients
            logging.info(
                json.dumps(
                    {
                        "event": "message_posted",
                        "message_id": msg_id,
                        "sender": sender,
                        "type": msg_type,
                        "thread_id": thread_id,
                        "recipients": recipients,
                        "content_preview": str(content)[:100],
                    }
                )
            )

            self._append_line(record)
            return record

    # ------------------------------ agent set --------------------------------
    def get_all_agents(self) -> List[str]:
        if self._configured_agents:
            return list(self._configured_agents)
        seen = set()
        for m in self._iter_messages() or []:
            s = m.get("sender")
            if s:
                seen.add(s)
        return sorted(seen)

    # ------------------------------ POLLS ------------------------------------
    def get_active_polls(self) -> List[Dict[str, Any]]:
        """Get all currently active (open) polls.

        IMPORTANT: Polls are NOT sorted by timestamp. The first poll to achieve
        voting consensus (N//2+1 votes) should provide the final answer, regardless
        of creation order.

        Returns:
            List[Dict]: Active polls in message iteration order
        """
        import json

        active_polls = []
        for msg in self._iter_messages() or []:
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", {})

            if isinstance(content, dict) and content.get("type") == "poll":
                status = content.get("status")
                if status == "open":  # Only open polls are active
                    active_polls.append(content)
            elif isinstance(content, str):
                # Try to parse string content as JSON
                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, dict) and parsed_content.get("type") == "poll":
                        status = parsed_content.get("status")
                        if status == "open":
                            active_polls.append(parsed_content)
                except json.JSONDecodeError:
                    pass  # Skip invalid JSON

        return active_polls

    def create_poll(
        self,
        *,
        question: str,
        proposal: Any,
        proposer: str,
        options: Optional[List[str]] = None,
        threshold: Optional[int] = None,
        thread_id: Optional[str] = None,
        final_answer: Optional[str] = None,  # Store clean final answer separately
    ) -> Dict[str, Any]:
        # Check if there are any active polls - only allow one poll at a time
        # active_polls = self.get_active_polls()
        # if active_polls:
        #    print("🚫 Cannot create poll: There is already an active poll in progress")
        #   print(f"   Active poll ID: {active_polls[0].get('poll_id')} by {active_polls[0].get('proposer')}")
        #  logging.info(json.dumps({
        #     "event": "poll_creation_blocked",
        #    "reason": "active_poll_exists",
        #   "blocked_proposer": proposer,
        #  "active_poll_id": active_polls[0].get('poll_id'),
        # "active_poll_proposer": active_polls[0].get('proposer')
        # }))
        # Return the existing active poll instead of creating a new one
        # return {"error": "Poll already active", "active_poll": active_polls[0]}

        poll_id = str(uuid.uuid4())
        n_agents = len(self.get_all_agents()) or len(self._configured_agents) or 4  # Default to 4 agents
        thr = threshold if threshold is not None else majority_plus_one(n_agents)

        # Log poll creation with explicit threshold
        print(f"🗳️  Creating poll: {thr} out of {n_agents} agents must vote YES for consensus")
        logging.info(
            json.dumps(
                {
                    "event": "poll_created",
                    "poll_id": poll_id,
                    "proposer": proposer,
                    "question": question,
                    "proposal_preview": str(proposal)[:200],
                    "threshold": thr,
                    "total_agents": n_agents,
                    "options": options or ["YES", "NO"],
                    "thread_id": thread_id,
                    "configured_agents": self._configured_agents,  # Debug info
                    "detected_agents": self.get_all_agents(),  # Debug info
                }
            )
        )

        payload = {
            "type": "poll",
            "poll_id": poll_id,
            "question": question,
            "proposal": proposal,
            "options": options or ["YES", "NO"],
            "threshold": thr,
            "status": "open",
            "proposer": proposer,
        }

        # Store clean final answer if provided (for final answer polls)
        if final_answer is not None:
            payload["final_answer"] = final_answer

        return self.append_message(
            sender=proposer,
            content=payload,
            thread_id=thread_id,
            msg_type="poll",
        )

    def record_vote(
        self,
        *,
        poll_id: str,
        voter: str,
        vote: str,
        confidence: float = 0.5,
        rationale: str = "",
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        v = vote.upper()
        if v not in {"YES", "NO"}:
            raise ValueError("vote must be 'YES' or 'NO'")

        logging.info(
            json.dumps(
                {
                    "event": "vote_recorded",
                    "poll_id": poll_id,
                    "voter": voter,
                    "vote": v,
                    "confidence": confidence,
                    "rationale": rationale[:200] if rationale else "",
                }
            )
        )

        return self.append_message(
            sender=voter,
            content={
                "type": "vote",
                "poll_id": poll_id,
                "voter": voter,
                "vote": v,
                "confidence": float(confidence),
                "rationale": rationale,
            },
            thread_id=thread_id,
            msg_type="vote",
            reply_to=poll_id,
        )

    def count_votes(self, poll_id: str) -> Dict[str, Any]:
        """Tally votes for a poll; latest vote per voter wins."""
        import json

        poll = None
        closed = False
        deleted = False
        votes_by_voter: Dict[str, Dict[str, Any]] = {}
        for msg in self._iter_messages() or []:
            if not isinstance(msg, dict):
                continue  # Skip non-dict messages

            c = msg.get("content", {})

            # Handle case where content might be a string (JSON)
            if isinstance(c, str):
                try:
                    c = json.loads(c)
                except json.JSONDecodeError:
                    continue  # Skip messages with unparsable content

            if not isinstance(c, dict):
                continue  # Skip if content is not a dict after parsing

            t = c.get("type")
            if t == "poll" and c.get("poll_id") == poll_id:
                poll = c
                status = c.get("status")
                closed = status == "closed"
                deleted = status == "deleted"
            elif t == "vote" and c.get("poll_id") == poll_id and not deleted:
                # Don't count votes for deleted polls
                voter = c.get("voter")
                if voter:
                    votes_by_voter[voter] = c  # last wins
            elif t == "final_answer" and c.get("poll_id") == poll_id:
                closed = True

        tally = defaultdict(int)
        for v in votes_by_voter.values():
            vv = v.get("vote")
            if vv in ("YES", "NO"):
                tally[vv] += 1

        n_agents = len(self.get_all_agents()) or 1
        # Ensure poll is a dict before accessing its attributes
        if isinstance(poll, dict):
            threshold = poll.get("threshold", majority_plus_one(n_agents))
        else:
            threshold = majority_plus_one(n_agents)

        return {
            "poll": poll,
            "closed": closed,
            "deleted": deleted,
            "tally": {"YES": tally["YES"], "NO": tally["NO"], "eligible": n_agents, "threshold": threshold},
            "votes_by_voter": votes_by_voter,
        }

    def finalize_poll_if_ready(self, poll_id: str) -> Optional[Dict[str, Any]]:
        info = self.count_votes(poll_id)
        poll = info.get("poll")
        if not poll or info.get("closed"):
            logging.info(
                json.dumps(
                    {
                        "event": "poll_finalization_skipped",
                        "poll_id": poll_id,
                        "reason": "poll_not_found_or_closed",
                        "has_poll": bool(poll),
                        "is_closed": info.get("closed", False),
                    }
                )
            )
            return None

        # Ensure poll is a dict before accessing its attributes
        if not isinstance(poll, dict):
            logging.error(
                json.dumps(
                    {
                        "event": "poll_finalization_error",
                        "poll_id": poll_id,
                        "error": "poll_is_not_dict",
                        "poll_type": type(poll).__name__,
                    }
                )
            )
            return None

        tally = info["tally"]
        logging.info(
            json.dumps(
                {
                    "event": "poll_finalization_check",
                    "poll_id": poll_id,
                    "yes_votes": tally["YES"],
                    "no_votes": tally["NO"],
                    "threshold": tally["threshold"],
                    "eligible_voters": tally["eligible"],
                }
            )
        )

        # Check if poll should be deleted due to too many NO votes
        if tally["NO"] >= 2:  # Delete poll if 2+ NO votes
            print(f"🗑️  Deleting poll {poll_id} due to {tally['NO']} NO votes (threshold: 2)")
            logging.info(
                json.dumps(
                    {
                        "event": "poll_deleted",
                        "poll_id": poll_id,
                        "no_votes": tally["NO"],
                        "reason": "too_many_no_votes",
                    }
                )
            )
            # Mark poll as closed/deleted
            self.append_message(
                sender="Coordinator",
                content={**poll, "status": "deleted", "reason": f"Too many NO votes ({tally['NO']})"},
                msg_type="poll",
                reply_to=poll_id,
            )
            return {"deleted": True, "poll_id": poll_id, "reason": f"Too many NO votes ({tally['NO']})"}

        # Check if poll passed with enough YES votes
        if tally["YES"] >= tally["threshold"]:
            print(f"✅ Poll {poll_id} passed with {tally['YES']} YES votes (threshold: {tally['threshold']})")
            logging.info(
                json.dumps(
                    {
                        "event": "poll_passed",
                        "poll_id": poll_id,
                        "yes_votes": tally["YES"],
                        "threshold": tally["threshold"],
                        "proposer": poll.get("proposer"),
                        "answer_preview": str(poll.get("proposal", ""))[:200],
                    }
                )
            )
            # mark closed (shadow append)
            self.append_message(
                sender=poll.get("proposer", "system"),
                content={**poll, "status": "closed"},
                msg_type="poll",
                reply_to=poll_id,
            )
            # emit final answer
            clean_answer = poll.get("final_answer") or poll.get("proposal", "No proposal found")
            return self.append_message(
                sender="Coordinator",
                content={
                    "type": "final_answer",
                    "poll_id": poll_id,
                    "answer": clean_answer,
                    "tally": tally,
                    "source_proposer": poll.get("proposer"),
                },
                msg_type="final_answer",
                reply_to=poll_id,
            )

        logging.info(
            json.dumps(
                {
                    "event": "poll_not_ready",
                    "poll_id": poll_id,
                    "yes_votes": tally["YES"],
                    "no_votes": tally["NO"],
                    "threshold": tally["threshold"],
                    "reason": "insufficient_votes",
                }
            )
        )
        return None
