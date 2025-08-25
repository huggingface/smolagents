"""Message store implementation for decentralized agents."""

import json
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


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

        # Message deduplication and rate limiting
        self.recent_messages = {}  # Store recent message hashes
        self.message_timestamps = {}  # Store timestamps for rate limiting
        self.rate_limit_window = 2.0  # seconds
        self.max_similar_messages = 2  # Max similar messages in window

        # simple duplicate guard (best‑effort in‑memory)
        self._recent_hashes: Dict[str, float] = {}
        self._dedup_window_sec = 2.0

    def post_message(self, message: Dict) -> str:
        """Post a new message to the store."""
        msg_id = str(uuid.uuid4())
        message.update({"id": msg_id, "timestamp": now_ts()})

        with self.messages_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(message) + "\n")
        return msg_id

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

                # Check message visibility
                recipients = msg.get("recipients", [])
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

    # --- Core append & read -------------------------------------------------

    def _compute_message_hash(self, sender: str, content: Any, msg_type: str) -> str:
        """Compute a hash of the message content for deduplication."""
        content_str = str(content) if isinstance(content, (dict, list)) else content
        hash_content = f"{sender}:{msg_type}:{content_str}"
        return sha256(hash_content.encode()).hexdigest()

    def _is_duplicate_message(self, msg_hash: str, timestamp: float) -> bool:
        """Check if a message is a duplicate within the rate limit window."""
        with self._lock:
            # Clean up old messages
            current_time = time.time()
            cutoff_time = current_time - self.rate_limit_window

            self.message_timestamps = {h: ts for h, ts in self.message_timestamps.items() if ts > cutoff_time}

            # Check if we've seen too many similar messages recently
            similar_messages = sum(
                1 for h, ts in self.message_timestamps.items() if h == msg_hash and ts > cutoff_time
            )

            if similar_messages >= self.max_similar_messages:
                return True

            # Record this message
            self.message_timestamps[msg_hash] = timestamp
            return False

    def append_message_with_dedup(
        self,
        sender: str,
        content: Any,
        recipients: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
        msg_type: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Append a message with deduplication and rate limiting."""
        msg_type = msg_type or "message"
        current_time = time.time()

        # For votes and final answer proposals, skip deduplication
        if msg_type not in ["vote", "final_answer_proposal", "poll"]:
            msg_hash = self._compute_message_hash(sender, content, msg_type)
            if self._is_duplicate_message(msg_hash, current_time):
                return None  # Skip duplicate message

        return self.append_message(
            sender=sender,
            content=content,
            recipients=recipients,
            thread_id=thread_id,
            msg_type=msg_type,
            reply_to=reply_to,
        )

    def tail(self, k: int = 200) -> List[Dict[str, Any]]:
        """Return last k messages by timestamp."""
        msgs = list(self._iter_messages())[-k:]
        msgs.sort(key=lambda m: m.get("timestamp", ""))
        return msgs

    def get_mentions(self, agent_id: str, since_ts: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all messages that mention the specified agent."""
        mentions = []
        mention_pattern = f"@{agent_id}"

        with self._lock:
            for msg in self._iter_messages():
                if since_ts and msg.get("timestamp", "") <= since_ts:
                    continue

                content_str = str(msg.get("content", ""))
                if mention_pattern in content_str:
                    mentions.append(msg)

        return sorted(mentions, key=lambda m: m.get("timestamp", ""))

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
                    sender = msg.get("sender", "")
                    content_str = str(msg.get("content", ""))

                    visible = (
                        not recipients
                        or "@all" in recipients  # Public
                        or agent_id in recipients  # Direct message
                        or f"@{agent_id}" in content_str  # Mentioned
                        or sender == agent_id  # Own message
                    )
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
                content_str = str(msg.get("content", ""))
                thread_id = msg.get("thread_id", "main")

                # Skip own messages
                if sender == agent_id:
                    continue

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

    def mark_as_read(self, agent_id: str, message_ids: List[str]) -> None:
        """Mark messages as read by an agent (for future read/unread functionality)."""
        # This could be extended to track read status in a separate file or database
        # For now, this is a placeholder for future implementation
        pass

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
                    sender = msg.get("sender", "")
                    content_str = str(msg.get("content", ""))

                    visible = (
                        not recipients
                        or "@all" in recipients  # Public
                        or agent_id in recipients  # Direct message
                        or f"@{agent_id}" in content_str  # Mentioned
                        or sender == agent_id  # Own message
                    )
                    if not visible:
                        continue

                thread_id = msg.get("thread_id", "main")
                threads.add(thread_id)

        return sorted(list(threads))

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
            msg_id = str(uuid.uuid4())
            record = {
                "id": msg_id,
                "timestamp": now_ts(),
                "sender": sender,
                "type": msg_type or (content.get("type") if isinstance(content, dict) else None),
                "content": content,
                "recipients": recipients,
                "thread_id": thread_id,
                "reply_to": reply_to,
            }
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
        """Get all currently active (open) polls - excludes closed and deleted polls."""
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
    ) -> Dict[str, Any]:
        # Check if there are any active polls - only allow one poll at a time
        active_polls = self.get_active_polls()
        if active_polls:
            print("🚫 Cannot create poll: There is already an active poll in progress")
            print(f"   Active poll ID: {active_polls[0].get('poll_id')} by {active_polls[0].get('proposer')}")
            # Return the existing active poll instead of creating a new one
            return {"error": "Poll already active", "active_poll": active_polls[0]}

        poll_id = str(uuid.uuid4())
        n_agents = len(self.get_all_agents()) or len(self._configured_agents) or 1
        thr = threshold if threshold is not None else majority_plus_one(n_agents)

        # Log poll creation with explicit threshold
        print(f"🗳️  Creating poll: {thr} out of {n_agents} agents must vote YES for consensus")

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
            return None

        # Ensure poll is a dict before accessing its attributes
        if not isinstance(poll, dict):
            return None

        tally = info["tally"]

        # Check if poll should be deleted due to too many NO votes
        if tally["NO"] >= 2:  # Delete poll if 2+ NO votes
            print(f"🗑️  Deleting poll {poll_id} due to {tally['NO']} NO votes (threshold: 2)")
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
            # mark closed (shadow append)
            self.append_message(
                sender=poll.get("proposer", "system"),
                content={**poll, "status": "closed"},
                msg_type="poll",
                reply_to=poll_id,
            )
            # emit final answer
            return self.append_message(
                sender="Coordinator",
                content={
                    "type": "final_answer",
                    "poll_id": poll_id,
                    "answer": poll.get("proposal", "No proposal found"),
                    "tally": tally,
                    "source_proposer": poll.get("proposer"),
                },
                msg_type="final_answer",
                reply_to=poll_id,
            )
        return None

    # ------------------------------ search ------------------------
    def search(self, query: str, *, thread_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        q = query.lower().strip()
        out: List[Dict[str, Any]] = []
        for msg in self._iter_messages() or []:
            if thread_id and msg.get("thread_id") != thread_id:
                continue
            blob = json.dumps(msg, ensure_ascii=False).lower()
            if q in blob:
                out.append(msg)
                if len(out) >= limit:
                    break
        out.sort(key=lambda m: m.get("timestamp", ""))
        return out
