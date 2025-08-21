"""Message store implementation for decentralized agents."""

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional


ISO = "%Y-%m-%dT%H:%M:%S.%fZ"


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime(ISO)


class MessageStore:
    def __init__(self, run_id: str):
        self.run_id = run_id
        script_dir = Path(__file__).parent.parent
        self.run_dir = script_dir / "runs" / run_id
        self.messages_file = self.run_dir / "messages.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Message deduplication and rate limiting
        self.recent_messages = {}  # Store recent message hashes
        self.message_timestamps = {}  # Store timestamps for rate limiting
        self.rate_limit_window = 2.0  # seconds
        self.max_similar_messages = 2  # Max similar messages in window
        self._lock = threading.Lock()

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
    ) -> List[Dict]:
        """Get messages visible to the given agent."""
        if not self.messages_file.exists():
            return []

        messages = []
        with self.messages_file.open() as f:
            for line in f:
                msg = json.loads(line)

                if last_seen_ts and msg["timestamp"] <= last_seen_ts:
                    continue

                # Skip if message is after last_seen
                if last_seen and msg["id"] <= last_seen:
                    continue

                # Filter by thread if specified
                if thread_id and msg.get("thread_id") != thread_id:
                    continue

                # Check message visibility
                recipients = msg.get("recipients", [])
                if not recipients or "@all" in recipients or agent_id in recipients:
                    messages.append(msg)

        return sorted(messages, key=lambda m: m["timestamp"])

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

    def append_message(
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
        if msg_type not in ["vote", "final_answer_proposal"]:
            msg_hash = self._compute_message_hash(sender, content, msg_type)
            if self._is_duplicate_message(msg_hash, current_time):
                return None  # Skip duplicate message

        mid = str(uuid.uuid4())
        msg = {
            "id": mid,
            "timestamp": now_ts(),
            "type": msg_type,
            "sender": sender,
            "recipients": recipients or [],  # [] means visible to all
            "thread_id": thread_id or "main",
            "reply_to": reply_to,
            "content": content,
        }

        with self._lock:
            with self.messages_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
                f.flush()

        return msg

    def _iter_messages(self):
        if not self.messages_file.exists():
            return
        with self.messages_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # skip corrupted line
                    continue

    def tail(self, k: int = 200) -> List[Dict[str, Any]]:
        """Return last k messages by timestamp."""
        msgs = list(self._iter_messages())[-k:]
        msgs.sort(key=lambda m: m.get("timestamp", ""))
        return msgs

    # --- Voting ------------------------------------------------------------

    def record_vote(
        self,
        proposal_id: str,
        voter: str,
        vote: str,
        confidence: float,
        rationale: str = "",
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        content = {
            "proposal_id": proposal_id,
            "voter": voter,
            "vote": vote,  # "yes" | "no"
            "confidence": float(confidence),
            "rationale": rationale,
        }
        msg = self.append_message(
            sender=voter,
            content=content,
            recipients=None,
            thread_id=thread_id,
            msg_type="vote",
            reply_to=proposal_id,
        )
        return msg

    def count_votes(self, proposal_id: str) -> Dict[str, Any]:
        """Count the last vote per voter on a given proposal."""
        last_vote: Dict[str, Dict[str, Any]] = {}
        for msg in self._iter_messages():
            if msg.get("type") != "vote":
                continue
            c = msg.get("content") or {}
           # Primary: explicit proposal_id in content
            proposal_ok = (c.get("proposal_id") == proposal_id)
            # Fallback: legacy votes that only used reply_to linkage
            if not proposal_ok and msg.get("reply_to") == proposal_id:
                proposal_ok = True
            if not proposal_ok:
                continue
            voter = c.get("voter")
            if not voter:
                continue
            # last write wins (by timestamp order later)
            last_vote[voter] = {"vote": c.get("vote"), "confidence": c.get("confidence")}
        yes = sum(1 for v in last_vote.values() if v.get("vote") == "yes")
        no = sum(1 for v in last_vote.values() if v.get("vote") == "no")
        return {"yes": yes, "no": no, "voters": list(last_vote.keys())}

    # --- Search ------------------------------------------------------------

    def search_messages(
        self,
        query: str,
        thread_id: Optional[str] = None,
        limit: int = 50,
        after_ts: Optional[str] = None,
        before_ts: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q = query.lower()
        out: List[Dict[str, Any]] = []
        for msg in self._iter_messages():
            ts = msg.get("timestamp", "")
            if after_ts and ts <= after_ts:
                continue
            if before_ts and ts >= before_ts:
                continue
            if thread_id and msg.get("thread_id") != thread_id:
                continue
            blob = json.dumps(msg, ensure_ascii=False).lower()
            if q in blob:
                out.append(msg)
                if len(out) >= limit:
                    break
        out.sort(key=lambda m: m.get("timestamp", ""))
        return out
