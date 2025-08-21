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

    def count_votes(self, proposal_id: str) -> Dict[str, int]:
        """Count yes/no votes for a given proposal."""
        votes = {"yes": 0, "no": 0}

        with self.messages_file.open() as f:
            for line in f:
                msg = json.loads(line)
                if msg.get("reply_to") == proposal_id and msg.get("vote") in ["yes", "no"]:
                    votes[msg["vote"]] += 1

        return votes
