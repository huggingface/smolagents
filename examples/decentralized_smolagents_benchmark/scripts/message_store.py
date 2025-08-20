"""Message store implementation for decentralized agents."""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class MessageStore:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.run_dir = Path("runs") / run_id
        self.messages_file = self.run_dir / "messages.jsonl"
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def post_message(self, message: Dict) -> str:
        """Post a new message to the store."""
        msg_id = str(uuid.uuid4())
        message.update({
            "id": msg_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

        with self.messages_file.open("a") as f:
            f.write(json.dumps(message) + "\n")
        return msg_id

    def get_messages(self,
                    agent_id: str,
                    last_seen: Optional[str] = None,
                    thread_id: Optional[str] = None) -> List[Dict]:
        """Get messages visible to the given agent."""
        if not self.messages_file.exists():
            return []

        messages = []
        with self.messages_file.open() as f:
            for line in f:
                msg = json.loads(line)

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
