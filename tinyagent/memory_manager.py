"""
MemoryManager: tracks full conversation, allows pruning, summarizing, and slicing for context.
"""
from typing import List, Dict, Callable, Optional

class MemoryManager:
    def __init__(self,
                 summary_func: Optional[Callable[[List[Dict]], str]] = None,
                 window_size: int = 10):
        # Full raw history of messages (dicts with role and content)
        self.full_messages: List[Dict] = []
        # Function to summarize a list of messages into a string prompt
        self.summary_func = summary_func
        # Sliding window size for context
        self.window_size = window_size
        # Indices of messages marked as critical
        self.critical_indices: set[int] = set()

    def add_message(self, message: Dict, critical: bool = False):
        """Append a new message; optionally mark as critical."""
        self.full_messages.append(message)
        if critical:
            self.critical_indices.add(len(self.full_messages) - 1)

    def prune_errors(self):
        """Remove tool error messages from history."""
        new_msgs = []
        new_crit = set()
        for idx, msg in enumerate(self.full_messages):
            if msg.get("role") == "tool" and msg.get("content", "").lower().startswith("error"):
                continue
            new_idx = len(new_msgs)
            new_msgs.append(msg)
            if idx in self.critical_indices:
                new_crit.add(new_idx)
        self.full_messages = new_msgs
        self.critical_indices = new_crit

    def get_context(self, mode: str = "sliding") -> List[Dict]:
        """
        Return a context slice based on mode:
         - 'full': full history
         - 'sliding': last window_size messages
         - 'summary': a single system message summarizing full history
         - 'mixed': keep critical then sliding
        """
        if mode == "full":
            return list(self.full_messages)
        if mode == "sliding":
            return self.full_messages[-self.window_size:]
        if mode == "summary" and self.summary_func:
            summary = self.summary_func(self.full_messages)
            return [{"role": "system", "content": summary}]
        if mode == "mixed":
            crit = [self.full_messages[i] for i in sorted(self.critical_indices) if 0 <= i < len(self.full_messages)]
            window = self.full_messages[-self.window_size:]
            seen = set()
            out = []
            for m in crit + window:
                key = (m.get("role"), m.get("content"))
                if key not in seen:
                    out.append(m)
                    seen.add(key)
            return out
        return self.full_messages[-self.window_size:]