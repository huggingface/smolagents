"""
Memory system for TinyAgent: record agent lifecycle events for extensible memory management.
"""
import time
from dataclasses import dataclass, asdict
from typing import Any, List, Dict

@dataclass
class MemoryStep:
    """A generic memory step recording an event."""
    event: str
    timestamp: float
    data: Dict[str, Any]

    def dict(self) -> Dict[str, Any]:
        return asdict(self)

class Memory:
    """
    Memory manager for TinyAgent: records steps of the agent lifecycle into MemoryStep entries.
    """
    def __init__(self, agent: Any):
        self.agent = agent
        self.steps: List[MemoryStep] = []

    def handle_event(self, event_name: str, agent: Any, **kwargs) -> None:
        """
        Callback hook for agent events. Records event name, timestamp, and payload.
        Arguments:
            event_name: name of the event
            agent: the TinyAgent instance
            **kwargs: event-specific data
        """
        ts = time.time()
        step = MemoryStep(event=event_name, timestamp=ts, data=kwargs)
        self.steps.append(step)

    def clear(self) -> None:
        """Clear all recorded memory steps."""
        self.steps.clear()