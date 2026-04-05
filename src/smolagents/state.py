import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# 1. The Atomic Unit of Action
class ToolCall(BaseModel):
    """Represents the Agent's intent to use a tool."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ToolResult(BaseModel):
    """The result returned by the tool."""
    call_id: str  # Links back to ToolCall.id
    output: Any
    error: Optional[str] = None
    duration_ms: float

# 2. The Step (One Cycle of Thought -> Action -> Result)
class AgentStep(BaseModel):
    """A discrete unit of the Agent's timeline."""
    step_number: int
    thought: str              # The LLM's reasoning
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    
    # Metadata for observability
    token_usage: int = 0
    latency_ms: float = 0.0

# 3. The "God Object" (The State)
class AgentState(BaseModel):
    """
    The Single Source of Truth.
    This object alone is sufficient to pause, resume, or replay an agent.
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: Literal["idle", "running", "waiting_user", "done", "failed"] = "idle"
    
    # Goal Alignment
    task: str
    plan: List[str] = []      # High-level plan (optional)
    
    # Memory (The "Flight Recorder")
    history: List[AgentStep] = []
    
    # Scratchpad (Variables the agent wants to keep)
    working_memory: Dict[str, Any] = {}
    
    # Governance
    total_tokens: int = 0
    total_cost: float = 0.0
