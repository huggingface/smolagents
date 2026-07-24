from typing import Optional, List, Dict, Any
from llm_studio_agents.AgentBase import AgentsTraceBase


class ContractFinderAgentTrace(AgentsTraceBase):
    """
    Trace schema for ContractFinderAgent execution.
    Captures execution metadata, intermediate artifacts, and observability metrics.
    """
    retrieved_contract_count: Optional[int] = None
    retrieved_chunks_count: Optional[int] = None
    applied_filters: Optional[List[str]] = None
    generated_citations_count: Optional[int] = None
    execution_duration_seconds: Optional[float] = None
    similarity_scores: Optional[Dict[str, float]] = None
    filter_details: Optional[Dict[str, Any]] = None
