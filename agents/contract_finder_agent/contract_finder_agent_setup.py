from pydantic import Field
from typing import Optional

from llm_studio_agents.AgentBase import AgentSetupBase
from llm_studio_tools.llm_configuration_tool import LLMConfigurationTool


class ContractFinderAgentSetup(AgentSetupBase):
    """
    Agent designed to retrieve contracts based on filters and a user query.
    """
    fallback_response: Optional[str] = Field(
        "I couldn't find relevant contracts for your query. Please try with different filters or a more specific query.",
        description="Default response when no suitable answer can be generated.",
        UITab="Request and Response Behavior"
    )
    show_headers: Optional[bool] = Field(
        False,
        description="Whether to show thinking steps for this agent.",
        UITab="Request and Response Behavior"
    )
    max_contract_ids: Optional[int] = Field(
        50,
        description="Maximum number of contract IDs to retrieve across all filters.",
        ge=1, le=1000,
        UITab="Request and Response Behavior"
    )
    max_chunks_to_use: Optional[int] = Field(
        10,
        description="Maximum number of top-ranked chunks to use for answer generation",
        ge=1, le=20,
        UITab="Request and Response Behavior"
    )
    generate_response_from_docs: Optional[bool] = Field(True,
                                                        description="This describes whether we need to generate response or just return the documents retrieved.", 
                                                        title="Generate Response From Retrieved Documents",
                                                        toolToggle=LLMConfigurationTool.__name__,
                                                        UITab="Request and Response Behavior")
    filter_citations: Optional[bool] = Field(False,
                                            description="Whether to filter the citations based on the generated response. (Works only when response is generated from the documents)",
                                            UITab="Request and Response Behavior")
    filter_citation_threshold: Optional[float] = Field(90, 
                                                     description="The threshold for fuzzy match for the citations", 
                                                     le=100, ge=0,
                                                     UITab="Request and Response Behavior")
    similarity_threshold: Optional[float] = Field(
        0.3,
        description="Threshold of similarity to be allowed for all filters.",
        UITab="Request and Response Behavior"
    )
    max_workers_for_citation_generation: Optional[int] = Field(
        10,
        description="MAX number of parallel workers to be used for citation generation.",
        title="Max Workers for Citation Generation",
        UITab="Request and Response Behavior"
    )
    context_threshold: Optional[float] = Field(
        0.85,
        description="Threshold for context score to be considered relevant for the response generation.",
        le=1.0, ge=0.0,
        UITab="Request and Response Behavior"
    )
    contracts_per_result: Optional[int] = Field(
        20,
        description="Number of contracts to be returned per result.",
        UITab="Request and Response Behavior"
    )
    embedding_dimensions: int = Field(
        1536,
        description="Dimension of the embeddings used in the semantic similarity tool.",
        UITab="Request and Response Behavior"
    )
    max_show_more_contracts: int = Field(
        100,
        description="Maximum number of contracts to be shown in the response in case of show more.",
        UITab="Request and Response Behavior"
    )
    similarity_weight: float = Field(
        0.45,
        description="Weight for similarity score in final document scoring.",
        UITab="Request and Response Behavior"
    )
    document_weight: float = Field(
        0.25,
        description="Weight for document score in final document scoring.",
        UITab="Request and Response Behavior"
    )
    keyword_weight: float = Field(
        0.3,
        description="Weight for keyword score in final document scoring.",
        UITab="Request and Response Behavior"
    )
    default_show_more_length: int = Field(
        10,
        description="Default number of contracts to fetch when show_more_details length is not specified.",
        UITab="Request and Response Behavior"
    )

    context_table: str = Field(
        "contract_retrieved_document_details",
        description="Name of the table which contains the context information for the contracts",
        title="Context Table for Retrieval",
        UITab="Miscellaneous"
    )
    llm_invocation_timeout: int = Field(
        60,
        description="Timeout in seconds for LLM invocation. If exceeded, error sentinel is returned.",
        ge=5, le=600,
        UITab="LLM Configuration"
    )
    llm_error_sentinel: str = Field(
        "[ERROR: LLM_INVOCATION_TIMEOUT]",
        description="Sentinel value returned by LLM tool when timeout occurs. Used for post-invocation detection.",
        UITab="LLM Configuration"
    )
    llm_timeout_fallback_message: str = Field(
        "I was unable to generate a response due to a timeout. Please try again later.",
        description="Hardcoded fallback message substituted when error sentinel is detected in LLM response.",
        UITab="LLM Configuration"
    )
    no_context_fallback_template: str = Field(
        "I understand you're looking for contracts matching your query. Unfortunately, no relevant contracts were found in the current database. Please try with different filters or search terms to broaden your search.",
        description="Prompt template used when context_string is empty (no relevant chunks found). Provides conversational fallback instead of blank context.",
        UITab="LLM Configuration",
        title="No-Context Fallback Template"
    )
