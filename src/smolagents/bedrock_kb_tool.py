"""Amazon Bedrock Knowledge Base retrieval tool for smolagents."""

import os

from .tools import Tool


def _get_source_uri(result: dict) -> str:
    """Extract source URI from a retrieval result, handling all location types."""
    location = result.get('location', {})
    loc_type = location.get('type', '')
    if loc_type == 'S3' or 's3Location' in location:
        return location.get('s3Location', {}).get('uri', '')
    elif loc_type == 'WEB' or 'webLocation' in location:
        return location.get('webLocation', {}).get('url', '')
    elif 'confluenceLocation' in location:
        return location.get('confluenceLocation', {}).get('url', '')
    elif 'salesforceLocation' in location:
        return location.get('salesforceLocation', {}).get('url', '')
    elif 'sharePointLocation' in location:
        return location.get('sharePointLocation', {}).get('url', '')
    elif 'customDocumentLocation' in location:
        return location.get('customDocumentLocation', {}).get('id', '')
    # Fallback to metadata._source_uri (for agentic results)
    return result.get('metadata', {}).get('_source_uri', '')


class BedrockKnowledgeBaseTool(Tool):
    """Retrieves relevant documents from an Amazon Bedrock Managed Knowledge Base.

    This tool queries a Bedrock Knowledge Base using managed search configuration
    (recommended) or agentic retrieval for complex queries with query decomposition
    and managed reranking.

    Args:
        knowledge_base_id: The ID of the Bedrock Knowledge Base.
        region_name: AWS region. Defaults to AWS_REGION env var or us-east-1.
        number_of_results: Maximum number of results to return. Defaults to 5.
        use_agentic_retrieval: Use AgenticRetrieveStream for complex queries. Defaults to True.
    """

    name = "bedrock_knowledge_base"
    description = (
        "Retrieves relevant documents from an Amazon Bedrock Knowledge Base. "
        "Use this tool to search a knowledge base for information relevant to a query. "
        "Input is the search query text, output is the retrieved document passages."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant documents in the knowledge base.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        knowledge_base_id: str | None = None,
        region_name: str | None = None,
        number_of_results: int = 5,
        use_agentic_retrieval: bool | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.knowledge_base_id = knowledge_base_id or os.environ.get("KNOWLEDGE_BASE_ID")
        self.region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
        self.number_of_results = number_of_results
        self.use_agentic_retrieval = use_agentic_retrieval if use_agentic_retrieval is not None else os.environ.get('USE_AGENTIC_RETRIEVAL', 'true').lower() != 'false'
        self._client = None

    def setup(self):
        """Initialize the boto3 client (called on first use)."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError:
            raise ImportError(
                "boto3 is required to use BedrockKnowledgeBaseTool. "
                "Install it with `pip install boto3>=1.43.2`"
            )
        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region_name,
            config=Config(user_agent_extra="smolagents/bedrock-kb"),
        )
        self.is_initialized = True

    def forward(self, query: str) -> str:
        """Retrieve documents from the knowledge base."""
        if not self.is_initialized:
            self.setup()

        if not self.knowledge_base_id:
            return "Error: No knowledge_base_id provided. Set it in the constructor or via KNOWLEDGE_BASE_ID env var."

        if self.use_agentic_retrieval:
            return self._agentic_retrieve(query)
        return self._managed_retrieve(query)

    def _managed_retrieve(self, query: str) -> str:
        """Retrieve using managedSearchConfiguration."""
        try:
            response = self._client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "managedSearchConfiguration": {
                        "numberOfResults": self.number_of_results
                    }
                },
            )
            results = response.get("retrievalResults", [])
            if not results:
                return "No relevant documents found."

            passages = []
            for i, result in enumerate(results, 1):
                content = result.get("content", {}).get("text", "")
                source = _get_source_uri(result) or "unknown"
                passages.append(f"[{i}] {content}\nSource: {source}")
            return "\n\n".join(passages)
        except Exception as e:
            return f"Error retrieving from knowledge base: {e}"

    def _agentic_retrieve(self, query: str) -> str:
        """Retrieve using AgenticRetrieveStream for complex queries."""
        try:
            response = self._client.agentic_retrieve_stream(
                knowledgeBaseId=self.knowledge_base_id,
                messages=[{"content": {"text": query}, "role": "user"}],
                retrievers=[{
                    "configuration": {
                        "knowledgeBase": {
                            "knowledgeBaseId": self.knowledge_base_id,
                            "retrievalOverrides": {
                                "maxNumberOfResults": self.number_of_results
                            },
                        }
                    }
                }],
                agenticRetrieveConfiguration={
                    "foundationModelType": "MANAGED",
                    "rerankingModelType": "MANAGED",
                },
            )
            # Process streaming response
            passages = []
            for event in response.get("stream", []):
                if "result" in event and "results" in event["result"]:
                    for result in event["result"]["results"]:
                        content = result.get("content", {}).get("text", "")
                        source = _get_source_uri(result) or "unknown"
                        passages.append(f"{content}\nSource: {source}")
            if not passages:
                return "No relevant documents found."
            return "\n\n".join(passages)
        except Exception:
            # Fall back to managed retrieve
            return self._managed_retrieve(query)
