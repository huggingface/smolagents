import os
from qdrant_client import QdrantClient,models
from smolagents import LiteLLMModel, Tool
from smolagents.agents import ToolCallingAgent

from dotenv import load_dotenv
load_dotenv()

collection_name = "hybrid"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
dense_model_name = "jinaai/jina-embeddings-v2-small-en"
sparse_model_name = "Qdrant/BM25"

client = QdrantClient(
    url = os.getenv("QDRANT_URL"),
    api_key = os.getenv("QDRANT_API_KEY"),
)

class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, client, **kwargs):
        super().__init__(**kwargs)
        self.client = client

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        prefetch = [
                models.Prefetch(
                    query=models.Document(text=query, model=dense_model_name),
                    using=dense_vector_name
                ),
                models.Prefetch(
                    query=models.Document(text=query, model=sparse_model_name),
                    using=sparse_vector_name,
                ),
        ]   
        relevant_docs = client.query_points(
                collection_name = collection_name,
                query= models.FusionQuery(fusion=models.Fusion.RRF), # apply RRF for Hybrid Search results
                prefetch=prefetch,
                with_payload = True,
                limit=3
        ).points

        
        context = [point.payload["content"] for point in relevant_docs]

        return "Retrieved Context: \n\n".join(context)
    
retriever_tool = RetrieverTool(client)
model = LiteLLMModel(
    model_id="huggingface/together/deepseek-ai/DeepSeek-R1",
    api_key=os.environ.get("HF_TOKEN"),
)

agent = ToolCallingAgent(
    tools=[retriever_tool],
    model=model
)

user_query = "warner bros acquired share price"
agent_output = agent.run(user_query)
print(agent_output)