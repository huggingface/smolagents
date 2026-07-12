"""
RAG Agent application using ChromaDB retrieval.

This script loads a persisted ChromaDB vector store
and runs a CodeAgent with a semantic search retriever tool.

Run ingest.py first to create the ChromaDB vector store.
"""

import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from smolagents import LiteLLMModel, Tool
from smolagents.agents import CodeAgent


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

    def __init__(self, vector_store, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        docs = self.vector_store.similarity_search(query, k=3)
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

    retriever_tool = RetrieverTool(vector_store)

    # Choose which LLM engine to use!

    # from smolagents import InferenceClientModel
    # model = InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking")

    # from smolagents import TransformersModel
    # model = TransformersModel(model_id="Qwen/Qwen3-4B-Instruct-2507")

    # For anthropic: change model_id below to 'anthropic/claude-4-sonnet-latest'
    # and also change 'os.environ.get("ANTHROPIC_API_KEY")'
    model = LiteLLMModel(
        model_id="groq/openai/gpt-oss-120b",
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    # # You can also use the ToolCallingAgent class
    # from smolagents.agents import ToolCallingAgent
    # agent = ToolCallingAgent(
    #     tools=[retriever_tool],
    #     model=model,
    #     verbose=True,
    # )

    agent = CodeAgent(
        tools=[retriever_tool],
        model=model,
        max_steps=4,
        verbosity_level=2,
        stream_outputs=True,
    )

    agent_output = agent.run("How can I push a model to the Hub?")

    print("Final output:")
    print(agent_output)
