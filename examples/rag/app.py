"""
RAG Agent application using BM25 retrieval.

This script loads pre-processed documents and runs
a CodeAgent with a BM25-based retriever tool.

Run ingest.py first to generate docs_processed.pkl.
"""

import pickle

from langchain_community.retrievers import BM25Retriever

from smolagents import CodeAgent, InferenceClientModel, Tool


class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses lexical search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be lexically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=10)

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [f"\n\n===== Document {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )


if __name__ == "__main__":
    with open("docs_processed.pkl", "rb") as f:
        docs_processed = pickle.load(f)

    retriever_tool = RetrieverTool(docs_processed)
    agent = CodeAgent(
        tools=[retriever_tool],
        model=InferenceClientModel(model_id="Qwen/Qwen3-Next-80B-A3B-Thinking"),
        max_steps=4,
        verbosity_level=2,
        stream_outputs=True,
    )

    agent_output = agent.run("For a transformers model training, which is slower, the forward or the backward pass?")

    print("Final output:")
    print(agent_output)
