# Agentic RAG

[[open-in-colab]]

## Introduction to Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) combines the power of large language models with external knowledge retrieval to produce more accurate, factual, and contextually relevant responses. At its core, RAG is about "using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base."

### Why Use RAG?

RAG offers several significant advantages over using vanilla or fine-tuned LLMs:

1. **Factual Grounding**: Reduces hallucinations by anchoring responses in retrieved facts
2. **Domain Specialization**: Provides domain-specific knowledge without model retraining
3. **Knowledge Recency**: Allows access to information beyond the model's training cutoff
4. **Transparency**: Enables citation of sources for generated content
5. **Control**: Offers fine-grained control over what information the model can access

### Limitations of Traditional RAG

Despite its benefits, traditional RAG approaches face several challenges:

- **Single Retrieval Step**: If the initial retrieval results are poor, the final generation will suffer
- **Query-Document Mismatch**: User queries (often questions) may not match well with documents containing answers (often statements)
- **Limited Reasoning**: Simple RAG pipelines don't allow for multi-step reasoning or query refinement
- **Context Window Constraints**: Retrieved documents must fit within the model's context window

## Agentic RAG: A More Powerful Approach

We can overcome these limitations by implementing an **Agentic RAG** system - essentially an agent equipped with retrieval capabilities. This approach transforms RAG from a rigid pipeline into an interactive, reasoning-driven process.

### Key Benefits of Agentic RAG

An agent with retrieval tools can:

1. ✅ **Formulate optimized queries**: The agent can transform user questions into retrieval-friendly queries
2. ✅ **Perform multiple retrievals**: The agent can retrieve information iteratively as needed
3. ✅ **Reason over retrieved content**: The agent can analyze, synthesize, and draw conclusions from multiple sources
4. ✅ **Self-critique and refine**: The agent can evaluate retrieval results and adjust its approach

This approach naturally implements advanced RAG techniques:
- **Hypothetical Document Embedding (HyDE)**: Instead of using the user query directly, the agent formulates retrieval-optimized queries ([paper reference](https://huggingface.co/papers/2212.10496))
- **Self-Query Refinement**: The agent can analyze initial results and perform follow-up retrievals with refined queries ([technique reference](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/))

## Building an Agentic RAG System

Let's build a complete Agentic RAG system step by step. We'll create an agent that can answer questions about the Hugging Face Transformers library by retrieving information from its documentation.

You can follow along with the code snippets below, or check out the full example in the smolagents GitHub repository: [examples/rag.py](https://github.com/huggingface/smolagents/blob/main/examples/rag.py).

### Step 1: Install Required Dependencies

First, we need to install the necessary packages:

```bash
pip install smolagents pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25 --upgrade
```

If you plan to use Hugging Face's Inference API, you'll need to set up your API token:

```python
# Load environment variables (including HF_TOKEN)
from dotenv import load_dotenv
load_dotenv()
```

### Step 2: Prepare the Knowledge Base

We'll use a dataset containing Hugging Face documentation and prepare it for retrieval:

```python
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

# Load the Hugging Face documentation dataset
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# Filter to include only Transformers documentation
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Convert dataset entries to Document objects with metadata
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Split documents into smaller chunks for better retrieval
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Characters per chunk
    chunk_overlap=50,  # Overlap between chunks to maintain context
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],  # Priority order for splitting
)
docs_processed = text_splitter.split_documents(source_docs)

print(f"Knowledge base prepared with {len(docs_processed)} document chunks")
```

### Step 3: Create a Retriever Tool

Now we'll create a custom tool that our agent can use to retrieve information from the knowledge base:

```python
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        # Initialize the retriever with our processed documents
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # Return top 10 most relevant documents
        )

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        assert isinstance(query, str), "Your search query must be a string"

        # Retrieve relevant documents
        docs = self.retriever.invoke(query)

        # Format the retrieved documents for readability
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Initialize our retriever tool with the processed documents
retriever_tool = RetrieverTool(docs_processed)
```

> [!TIP]
> We're using BM25, a lexical retrieval method, for simplicity and speed. For production systems, you might want to use semantic search with embeddings for better retrieval quality. Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for high-quality embedding models.

### Step 4: Create an Advanced Retrieval Agent

Now we'll create an agent that can use our retriever tool to answer questions:

```python
from smolagents import InferenceClientModel, CodeAgent

# Initialize the agent with our retriever tool
agent = CodeAgent(
    tools=[retriever_tool],  # List of tools available to the agent
    model=InferenceClientModel(),  # Default model "Qwen/Qwen2.5-Coder-32B-Instruct"
    max_steps=4,  # Limit the number of reasoning steps
    verbosity_level=2,  # Show detailed agent reasoning
)

# To use a specific model, you can specify it like this:
# model=InferenceClientModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
```

> [!TIP]
> Inference Providers give access to hundreds of models, powered by serverless inference partners. A list of supported providers can be found [here](https://huggingface.co/docs/inference-providers/index).

### Step 5: Run the Agent to Answer Questions

Let's use our agent to answer a question about Transformers:

```python
# Ask a question that requires retrieving information
question = "For a transformers model training, which is slower, the forward or the backward pass?"

# Run the agent to get an answer
agent_output = agent.run(question)

# Display the final answer
print("\nFinal answer:")
print(agent_output)
```

## Practical Applications of Agentic RAG

Agentic RAG systems can be applied to various use cases:

1. **Technical Documentation Assistance**: Help users navigate complex technical documentation
2. **Research Paper Analysis**: Extract and synthesize information from scientific papers
3. **Legal Document Review**: Find relevant precedents and clauses in legal documents
4. **Customer Support**: Answer questions based on product documentation and knowledge bases
5. **Educational Tutoring**: Provide explanations based on textbooks and learning materials

## Conclusion

Agentic RAG represents a significant advancement over traditional RAG pipelines. By combining the reasoning capabilities of LLM agents with the factual grounding of retrieval systems, we can build more powerful, flexible, and accurate information systems.

The approach we've demonstrated:
- Overcomes the limitations of single-step retrieval
- Enables more natural interactions with knowledge bases
- Provides a framework for continuous improvement through self-critique and query refinement

As you build your own Agentic RAG systems, consider experimenting with different retrieval methods, agent architectures, and knowledge sources to find the optimal configuration for your specific use case.
