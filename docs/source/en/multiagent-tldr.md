# Multi-Agent Systems — TL;DR

> Full details: [examples/multiagents.md](./examples/multiagents.md) | [examples/rag.md](./examples/rag.md)

## Manager + Subagent Pattern

In smolagents, a **manager agent** orchestrates one or more **subagents**. The manager receives the top-level task, delegates subtasks to specialists, and synthesizes the final answer.

```
Manager Agent (CodeAgent — reasoning + orchestration)
    |
    +-- Web Search Agent (ToolCallingAgent — web search + page visits)
    |       +-- WebSearchTool
    |       +-- VisitWebpageTool
    |
    +-- Code Interpreter (built-in to manager)
```

## Setting Up Multi-Agent Orchestration

```python
from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel, WebSearchTool

model = InferenceClientModel(model_id="Qwen/Qwen2.5-72B-Instruct")

# 1. Create a subagent with name + description (required for manager to call it)
web_agent = ToolCallingAgent(
    tools=[WebSearchTool()],
    model=model,
    max_steps=10,
    name="web_search_agent",
    description="Runs web searches and visits webpages for you.",
)

# 2. Create manager agent, passing subagents in managed_agents
manager_agent = CodeAgent(
    tools=[],
    model=model,
    managed_agents=[web_agent],
    additional_authorized_imports=["time", "numpy", "pandas"],
)

# 3. Run the top-level task — manager delegates automatically
result = manager_agent.run("What is the GDP of France divided by its population?")
```

Key points:
- Subagents must have `name` and `description` attributes set — this is how the manager knows to call them
- The manager is typically a `CodeAgent` (better at reasoning and orchestration)
- Subagents can be `ToolCallingAgent` (simpler, more reliable for atomic tasks like web fetch)
- Nest as deeply as needed; subagents can themselves be managers

## RAG Integration Pattern

Build an agentic RAG system by wrapping a retriever as a tool:

```python
from smolagents import Tool, CodeAgent

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves documents from the knowledge base relevant to the query."
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def __init__(self, vector_store):
        super().__init__()
        self.vector_store = vector_store

    def forward(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n---\n".join(doc.page_content for doc in docs)

agent = CodeAgent(tools=[RetrieverTool(my_vector_store)], model=model)
agent.run("What does the documentation say about authentication?")
```

The agent automatically performs multi-step retrieval: it reformulates queries, retrieves multiple times, and synthesizes answers — going far beyond single-shot RAG.

## Web Browser Agent

For web browsing tasks, a single `ToolCallingAgent` with `WebSearchTool` + `VisitWebpageTool` is sufficient:

```python
from smolagents import ToolCallingAgent, WebSearchTool, VisitWebpageTool
agent = ToolCallingAgent(tools=[WebSearchTool(), VisitWebpageTool()], model=model)
agent.run("Find the latest smolagents release notes.")
```

> For full multi-agent walkthrough: [examples/multiagents.md](./examples/multiagents.md)
> For complete RAG guide: [examples/rag.md](./examples/rag.md)
