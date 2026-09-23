# Memanto Long-Term Memory Example

This example shows how to give a smolagents `CodeAgent` **persistent, cross-session memory** using [Memanto](https://memanto.ai) custom `Tool` classes — the same pattern as `examples/rag_using_chromadb.py`, but for user facts and preferences instead of document chunks.

## Architecture

```
CodeAgent
   ├── recall_memory   (MemantoRecallTool)
   ├── remember        (MemantoRememberTool)
   └── answer_from_memory (optional MemantoAnswerTool)
              │
              ▼
        MemantoClient (httpx)
              │
              ▼
        memanto serve  →  Moorcheh backend
```

**Two memory layers:**

| Layer | Scope | Cleared when |
|-------|-------|--------------|
| `AgentMemory` (built-in) | Current run steps | `agent.run(..., reset=True)` |
| Memanto | Cross-session facts | Never (unless you delete memories) |

## Prerequisites

1. Python 3.10+
2. [Memanto](https://docs.memanto.ai/getting-started/installation) installed and configured (cloud or on-prem)

```bash
pip install memanto
memanto                    # follow setup wizard (cloud or on-prem)
memanto agent create smolagents-demo
memanto serve              # starts REST API on http://localhost:8000
```

## Install example dependencies

From the repo root:

```bash
pip install -e .
pip install -r examples/memanto/requirements.txt
```

## Run the demo

```bash
python examples/memanto/memanto_memory_agent.py
```

On the **first run** against an empty agent namespace, the script seeds one demo preference memory automatically. Subsequent runs skip seeding to avoid duplicates. To force re-seed:

```bash
python examples/memanto/memanto_memory_agent.py --seed
```

Custom task:

```bash
python examples/memanto/memanto_memory_agent.py --task "What do I prefer for response format?"
```

Optional environment variables (or `.env` in repo root):

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMANTO_URL` | `http://localhost:8000` | Memanto REST server URL |
| `MEMANTO_AGENT_ID` | `smolagents-demo` | Memory namespace (shared across clients if same id) |
| `OPENAI_API_KEY` | — | **Recommended.** Uses OpenAI directly (`gpt-4o-mini` by default) |
| `OPENAI_MODEL_ID` | `gpt-4o-mini` | OpenAI model when `OPENAI_API_KEY` is set |
| `SMOLAGENTS_MODEL_ID` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | Hugging Face model id (fallback) |
| `SMOLAGENTS_MODEL_PROVIDER` | _(auto)_ | HF inference provider, e.g. `fireworks-ai`, `nebius` |
| `LITELLM_MODEL_ID` | — | If set, uses LiteLLM instead of HF Inference |
| `GROQ_API_KEY` | — | If set, uses Groq via LiteLLM |

### Model setup

**Memanto is memory only** — it stores and recalls facts. The agent still needs a separate **LLM** to think, write code, and call tools. That LLM can be OpenAI, Hugging Face, Groq, Ollama, etc.

**Easiest path — OpenAI** (no Hugging Face inference providers needed):

```bash
pip install "smolagents[openai]"
set OPENAI_API_KEY=sk-...
set OPENAI_MODEL_ID=gpt-4o-mini
python examples/memanto/memanto_memory_agent.py
```

Or in your own code:

```python
from smolagents import CodeAgent, OpenAIModel

agent = CodeAgent(
    tools=memanto_tools,
    model=OpenAIModel(model_id="gpt-4o-mini"),
)
```

**Fallback — Hugging Face Inference** (only if you don't use OpenAI):

Enable a provider at [hf.co/settings/inference-providers](https://hf.co/settings/inference-providers), then:

```bash
set SMOLAGENTS_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
set SMOLAGENTS_MODEL_PROVIDER=fireworks-ai
python examples/memanto/memanto_memory_agent.py
```

## Files

| File | Purpose |
|------|---------|
| `memanto_client.py` | HTTP wrapper: activate session, recall, remember, answer |
| `memanto_tools.py` | smolagents `Tool` classes + `create_memanto_tools()` helper |
| `memanto_memory_agent.py` | Runnable demo script |
| `requirements.txt` | Example-specific Python deps |

## Usage in your own agent

```python
from memanto_client import MemantoClient
from memanto_tools import create_memanto_tools
from smolagents import CodeAgent, InferenceClientModel

client = MemantoClient(agent_id="my-project")
tools = create_memanto_tools(client, include_answer=True)

agent = CodeAgent(
    tools=tools,
    model=InferenceClientModel(model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"),
)
agent.run("What do I prefer for API response format?", reset=True)
```

## Sharing memory with Cursor

If you also use Memanto MCP in Cursor (`.cursor/mcp.json`), set the same agent id:

```json
"MEMANTO_DEFAULT_AGENT_ID": "smolagents-demo"
```

Memories stored by the smolagents example will be recallable in Cursor, and vice versa.

## Related examples

- `examples/rag_using_chromadb.py` — document RAG with a custom retriever tool
- `examples/server/main.py` — web chat server with MCP tools
