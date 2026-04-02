# Getting Started with smolagents — TL;DR

> Full details: [guided_tour.md](./guided_tour.md)

## CodeAgent vs ToolCallingAgent

**CodeAgent** writes its actions as Python code snippets. The LLM generates code that calls your tools as Python functions, executes loops, chains results, and performs complex logic dynamically. Use CodeAgent when you need multi-step reasoning, composability, or dynamic tool chaining. Requires a secure execution environment (sandbox recommended for production).

**ToolCallingAgent** writes actions as structured JSON. Tools are called by name with typed arguments — no code execution happens. Use ToolCallingAgent when you have simple, atomic tools (call an API, fetch a document), need strict validation, or want maximum predictability. Less expressive but safer and more interoperable.

**Rule of thumb:** CodeAgent for problem-solvers and programmers; ToolCallingAgent for dispatchers and controllers.

## Minimal Working Example

```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize model (HF Inference API, free tier available)
model = InferenceClientModel()

# Create agent — tools=[] uses base Python interpreter only
agent = CodeAgent(tools=[], model=model)

# Run a task
result = agent.run("Calculate the sum of numbers from 1 to 100")
print(result)
```

With web search:

```python
from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool

model = InferenceClientModel()
agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model)
result = agent.run("What are the latest smolagents features?")
```

## LLM Integrations

smolagents supports many model backends:

- **InferenceClientModel** — HuggingFace Inference API (Cerebras, Cohere, Together, Nebius, SambaNova, etc.). Needs `HF_TOKEN`.
- **LiteLLMModel** — 100+ providers via LiteLLM, including OpenAI (`gpt-4o`), Anthropic (`claude-3-5-sonnet-latest`), Ollama (local), and more.
- **TransformersModel** — Run any HuggingFace model locally via `transformers` pipeline.
- **AzureOpenAIModel** — OpenAI models deployed on Azure.
- **AmazonBedrockModel** — Models on AWS Bedrock.
- **MLXModel** — Local inference via `mlx-lm` (Apple Silicon).

```python
# Anthropic via LiteLLM
from smolagents import CodeAgent, LiteLLMModel
model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key="YOUR_KEY")
agent = CodeAgent(tools=[], model=model)

# Local Ollama
model = LiteLLMModel(model_id="ollama_chat/llama3.2", api_base="http://localhost:11434", num_ctx=8192)
agent = CodeAgent(tools=[], model=model)
```

All model classes accept extra kwargs (`temperature`, `max_tokens`, `top_p`) at instantiation.

## CLI Quick Start

```bash
pip install 'smolagents[toolkit]'
smolagent "Plan a trip to Tokyo." --model-type InferenceClientModel --tools web_search
```

> For full model setup, tools, multi-agents, and Hub sharing: see [guided_tour.md](./guided_tour.md)
