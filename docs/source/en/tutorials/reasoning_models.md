# 🧠 Using Reasoning Models

Reasoning models (also called thinking models) are LLMs that expose their chain-of-thought in a
separate field alongside their final answer. Examples include DeepSeek-R1, Kimi-K2-Thinking, and
Minimax-M2. smolagents has first-class support for these models: it extracts the reasoning
automatically and can optionally include it in the conversation history passed back to the model.

## How it works

When smolagents calls a reasoning model, the raw API response contains two things:

1. **`content`** — the final answer or action the model decided on.
2. **`reasoning_content`** — the chain-of-thought the model used to reach that answer.

smolagents captures both in the [`ChatMessage`] dataclass. The reasoning content is then stored on
[`ActionStep`] in memory, and — depending on the `preserve_reasoning` flag — may be re-sent to
the model in subsequent turns.

```
LLM response
    │
    ▼
_extract_reasoning_content()   ← probes provider-specific fields
    │
    ▼
ChatMessage(content=..., reasoning_content=...)
    │
    ▼
ActionStep(reasoning_content=...)  ← stored in agent memory
    │
    ▼
get_clean_message_list()  ← reasoning_content included in history when preserve_reasoning=True
    │
    ▼
Next LLM call
```

## The `preserve_reasoning` flag

Different providers have **opposite** requirements for whether reasoning should be re-sent:

| Provider | Field name | Re-send reasoning in history? |
|----------|-----------|-------------------------------|
| DeepSeek | `reasoning_content` | **No** — returns HTTP 400 if included |
| Kimi / Moonshot | `reasoning_content` | **Yes** — returns HTTP 400 if missing |
| Minimax | `reasoning_details` | Yes (recommended for quality) |
| Ollama | `reasoning` | Model-dependent |
| Anthropic | content blocks | N/A — handled separately, out of scope |

This is controlled by the `preserve_reasoning` parameter on any agent:

```python
from smolagents import CodeAgent, LiteLLMModel

# Kimi requires reasoning in history → preserve_reasoning=True
agent = CodeAgent(
    model=LiteLLMModel("moonshot/kimi-k2"),
    tools=[],
    preserve_reasoning=True,
)

# DeepSeek crashes if you send reasoning back → preserve_reasoning=False (default)
agent = CodeAgent(
    model=LiteLLMModel("deepseek/deepseek-reasoner"),
    tools=[],
    preserve_reasoning=False,
)
```

`preserve_reasoning` defaults to `False`, which is safe for all providers that don't require
reasoning in history. Only set it to `True` when the provider explicitly requires it (like Kimi).

> [!TIP]
> Even when `preserve_reasoning=False`, the raw reasoning is still accessible on
> `chat_message.raw` for debugging. It is simply not forwarded to the next model call.

## Supported providers

### DeepSeek (via LiteLLM)

DeepSeek-R1 returns reasoning in `reasoning_content`. **Do not** set `preserve_reasoning=True`
with DeepSeek — it will return a 400 error.

```python
import os
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="deepseek/deepseek-reasoner",
    api_key=os.environ["DEEPSEEK_API_KEY"],
)

agent = CodeAgent(
    model=model,
    tools=[],
    preserve_reasoning=False,  # DeepSeek rejects history with reasoning
)

result = agent.run("What is the 10th Fibonacci number?")
print(result)
```

### Kimi / Moonshot (via LiteLLM)

Kimi K2 Thinking returns reasoning in `reasoning_content` and **requires** it to be re-sent in
conversation history.

```python
import os
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="moonshot/kimi-k2",
    api_key=os.environ["MOONSHOT_API_KEY"],
)

agent = CodeAgent(
    model=model,
    tools=[],
    preserve_reasoning=True,  # Kimi requires reasoning to be in history
)

result = agent.run("Solve: if 3x + 7 = 22, what is x?")
print(result)
```

### Ollama (local reasoning models)

Ollama exposes reasoning in a `reasoning` field for models that support it (e.g. `qwq`,
`deepseek-r1`). Use `preserve_reasoning=True` for models that benefit from it, or `False` if
unsure.

```python
from smolagents import CodeAgent, OpenAIModel

model = OpenAIModel(
    model_id="qwq:32b",          # or "deepseek-r1:8b", etc.
    api_base="http://localhost:11434/v1",
    api_key="ollama",            # Ollama ignores the key but requires a value
)

agent = CodeAgent(
    model=model,
    tools=[],
    preserve_reasoning=True,
)

result = agent.run("What are the prime factors of 360?")
print(result)
```

> [!TIP]
> To check whether an Ollama model supports reasoning, run
> `ollama show <model_name>` and look for `thinking` in the capabilities list.

### Minimax (via LiteLLM)

Minimax M2 returns reasoning in `reasoning_details`. Re-sending it is recommended for quality.

```python
import os
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(
    model_id="minimax/MiniMax-M2",
    api_key=os.environ["MINIMAX_API_KEY"],
)

agent = CodeAgent(
    model=model,
    tools=[],
    preserve_reasoning=True,
)

result = agent.run("Explain the difference between TCP and UDP in one paragraph.")
print(result)
```

## Accessing reasoning content in results

The reasoning content is stored on each [`ActionStep`] in the agent's memory. You can inspect it
after a run:

```python
agent.run("What is the square root of 144?")

for step in agent.memory.steps:
    if hasattr(step, "reasoning_content") and step.reasoning_content:
        print("=== Reasoning ===")
        print(step.reasoning_content)
        print("=== Answer ===")
        print(step.model_output)
```

## Streaming

> [!WARNING]
> Streaming mode (`generate_stream`) does not yet accumulate `reasoning_content` deltas.
> Reasoning is only captured in non-streaming (batch) calls. Streaming support is planned.

## Non-reasoning models are unaffected

If the model you use does not return reasoning content, `reasoning_content` will simply be `None`
everywhere. No configuration is needed — the feature is completely invisible for standard models.
