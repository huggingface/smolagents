# Async Support in Smolagents

Smolagents now supports asynchronous execution, enabling concurrent agent operations and improved performance for I/O-bound tasks.

## Overview

Async support allows you to:
- **Non-blocking I/O**: While one agent waits for an API response, others can continue working instead of blocking
- **Efficient concurrency**: Handle many concurrent operations with lower resource overhead than threading
- **Better scalability**: Run hundreds or thousands of concurrent agents efficiently
- **Async framework integration**: Native compatibility with FastAPI, aiohttp, and other async frameworks

### ⚠️ Important: Agent Instances and Concurrency

**Each concurrent task must use its own agent instance.** Agent memory is stateful, so sharing an agent between concurrent tasks will cause memory corruption. See the [Concurrent Execution](#concurrent-execution) section for details.

## Installation

To use async features, install smolagents with the async extra:

```bash
pip install smolagents[async]
```

This installs the required dependencies:
- `aiohttp>=3.10.0` - For async HTTP requests
- `aiofiles>=24.1.0` - For async file operations

## Async API

### Models

All API-based model classes now support async methods:

```python
import asyncio
from smolagents import LiteLLMModel
from smolagents.models import ChatMessage, MessageRole

async def example():
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "Hello!"}]
        )
    ]

    # Async generate
    response = await model.agenerate(messages)
    print(response.content)

    # Async streaming
    async for delta in model.agenerate_stream(messages):
        print(delta.content, end="")

asyncio.run(example())
```

#### Supported Models

The following model classes support async:
- `LiteLLMModel` - via `litellm.acompletion()`
- `LiteLLMRouterModel` - inherits from LiteLLMModel
- `InferenceClientModel` - via `AsyncInferenceClient`
- `OpenAIServerModel` - via `openai.AsyncOpenAI`
- `AzureOpenAIServerModel` - inherits from OpenAIServerModel

**Methods:**
- `agenerate(messages, ...)` - Async version of `generate()`
- `agenerate_stream(messages, ...)` - Async generator version of `generate_stream()`

### Agents

The base `Agent` class and subclasses now support async execution:

```python
import asyncio
from smolagents import CodeAgent, LiteLLMModel

async def example():
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")
    agent = CodeAgent(model=model, tools=[])

    # Simple async execution
    result = await agent.arun("What is 2 + 2?")
    print(result)

    # Async streaming
    async for step in agent.arun("Calculate 10 * 5", stream=True):
        print(f"Step: {step}")

asyncio.run(example())
```

**Methods:**
- `arun(task, stream=False, ...)` - Async version of `run()`

## Concurrent Execution

The main benefit of async support is **efficient non-blocking I/O**. When an agent awaits an API response, the event loop can switch to other agents instead of blocking. This provides better performance and scalability compared to thread-based concurrency:

- **Threading approach**: Each blocking API call holds a thread idle (1-8MB memory per thread, context switching overhead)
- **Async approach**: Thousands of agents can share a single thread, switching efficiently during I/O waits

This makes async ideal for running many agents concurrently, especially when each agent makes multiple API calls.

### ⚠️ IMPORTANT: Agent Instance Per Task

**Each concurrent task MUST use its own agent instance!** Agent memory is stateful and stores execution history. Sharing an agent instance between concurrent tasks will cause memory corruption where steps from different tasks get mixed together, leading to incorrect results.

```python
import asyncio
from smolagents import CodeAgent, LiteLLMModel

async def run_multiple_agents():
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    tasks = [
        "What is 2 + 2?",
        "What is 10 * 5?",
        "What is 100 / 4?",
    ]

    # ✅ CORRECT: Create separate agent instances for each task
    agents = [CodeAgent(model=model, tools=[]) for _ in tasks]
    results = await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])

    # ❌ INCORRECT: Reusing same agent (causes memory corruption!)
    # agent = CodeAgent(model=model, tools=[])
    # results = await asyncio.gather(*[agent.arun(task) for task in tasks])  # BAD!

    for task, result in zip(tasks, results):
        print(f"{task} = {result}")

asyncio.run(run_multiple_agents())
```

**Why?** Agent instances maintain state in `self.memory`, `self.state`, and `self.step_number`. When multiple `arun()` calls execute concurrently on the same agent:
- Steps from different tasks get interleaved in memory
- The LLM receives confused context mixing multiple tasks
- Results become incorrect or nonsensical
- Race conditions can cause crashes

**Solution:** Always create a separate agent instance for each concurrent task. Models can be safely shared since they're stateless.

### Performance Benefits

Async provides significant speedup for I/O-bound operations by avoiding blocking. While one agent waits for an API response, others can continue executing:

```python
import asyncio
import time
from smolagents import CodeAgent, LiteLLMModel

async def compare_performance():
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

    tasks = ["Task " + str(i) for i in range(10)]

    # Async (concurrent) - separate agent per task
    start = time.time()
    agents = [CodeAgent(model=model, tools=[]) for _ in tasks]
    await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])
    async_time = time.time() - start

    # Sync (sequential)
    start = time.time()
    for task in tasks:
        agent = CodeAgent(model=model, tools=[])
        agent.run(task)
    sync_time = time.time() - start

    print(f"Async: {async_time:.2f}s")
    print(f"Sync: {sync_time:.2f}s")
    print(f"Speedup: {sync_time / async_time:.2f}x")

asyncio.run(compare_performance())
```

**Expected speedup:** ~10x for 10 concurrent agents.

**Why?** The speedup comes from non-blocking I/O:
- **Sequential (sync)**: Agent 1 → wait for API → Agent 2 → wait for API → ... (total = sum of all wait times)
- **Concurrent (async)**: All agents run concurrently, waiting in parallel (total ≈ longest single wait time)

This is more efficient than threading because async tasks have minimal overhead (~few KB per task vs ~1-8MB per thread).

## Integration with Async Frameworks

Async support allows seamless integration with async web frameworks:

### FastAPI Example

```python
from fastapi import FastAPI
from smolagents import CodeAgent, LiteLLMModel

app = FastAPI()
# Model can be shared (stateless)
model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

@app.post("/run-agent")
async def run_agent(task: str):
    # Create a new agent instance for each request
    # This prevents memory corruption from concurrent requests
    agent = CodeAgent(model=model, tools=[])
    result = await agent.arun(task)
    return {"result": result}
```

**Note:** Creating a new agent per request is necessary to avoid memory corruption when handling concurrent requests.

### aiohttp Example

```python
from aiohttp import web
from smolagents import CodeAgent, LiteLLMModel

# Model can be shared (stateless)
model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")

async def handle_agent(request):
    data = await request.json()
    task = data['task']
    # Create a new agent instance for each request
    agent = CodeAgent(model=model, tools=[])
    result = await agent.arun(task)
    return web.json_response({'result': result})

app = web.Application()
app.router.add_post('/run-agent', handle_agent)
web.run_app(app)
```

## Implementation Details

### Backward Compatibility

All sync methods remain unchanged. Async methods are additive:
- Sync: `run()`, `generate()`, `generate_stream()`
- Async: `arun()`, `agenerate()`, `agenerate_stream()`

### Internal Changes

1. **Models**: Async methods use native async clients:
   - LiteLLM: `litellm.acompletion()`
   - OpenAI: `openai.AsyncOpenAI()`
   - HuggingFace: `AsyncInferenceClient()`

2. **Agents**: Async execution flow:
   - `arun()` → `_arun_stream()` → `_astep_stream()`
   - Async generators for streaming
   - Async helper methods: `aprovide_final_answer()`, `_agenerate_planning_step()`

3. **Rate Limiting**: Rate limiters are synchronous (apply before async calls)

### Current Limitations

1. **Stateful Agent Memory**: Each concurrent task requires a separate agent instance
   - Agents maintain state in `self.memory`, `self.state`, and `self.step_number`
   - Sharing an agent across concurrent tasks causes memory corruption
   - **Workaround**: Create separate agent instances; models can be shared
   - **Future**: Could implement per-task memory isolation using contextvars

2. **Planning**: Async planning doesn't support Live display streaming (simplified mode only)

3. **Tools**: Tools are currently synchronous only
   - `Tool.__call__()` is not async, so tools block during execution
   - **Use case impact**: Human-in-the-loop workflows (waiting for approvals, input) will block
   - **Workaround**: Tools return immediately and poll/check status separately
   - **Future**: Async tool support would enable non-blocking `await` for long-running tools

4. **CodeAgent/ToolCallingAgent**: Need async implementations of `_astep_stream()`

5. **Remote Executors**: E2B, Modal, Docker executors don't yet support async

6. **Local Executors**: LocalPythonExecutor remains synchronous (executes arbitrary code)

## Migration Guide

### From Sync to Async

**Before (sync):**
```python
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")
agent = CodeAgent(model=model, tools=[])

result = agent.run("What is 2 + 2?")
print(result)
```

**After (async):**
```python
import asyncio
from smolagents import CodeAgent, LiteLLMModel

async def main():
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620")
    agent = CodeAgent(model=model, tools=[])

    result = await agent.arun("What is 2 + 2?")
    print(result)

asyncio.run(main())
```

### When to Use Async

**Use async when:**
- Running multiple agents concurrently
- Integrating with async frameworks
- Building high-throughput applications
- Network I/O is the bottleneck

**Use sync when:**
- Simple scripts
- Single agent execution
- Synchronous codebase
- Simplicity is preferred

## Best Practices

1. **Use asyncio.gather() for concurrency:**
   ```python
   results = await asyncio.gather(*[agent.arun(task) for task in tasks])
   ```

2. **Handle errors properly:**
   ```python
   try:
       result = await agent.arun(task)
   except Exception as e:
       print(f"Error: {e}")
   ```

3. **Use context managers for cleanup:**
   ```python
   async with aiohttp.ClientSession() as session:
       # Use session
       pass
   ```

4. **Limit concurrency to avoid rate limits:**
   ```python
   import asyncio

   async def run_with_limit(tasks, limit=5):
       semaphore = asyncio.Semaphore(limit)

       async def run_task(task):
           async with semaphore:
               return await agent.arun(task)

       return await asyncio.gather(*[run_task(task) for task in tasks])
   ```

## Troubleshooting

### Incorrect results or mixed task context

**Problem:** When running concurrent tasks, results are incorrect or contain information from other tasks.

**Cause:** You're reusing the same agent instance across concurrent tasks, causing memory corruption.

**Solution:** Create a separate agent instance for each concurrent task:

```python
# ❌ WRONG - reuses same agent
agent = CodeAgent(model=model, tools=[])
results = await asyncio.gather(*[agent.arun(task) for task in tasks])

# ✅ CORRECT - separate agent per task
agents = [CodeAgent(model=model, tools=[]) for _ in tasks]
results = await asyncio.gather(*[agent.arun(task) for agent, task in zip(agents, tasks)])
```

### Error: "This method must be implemented in child classes that support async"

This means the agent class doesn't yet have an async implementation of `_astep_stream()`. Currently, only the base `Agent` class has full async support. Implementations for `CodeAgent` and `ToolCallingAgent` are in progress.

### Error: "agenerate() is not implemented"

The model class doesn't support async yet. Check if you're using a supported model (see "Supported Models" above).

### Performance not improving with async

- Check if you're actually running tasks concurrently with `asyncio.gather()`
- Ensure you're creating separate agent instances (sharing causes serialization)
- Ensure rate limits aren't throttling concurrent requests
- Verify network I/O is the bottleneck (not CPU-bound operations)

## Contributing

To add async support to a component:

1. Add async methods with "a" prefix: `agenerate()`, `arun()`, etc.
2. Use async/await syntax throughout
3. Replace blocking I/O with async equivalents (aiohttp, aiofiles, etc.)
4. Test with multiple concurrent calls
5. Maintain backward compatibility (keep sync methods)

## Future Work

- [ ] Complete async support for CodeAgent._astep_stream()
- [ ] Complete async support for ToolCallingAgent._astep_stream()
- [ ] Async remote executors (E2B, Modal, Docker)
- [ ] Async file operations in tools
- [ ] Async tool execution with asyncio.gather()
- [ ] Live display support for async streaming
- [ ] Async monitoring and telemetry

## References

- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [LiteLLM async support](https://docs.litellm.ai/docs/completion/stream)
- [OpenAI Python async](https://github.com/openai/openai-python#async-usage)
- [aiohttp documentation](https://docs.aiohttp.org/)
