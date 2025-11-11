# Async Applications with Agents

This guide demonstrates two approaches for using agents in asynchronous Python applications:
1. **Native async support** - Using agents with async/await (recommended)
2. **Threading approach** - Running sync agents in background threads (backward compatibility)

## Overview

Smolagents now supports async/await natively, enabling:
- **Async tools**: Tools can await human input, external APIs, or long-running operations
- **Non-blocking I/O**: While one agent waits, the event loop switches to other agents
- **Transparent execution**: Generated code doesn't need `await` - the executor handles it automatically

## Approach 1: Native Async Support (Recommended)

Use this when you want efficient non-blocking I/O and have async tools.

### Example: Async Tools for Human-in-the-Loop

```python
import asyncio
from smolagents import CodeAgent, Tool, LiteLLMModel

class HumanApprovalTool(Tool):
    name = "human_approval"
    description = "Request human approval for an action"
    inputs = {"action": {"type": "string", "description": "Action requiring approval"}}
    output_type = "string"

    async def forward(self, action: str):
        """Async tool - waits for approval without blocking"""
        # In production: await redis_queue.get(f"approval:{id}")
        await asyncio.sleep(0.5)  # Simulate waiting
        return f"approved: {action}"

# Create agent with async tool
agent = CodeAgent(
    model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620"),
    tools=[HumanApprovalTool()]
)

# Run asynchronously - automatically handles async tools!
result = await agent.arun("Get approval to delete user account")
```

### Key Benefits

- **Non-blocking**: While waiting for approval, other agents can run
- **Memory efficient**: Async tasks use ~few KB vs threads at 1-8MB each
- **Transparent**: Generated code calls tools normally (no `await` needed)
- **Human-in-the-loop**: Perfect for approval workflows, interactive tools

## Approach 2: Threading (For Sync-Only Code)

Use this when you have legacy sync code or can't use async.

### Why Use a Background Thread?

`CodeAgent.run()` without async tools executes synchronously. In an async web server, this would block the event loop. Using `anyio.to_thread.run_sync` offloads sync work to a thread, keeping the app responsive.

## Building a Starlette App with Native Async

### 1. Install Dependencies

```bash
pip install smolagents starlette uvicorn
```

### 2. Application Code with Native Async (`main.py`)

```python
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, Tool, LiteLLMModel
import asyncio

class HumanApprovalTool(Tool):
    name = "human_approval"
    description = "Request human approval"
    inputs = {"action": {"type": "string"}}
    output_type = "string"

    async def forward(self, action: str):
        # In production: await your approval queue
        await asyncio.sleep(0.1)
        return f"approved: {action}"

# Create agent with async tools
agent = CodeAgent(
    model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620"),
    tools=[HumanApprovalTool()]
)

async def run_agent(request: Request):
    data = await request.json()
    task = data.get("task", "")
    # Run agent natively with async - no threading needed!
    result = await agent.arun(task)
    return JSONResponse({"result": result})

app = Starlette(routes=[
    Route("/run-agent", run_agent, methods=["POST"]),
])
```

### Threading Approach (Legacy)

For backward compatibility with sync-only agents:

```python
import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
    tools=[],  # Sync tools only
)

async def run_agent(request: Request):
    data = await request.json()
    task = data.get("task", "")
    # Run sync agent in background thread
    result = await anyio.to_thread.run_sync(agent.run, task)
    return JSONResponse({"result": result})

app = Starlette(routes=[
    Route("/run-agent", run_agent, methods=["POST"]),
])
```

### 3. Run the App

```bash
uvicorn main:app --reload
```

### 4. Test the Endpoint

**With native async:**
```bash
curl -X POST http://localhost:8000/run-agent \
  -H 'Content-Type: application/json' \
  -d '{"task": "Get approval to delete account"}'
```

**Expected Response:**
```json
{"result": "approved: delete account"}
```

## Comparison: Threading vs Native Async

| Feature | Native Async | Threading |
|---------|-------------|-----------|
| **Async tools** | ✅ Full support | ❌ Blocks thread |
| **Memory per task** | ~Few KB | ~1-8 MB |
| **Context switching** | Efficient (event loop) | OS overhead |
| **Scalability** | Thousands of tasks | Hundreds of threads |
| **Best for** | Human-in-the-loop, API calls | Legacy sync code |

## Further Reading

- [Async Support Documentation](../../../async_support.md)
- [Starlette Documentation](https://www.starlette.io/)
- [Python Asyncio Guide](https://docs.python.org/3/library/asyncio.html)

---

For full examples, see:
- [`examples/async_agent_example.py`](https://github.com/huggingface/smolagents/tree/main/examples/async_agent_example.py) - Comprehensive async examples
- [`examples/async_agent`](https://github.com/huggingface/smolagents/tree/main/examples/async_agent) - Starlette integration
