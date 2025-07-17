# Async Applications with Agents

This guide demonstrates how to integrate a synchronous agent from the `smolagents` library into an asynchronous Python web application using Starlette.
The example is designed to help users new to async Python and agent integration understand best practices for combining synchronous agent logic with async web servers.

## Overview

- **Starlette**: A lightweight ASGI framework for building asynchronous web applications in Python.
- **anyio.to_thread.run_sync**: Utility to run blocking (synchronous) code in a background thread, preventing it from blocking the async event loop.
- **CodeAgent**: An agent from the `smolagents` library capable of programmatically solving tasks.

## Why Use a Background Thread?

`CodeAgent.run()` executes Python code synchronously. If called directly in an async endpoint, it would block Starlette's event loop, reducing performance and scalability. By offloading this operation to a background thread with `anyio.to_thread.run_sync`, you keep the app responsive and efficient, even under high concurrency.

## Example Workflow

- The Starlette app exposes a `/run-agent` endpoint that accepts a JSON payload with a `task` string.
- When a request is received, the agent is run in a background thread using `anyio.to_thread.run_sync`.
- The result is returned as a JSON response.

## Building a Starlette App with a CodeAgent

### 1. Install Dependencies

```bash
pip install smolagents starlette anyio uvicorn
```

### 2. Application Code (`main.py`)

```python
import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(
    model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
    tools=[],
)

async def run_agent(request: Request):
    data = await request.json()
    task = data.get("task", "")
    # Run the agent synchronously in a background thread
    result = await anyio.to_thread.run_sync(agent.run, task)
    return JSONResponse({"result": result})

app = Starlette(routes=[
    Route("/run-agent", run_agent, methods=["POST"]),
])
```

### 3. Run the App

```bash
uvicorn async_agent.main:app --reload
```

### 4. Test the Endpoint

```bash
curl -X POST http://localhost:8000/run-agent -H 'Content-Type: application/json' -d '{"task": "What is 2+2?"}'
```

**Expected Response:**

```json
{"result": "4"}
```

## Further Reading

- [Starlette Documentation](https://www.starlette.io/)
- [anyio Documentation](https://anyio.readthedocs.io/)

---

For the full code, see [`examples/async_agent`](https://github.com/huggingface/smolagents/tree/main/examples/async_agent).
