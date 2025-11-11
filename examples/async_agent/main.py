"""
Async CodeAgent Example with Starlette

This example demonstrates two approaches:
1. Native async support with async tools (recommended)
2. Threading approach for backward compatibility (legacy)

Choose the approach that fits your use case:
- Use native async if you have async tools (human-in-the-loop, API calls, etc.)
- Use threading if you have only sync tools or legacy code
"""

import asyncio
import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, Tool, LiteLLMModel, InferenceClientModel


# Example async tool for human-in-the-loop workflows
class HumanApprovalTool(Tool):
    name = "human_approval"
    description = "Request human approval for an action"
    inputs = {"action": {"type": "string", "description": "Action requiring approval"}}
    output_type = "string"

    async def forward(self, action: str):
        """Async tool that waits for human approval without blocking."""
        # In production: await redis_queue.get(f"approval:{action_id}")
        # For demo: simulate waiting for approval
        await asyncio.sleep(0.1)
        return f"approved: {action}"


# Approach 1: Native async agent (recommended)
def get_async_agent():
    """Create agent with async tools for native async support."""
    return CodeAgent(
        model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620"),
        tools=[HumanApprovalTool()],
    )


# Approach 2: Threading agent (legacy)
def get_sync_agent():
    """Create agent with sync tools for threading approach."""
    return CodeAgent(
        model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
        tools=[],
    )


async def run_agent_native_async(task: str):
    """Run agent natively with async - no threading needed!"""
    agent = get_async_agent()
    result = await agent.arun(task)
    return result


async def run_agent_in_thread(task: str):
    """Run sync agent in background thread (legacy approach)."""
    agent = get_sync_agent()
    result = await anyio.to_thread.run_sync(agent.run, task)
    return result


async def run_agent_endpoint(request: Request):
    """Endpoint using native async approach (recommended)."""
    data = await request.json()
    task = data.get("task")
    if not task:
        return JSONResponse({"error": 'Missing "task" in request body.'}, status_code=400)
    try:
        # Native async - efficient for async tools
        result = await run_agent_native_async(task)
        return JSONResponse({"result": result, "approach": "native-async"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def run_agent_threading_endpoint(request: Request):
    """Endpoint using threading approach (legacy)."""
    data = await request.json()
    task = data.get("task")
    if not task:
        return JSONResponse({"error": 'Missing "task" in request body.'}, status_code=400)
    try:
        # Threading - for sync-only tools
        result = await run_agent_in_thread(task)
        return JSONResponse({"result": result, "approach": "threading"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/run-agent", run_agent_endpoint, methods=["POST"]),
    Route("/run-agent-threading", run_agent_threading_endpoint, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)
