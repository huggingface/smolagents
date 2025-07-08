"""
Async CodeAgent Example with Starlette

This example demonstrates how to use a CodeAgent in an async Starlette app,
running the agent in a background thread using anyio.to_thread.run_sync.
"""

import anyio.to_thread
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from smolagents import CodeAgent, InferenceClientModel


# Create a simple agent instance (customize as needed)
def get_agent():
    # You can set custom model, or tools as needed
    return CodeAgent(
        model=InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct"),
        tools=[],
    )


async def run_agent_in_thread(task: str):
    agent = get_agent()
    # The agent's run method is synchronous
    result = await anyio.to_thread.run_sync(agent.run, task)
    return result


async def run_agent_endpoint(request: Request):
    data = await request.json()
    task = data.get("task")
    if not task:
        return JSONResponse({"error": 'Missing "task" in request body.'}, status_code=400)
    try:
        result = await run_agent_in_thread(task)
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


routes = [
    Route("/run-agent", run_agent_endpoint, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)
