# Async Applications with Agents

This example demonstrates how to use a `CodeAgent` from the `smolagents` library in an asynchronous Starlette web application.
The agent is executed in a background thread using `anyio.to_thread.run_sync`, allowing you to integrate synchronous agent logic into an async web server.

## Key Concepts

- **Starlette**: A lightweight ASGI framework for building async web apps.
- **anyio.to_thread.run_sync**: Runs blocking (sync) code in a thread, so it doesn't block the async event loop.
- **CodeAgent**: An agent from the `smolagents` library that can be used to solve tasks programmatically.

## How it works

- The Starlette app exposes a `/run-agent` endpoint that accepts a JSON payload with a `task` string.
- When a request is received, the agent is run in a background thread using `anyio.to_thread.run_sync`.
- The result is returned as a JSON response.

## Implementation Note

**Why use a background thread?** 

`CodeAgent.run()` executes Python code synchronously, which would block Starlette's async event loop if called directly. By offloading this synchronous operation to a separate thread with `anyio.to_thread.run_sync`, we maintain the application's responsiveness while the agent processes requests, ensuring optimal performance in high-concurrency scenarios.

## Usage

1. **Install dependencies**:
   ```bash
   pip install smolagents starlette anyio uvicorn
   ```

2. **Run the app**:
   ```bash
   uvicorn async_codeagent_starlette.main:app --reload
   ```

3. **Test the endpoint**:
   ```bash
   curl -X POST http://localhost:8000/run-agent -H 'Content-Type: application/json' -d '{"task": "What is 2+2?"}'
   ```

## Files

- `main.py`: Main Starlette application with async endpoint using CodeAgent.
- `README.md`: This file.

---
This example is designed to be clear and didactic for users new to async Python and agent integration.
