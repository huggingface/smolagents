# Async Applications with Agents

This guide demonstrates how to integrate a synchronous agent from the `smolagents` library into an asynchronous Python web application using Starlette. The example is designed to be clear and educational for users new to async Python and agent-based architectures.

## Overview

- **Starlette**: A lightweight ASGI framework for building async web apps.
- **anyio.to_thread.run_sync**: Runs blocking (sync) code in a thread, so it doesn't block the async event loop.
- **CodeAgent**: An agent from the `smolagents` library that can be used to solve tasks programmatically.

## Why Use a Background Thread?

`CodeAgent.run()` executes Python code synchronously. If called directly in an async endpoint, it would block Starlette's event loop, reducing performance and scalability. By offloading this operation to a background thread with `anyio.to_thread.run_sync`, you keep the app responsive and efficient, even under high concurrency.

## How the Example Works

- The Starlette app exposes a `/run-agent` endpoint that accepts a JSON payload with a `task` string.
- When a request is received, the agent is run in a background thread using `anyio.to_thread.run_sync`.
- The result is returned as a JSON response.

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

## Files in This Example

- `main.py`: Main Starlette application with async endpoint using CodeAgent.
- `README.md`: Example usage and explanation (this file).
