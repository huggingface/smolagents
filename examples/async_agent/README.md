# Async Applications with Agents

This example demonstrates **two approaches** for using agents in asynchronous Starlette web applications:

1. **Native async support** (recommended) - Uses agents with async tools
2. **Threading approach** (legacy) - Runs sync agents in background threads

## Key Concepts

- **Starlette**: Lightweight ASGI framework for async web apps
- **Native async**: Agents natively support async/await with async tools
- **Async tools**: Tools that can await human input, external APIs, or long-running operations
- **Threading (legacy)**: For backward compatibility with sync-only tools

## How it works

The app exposes two endpoints:

### `/run-agent` - Native Async (Recommended)

- Uses `CodeAgent.arun()` with async tools
- **Non-blocking I/O**: While waiting for approval, event loop can handle other requests
- **Memory efficient**: Async tasks use ~few KB vs threads at ~1-8MB each
- **Perfect for**: Human-in-the-loop workflows, API calls, long-running operations

### `/run-agent-threading` - Threading (Legacy)

- Uses `anyio.to_thread.run_sync()` to run sync agent in background thread
- **Backward compatible**: Works with sync-only tools
- **Resource intensive**: Each thread consumes 1-8MB of memory
- **Use when**: You have legacy sync code that can't be made async

## Why Native Async?

**Native async approach:**
```python
# Agent with async tool
agent = CodeAgent(model=model, tools=[HumanApprovalTool()])
result = await agent.arun(task)  # Efficient non-blocking I/O
```

**Threading approach (legacy):**
```python
# Sync agent in background thread
agent = CodeAgent(model=model, tools=[])
result = await anyio.to_thread.run_sync(agent.run, task)  # Blocks thread
```

## Usage

1. **Install dependencies**:
   ```bash
   pip install smolagents starlette anyio uvicorn
   ```

2. **Run the app**:
   ```bash
   uvicorn async_agent.main:app --reload
   ```

3. **Test native async endpoint** (recommended):
   ```bash
   curl -X POST http://localhost:8000/run-agent \
     -H 'Content-Type: application/json' \
     -d '{"task": "Get approval to delete user account"}'
   ```

   **Response:**
   ```json
   {"result": "approved: delete user account", "approach": "native-async"}
   ```

4. **Test threading endpoint** (legacy):
   ```bash
   curl -X POST http://localhost:8000/run-agent-threading \
     -H 'Content-Type: application/json' \
     -d '{"task": "What is 2+2?"}'
   ```

   **Response:**
   ```json
   {"result": "4", "approach": "threading"}
   ```

## Performance Comparison

| Metric | Native Async | Threading |
|--------|-------------|-----------|
| **Memory per request** | ~Few KB | ~1-8 MB |
| **Concurrent requests** | Thousands | Hundreds |
| **Async tools** | ✅ Supported | ❌ Blocks thread |
| **Best for** | Modern async tools | Legacy sync code |

## Files

- `main.py`: Starlette application showing both approaches
- `README.md`: This file

---

For more examples, see [`examples/async_agent_example.py`](../async_agent_example.py) for comprehensive async patterns.
