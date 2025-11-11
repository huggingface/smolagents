# Async Applications with Agents

This example demonstrates **two approaches** for integrating agents in asynchronous Starlette web applications:

1. **Native async** - Use `await agent.arun()` for async tools and non-blocking I/O
2. **Threading** - Use `await anyio.to_thread.run_sync(agent.run, ...)` to run sync operations in async contexts

## Key Concepts

- **Starlette**: Lightweight ASGI framework for async web apps
- **Native async**: Agents support `arun()` for async tools with non-blocking I/O
- **Async tools**: Tools that can await human input, external APIs, or long-running operations
- **Threading**: Run sync operations (like `agent.run()`) in background threads from async contexts

## How it works

The app exposes two endpoints:

### `/run-agent` - Native Async (For Async Tools)

- Uses `await agent.arun(task)` with async tools
- **Non-blocking I/O**: While waiting for async tools, event loop can handle other requests
- **Memory efficient**: Async tasks use ~few KB vs threads at ~1-8MB each
- **Perfect for**: Async tools (human-in-the-loop, API calls, long-running operations)

### `/run-agent-threading` - Threading (For Sync Operations)

- Uses `await anyio.to_thread.run_sync(agent.run, task)` to run sync operations
- **Valid pattern**: Run sync `agent.run()` in background thread from async context
- **Resource intensive**: Each thread consumes 1-8MB of memory
- **Use when**: You need to call sync operations from async contexts, or have sync-only tools

## Choosing the Right Approach

**When to use native async:**
```python
# Use when you have async tools (recommended for new code)
agent = CodeAgent(model=model, tools=[HumanApprovalTool()])
result = await agent.arun(task)  # Efficient non-blocking I/O
```

**When to use threading:**
```python
# Use when calling sync operations from async contexts
agent = CodeAgent(model=model, tools=[])
result = await anyio.to_thread.run_sync(agent.run, task)  # Runs in thread
```

**Both are valid!** Threading isn't just for legacy code - use it whenever you need to call sync operations from async contexts (like async web servers).

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

4. **Test threading endpoint**:
   ```bash
   curl -X POST http://localhost:8000/run-agent-threading \
     -H 'Content-Type: application/json' \
     -d '{"task": "What is 2+2?"}'
   ```

   **Response:**
   ```json
   {"result": "4", "approach": "threading"}
   ```

## Comparison

| Metric | Native Async (`arun`) | Threading (`run_sync`) |
|--------|-------------|-----------|
| **Memory per request** | ~Few KB | ~1-8 MB |
| **Concurrent requests** | Thousands | Hundreds |
| **Async tools** | ✅ Non-blocking | ❌ Blocks thread |
| **Sync operations** | ❌ Would block | ✅ Runs in thread |
| **Best for** | Async tools, I/O operations | Sync operations in async contexts |

## Files

- `main.py`: Starlette application showing both approaches
- `README.md`: This file

---

For more examples, see [`examples/async_agent_example.py`](../async_agent_example.py) for comprehensive async patterns.
