# Add comprehensive async support with async tools and transparent execution

## Summary

This PR adds comprehensive async/await support to smolagents, enabling efficient non-blocking I/O for human-in-the-loop workflows, external API calls, and concurrent agent execution.

### Key Features

1. **Async Tools** - Tools can now have async `forward()` methods for:
   - Human-in-the-loop approval workflows
   - External API calls with network latency
   - Long-running operations without blocking
   - Database queries and I/O operations

2. **Transparent Async Execution** - LLM-generated code doesn't need `await`:
   - Executor automatically detects and awaits coroutines
   - Works seamlessly with both sync and async tools
   - No changes required to generated code patterns

3. **Async CodeAgent Support** - Full async execution pipeline:
   - `CodeAgent._astep_stream()` for async execution
   - `LocalPythonExecutor.async_call()` for transparent async handling
   - Automatic coroutine detection and awaiting in `evaluate_call_async()`

4. **Updated Documentation** - Comprehensive guides and examples:
   - Native async vs threading comparison
   - Real-world usage patterns
   - Performance benchmarks and best practices

### Technical Details

#### Tool Layer
Tools detect async `forward()` methods and return coroutines:

```python
class HumanApprovalTool(Tool):
    name = "human_approval"

    async def forward(self, action: str):
        # Wait for approval without blocking
        await approval_queue.get()
        return "approved"
```

#### Agent Layer
Agents await tool results using `async_execute_tool_call()`:

```python
# In ToolCallingAgent
result = tool(**arguments)
if inspect.iscoroutine(result):
    result = await result
```

#### Executor Layer (Key Innovation)
The executor automatically awaits coroutines in generated code:

```python
# In evaluate_call_async() (lines 1817-1818 in local_python_executor.py)
result = func(*args, **kwargs)
if inspect.iscoroutine(result):
    result = await result  # Automatic await!
```

This means generated code looks synchronous:
```python
# LLM generates this (no await needed!)
result = human_approval("delete file")
final_answer(result)

# Executor awaits automatically if human_approval is async
```

### Benefits

| Feature | Native Async | Threading |
|---------|-------------|-----------|
| **Async tools** | ✅ Full support | ❌ Blocks thread |
| **Memory per task** | ~Few KB | ~1-8 MB |
| **Context switching** | Efficient (event loop) | OS overhead |
| **Scalability** | Thousands of tasks | Hundreds of threads |
| **Non-blocking I/O** | ✅ Event loop switches during waits | ❌ Thread blocks |

### Changes

#### Core Implementation
- `src/smolagents/tools.py` - Async tool detection in `Tool.__call__()`
- `src/smolagents/agents.py` - `async_execute_tool_call()` and `CodeAgent._astep_stream()`
- `src/smolagents/local_python_executor.py` - Async evaluation functions for transparent execution

#### Documentation
- `docs/async_support.md` - Updated to emphasize non-blocking I/O benefits
- `docs/source/en/examples/async_agent.md` - Shows both native async and threading approaches
- `examples/async_agent/` - Updated Starlette example with both patterns
- `examples/async_agent_example.py` - Comprehensive async examples with working tools

#### Tests
- Added `TestAsyncCodeAgent` with 6 tests for CodeAgent async support
- Added `TestAsyncRealWorldPatterns` with human approval and API call examples
- Added tests demonstrating concurrent execution and non-blocking I/O benefits
- All new async tests pass (24 passing tests total)

### Usage Examples

#### Simple Async Tool
```python
from smolagents import CodeAgent, Tool, LiteLLMModel

class HumanApprovalTool(Tool):
    name = "human_approval"
    async def forward(self, action: str):
        await asyncio.sleep(0.5)  # Simulate waiting
        return f"approved: {action}"

agent = CodeAgent(
    model=LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-20240620"),
    tools=[HumanApprovalTool()]
)

# Run asynchronously - handles async tools transparently!
result = await agent.arun("Get approval to delete account")
```

#### Concurrent Agents with Non-Blocking I/O
```python
# Create separate agent instances (required - memory is stateful)
agents = [CodeAgent(model=model, tools=[HumanApprovalTool()]) for _ in range(3)]
tasks = ["task1", "task2", "task3"]

# Run concurrently - event loop switches during I/O waits
results = await asyncio.gather(*[
    agent.arun(task) for agent, task in zip(agents, tasks)
])
```

### Backward Compatibility

✅ **Fully backward compatible**:
- Sync tools continue to work normally
- Mixed sync/async tools supported
- Sync methods (`run()`, `execute_tool_call()`) unchanged
- Threading approach still available for legacy code

### Test Results

```bash
$ pytest tests/test_async.py::TestAsyncCodeAgent -v
======================== 6 passed in 0.12s =========================

$ pytest tests/test_async.py::TestAsyncRealWorldPatterns -v
======================== 2 passed in 0.16s =========================
```

### Commits

1. `3f25cf5` - Add comprehensive async support to smolagents
2. `acaac7c` - Refactor async/sync methods to eliminate duplication (DRY)
3. `af77ac7` - Further DRY refactoring: extract common logic
4. `9ef1bc6` - Fix async documentation to accurately describe benefits
5. `b1abf2b` - Document that tools are currently synchronous
6. `3a8d8d9` - Add async tool support for human-in-the-loop
7. `890749b` - Add async CodeAgent support for transparent execution
8. `f646064` - Update documentation, examples, and tests

### Related

This implements the async support needed for efficient human-in-the-loop workflows and addresses the limitation where tools could only be synchronous.
