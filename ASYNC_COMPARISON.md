# Async Implementation Comparison: PR #1669 vs Our Implementation

## Overview

There are two different approaches to adding async support to smolagents:
- **PR #1669** by hkjeon13 (opened Aug 2025)
- **Our implementation** (current branch)

## Key Differences

### Architecture Approach

| Aspect | PR #1669 | Our Implementation |
|--------|----------|-------------------|
| **Structure** | Separate `smolagents.asynchronous` module | Async methods added to existing classes |
| **Classes** | New: `AsyncModel`, `AsyncToolCallingAgent`, `AsyncCodeAgent` | Enhanced: `Model.agenerate()`, `Agent.arun()`, etc. |
| **Code Duplication** | Duplicate async classes | DRY approach - shared code between sync/async |
| **Import Path** | `from smolagents.asynchronous import AsyncCodeAgent` | `from smolagents import CodeAgent; await agent.arun()` |

### Implementation Details

#### PR #1669 Approach
```python
# Separate async classes
from smolagents.asynchronous.agents import AsyncCodeAgent
from smolagents.asynchronous.models import AsyncOpenAIModel

model = AsyncOpenAIModel(...)
agent = AsyncCodeAgent(model=model, tools=[])
result = await agent.run(task)  # Different class, different API
```

**Pros:**
- Complete separation - no risk of breaking sync code
- Clear distinction between sync and async usage
- Can optimize async implementation independently

**Cons:**
- Code duplication across sync and async classes
- Maintenance burden - changes need to be made in two places
- Users need to choose between `CodeAgent` and `AsyncCodeAgent`
- More complex import structure

#### Our Implementation Approach
```python
# Same classes with async methods
from smolagents import CodeAgent, LiteLLMModel

model = LiteLLMModel(...)
agent = CodeAgent(model=model, tools=[])

# Use sync or async as needed
result_sync = agent.run(task)      # Sync API
result_async = await agent.arun(task)  # Async API
```

**Pros:**
- Single class supports both sync and async
- DRY principle - shared code, single source of truth
- Users can mix sync/async in same codebase
- Simpler imports and API
- Easier to maintain - one place to fix bugs

**Cons:**
- More complex class implementation
- Need to ensure sync/async code paths don't interfere

### Feature Comparison

| Feature | PR #1669 | Our Implementation |
|---------|----------|-------------------|
| **Async Models** | ✅ Separate AsyncModel classes | ✅ `Model.agenerate()` method |
| **Async Agents** | ✅ Separate AsyncAgent classes | ✅ `Agent.arun()` method |
| **Async Tools** | ❓ Unknown | ✅ Tools with async `forward()` |
| **Transparent Execution** | ❓ Unknown | ✅ Executor auto-awaits coroutines |
| **LocalPythonExecutor** | ✅ Added `run_async` method | ✅ Added `async_call()` + async evaluators |
| **Rate Limiting** | ✅ Added `athrottle` | ❌ Not implemented |
| **Documentation** | ❌ Not updated | ✅ Comprehensive docs + examples |
| **Tests** | ✅ test_async_agents.py, test_async_model.py | ✅ test_async.py (31 tests) |

## Technical Innovation

### Our Key Innovation: Transparent Async Execution

One of the most significant features in our implementation is **transparent async tool execution** in CodeAgent:

```python
# Generated code looks synchronous (no await needed!)
result = human_approval("delete file")
final_answer(result)

# Executor automatically detects and awaits async tools
# (lines 1817-1818 in local_python_executor.py)
result = func(*args, **kwargs)
if inspect.iscoroutine(result):
    result = await result  # Automatic!
```

This means:
- LLM doesn't need to generate `await` keywords
- Same generated code works for both sync and async tools
- Seamless backward compatibility

**PR #1669 Status:** Unknown if they implemented this feature.

## Backward Compatibility

| Aspect | PR #1669 | Our Implementation |
|--------|----------|-------------------|
| **Existing Code** | ✅ Unchanged (new module) | ✅ Unchanged (added methods) |
| **Migration Path** | New imports required | Add `await` to calls |
| **Mixed Usage** | Use both modules | Use sync/async methods |

## Community Status

### PR #1669
- **Status:** Open since Aug 10, 2025
- **Commits:** 18 commits (iterative development)
- **Files Changed:** 6 files (+2,992, -1)
- **Reviews:** 2 comments
- **Mergeable:** "dirty" status (needs resolution)
- **Author Note:** "has been working fine" in their projects

### Our Implementation
- **Status:** Ready for PR
- **Commits:** 9 commits (planned approach)
- **Files Changed:** 8 files
- **Tests:** 31 passing tests
- **Documentation:** Comprehensive guides + examples
- **Mergeable:** Clean working tree

## Recommendation

Both approaches are valid but serve different philosophies:

### Choose PR #1669 If:
- You want complete separation between sync and async
- You're okay with maintaining duplicate classes
- You prefer explicit async-specific imports

### Choose Our Implementation If:
- You want a single unified API
- You prefer the DRY principle
- You want transparent async tool execution
- You need comprehensive documentation
- You want to mix sync/async in the same codebase

## Potential Path Forward

The maintainers might consider:

1. **Merge our implementation** - More aligned with Python's standard approach (e.g., `httpx` has sync/async methods on same client)
2. **Merge PR #1669** - If they prefer complete separation
3. **Hybrid approach** - Take best ideas from both (e.g., our transparent execution + their rate limiting)

The Python community trend is toward **unified APIs** (like `httpx.Client` having both sync and async methods), which aligns with our approach.
