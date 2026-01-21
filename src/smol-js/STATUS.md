# smol-js Status Report

**Date**: January 21, 2026
**Version**: 0.2.0
**npm**: [@samrahimi/smol-js](https://www.npmjs.com/package/@samrahimi/smol-js)

---

## Session Summary

This document summarizes the development session that created smol-js, a TypeScript port of the Python [smolagents](https://github.com/huggingface/smolagents) framework by Hugging Face.

---

## What Was Accomplished

### Core Implementation

| Component | Status | Description |
|-----------|--------|-------------|
| **CodeAgent** | ✅ Complete | Main agent class implementing the ReAct loop |
| **LocalExecutor** | ✅ Complete | Sandboxed JavaScript execution via Node's vm module |
| **Tool Base Class** | ✅ Complete | Extensible tool system with validation |
| **AgentTool** | ✅ Complete | Wrap agents as tools for nested architectures |
| **OpenAIModel** | ✅ Complete | OpenAI-compatible API client (OpenRouter default) |
| **AgentMemory** | ✅ Complete | Conversation history and step tracking |
| **AgentLogger** | ✅ Complete | Color-coded console logging with chalk |
| **System Prompts** | ✅ Complete | Code agent prompt generation |

### Key Features Implemented

1. **ReAct Execution Loop**
   - Thought → Code → Observation → repeat
   - Automatic error recovery and retry
   - Configurable max steps

2. **Sandboxed Code Execution**
   - Node.js vm module isolation
   - Variable persistence between steps
   - Tool injection as async functions
   - Built-in `fetch()` for HTTP requests

3. **Dynamic npm Imports**
   - Fetch packages from jsdelivr CDN
   - Local caching in `~/.smol-js/packages/`
   - Authorization whitelist for security

4. **Nested Agents (AgentTool)**
   - Wrap any Agent as a Tool
   - Manager/worker delegation pattern
   - Hierarchical task decomposition

5. **Developer Experience**
   - Color-coded terminal output
   - Session logging to `~/.smol-js/`
   - Configurable execution delay for safety
   - Streaming LLM output support

### Testing

- **51 unit tests** - All passing
- Tests cover: LocalExecutor, Tool, AgentMemory, CodeAgent, prompts

### Examples Created

| # | Example | Description |
|---|---------|-------------|
| 01 | simple-math | Basic calculation task |
| 02 | dynamic-imports | Using npm packages (lodash, dayjs) |
| 03 | variable-persistence | Multi-step state management |
| 04 | research-with-tools | Custom tools for web research |
| 05 | error-recovery | Handling and recovering from errors |
| 06 | deep-research | Real API calls (DuckDuckGo, Wikipedia) |
| 07 | npm-package-import | Using the published npm package |
| 08 | fetch-agent | Agent using fetch() directly (no tools) |
| 09 | nested-agents | Manager delegating to worker agents |

### npm Publication

- **v0.1.0** - Initial release
- **v0.2.0** - Added AgentTool for nested agents

---

## smolagents Feature Parity

### Available in smol-js ✅

| Feature | Python | smol-js | Notes |
|---------|--------|---------|-------|
| CodeAgent | `CodeAgent` | `CodeAgent` | Full implementation |
| Tool class | `Tool` / `@tool` | `Tool` / `createTool()` | Class-based (no decorators in TS) |
| Managed agents | `ManagedAgent` | `AgentTool` | Agents as tools |
| Local executor | `LocalPythonInterpreter` | `LocalExecutor` | vm module sandbox |
| Memory system | `AgentMemory` | `AgentMemory` | Step tracking |
| Streaming | Yes | Yes | Via async generators |
| Custom prompts | Yes | Yes | `customInstructions` |
| Multi-step | Yes | Yes | Variable persistence |
| Error recovery | Yes | Yes | Automatic retry |
| Logging | Yes | Yes | Color-coded + file |

### Not Yet Implemented ❌

| Feature | Python | Notes |
|---------|--------|-------|
| **ToolCallingAgent** | `ToolCallingAgent` | Uses function calling instead of code generation |
| **Remote Executors** | E2B, Docker, Modal | Only local execution currently |
| **Vision Support** | Image inputs | Not implemented |
| **Gradio UI** | `GradioUI` | No built-in UI |
| **Tool collections** | `DuckDuckGoSearchTool`, etc. | Must create custom tools |
| **Planning agent** | `PlanningAgent` | Not implemented |
| **Retrieval** | RAG tools | Not implemented |
| **Model providers** | HfEngine, TransformersModel | Only OpenAI-compatible API |

---

## Known Issues

### 1. Token Usage Not Tracked
**Severity**: Low
**Description**: The `tokenUsage` field in `RunResult` always shows 0. OpenRouter/OpenAI response headers with usage data are not being captured.
**Workaround**: Monitor usage via provider dashboard.

### 2. Husky Warning on npm Publish
**Severity**: Low
**Description**: Warning ".git can't be found" appears during `npm publish` due to husky prepare script.
**Workaround**: Can be ignored; publish still succeeds.

### 3. vm Module Security Limitations
**Severity**: Medium
**Description**: Node's vm module is not a true security sandbox. Determined attackers could potentially escape. The `vm2` package (now deprecated) or isolated-vm could provide stronger isolation.
**Mitigation**: Only run trusted code; use execution delay for human review.

### 4. Large Package Imports May Fail
**Severity**: Medium
**Description**: Complex npm packages with many dependencies may fail to import via CDN due to circular dependencies or bundling issues.
**Workaround**: Use simpler packages or implement tools instead.

### 5. No Abort Signal for Nested Agents
**Severity**: Low
**Description**: When a manager agent calls a worker agent, there's no way to abort the worker mid-execution.
**Future**: Implement abort controller propagation.

---

## Recommended Next Steps

### High Priority

1. **ToolCallingAgent**
   - Implement agent that uses OpenAI function calling instead of code generation
   - Useful for simpler tasks and when code generation is overkill
   - Would match Python smolagents' dual-agent approach

2. **Token Usage Tracking**
   - Parse OpenAI response headers for usage data
   - Aggregate across nested agent calls
   - Enable cost monitoring

3. **Built-in Tool Collection**
   - `WebSearchTool` - DuckDuckGo/Google search
   - `WebFetchTool` - Fetch and parse web pages
   - `FileReadTool` / `FileWriteTool` - File operations
   - `ShellTool` - Execute shell commands (with safety)

### Medium Priority

4. **Remote Executors**
   - E2B integration for true sandboxing
   - Docker executor for isolated environments
   - Would match Python smolagents capabilities

5. **Vision Support**
   - Accept image inputs in agent.run()
   - Pass to vision-capable models
   - Enable image analysis tasks

6. **Improved Sandbox**
   - Evaluate `isolated-vm` for better isolation
   - Memory limits and CPU quotas
   - Network restrictions

### Lower Priority

7. **Gradio/Web UI**
   - Simple web interface for testing agents
   - Message history display
   - Tool configuration

8. **Planning Agent**
   - High-level planning before execution
   - Task decomposition
   - Progress tracking

9. **Persistence**
   - Save/restore agent state
   - Resume interrupted sessions
   - Export conversation history

---

## Project Structure

```
smol-js/
├── src/
│   ├── index.ts              # Public exports
│   ├── types.ts              # TypeScript types
│   ├── agents/
│   │   ├── Agent.ts          # Abstract base class
│   │   └── CodeAgent.ts      # Main implementation
│   ├── models/
│   │   ├── Model.ts          # Abstract base class
│   │   └── OpenAIModel.ts    # OpenAI-compatible client
│   ├── tools/
│   │   ├── Tool.ts           # Tool base class
│   │   ├── AgentTool.ts      # Agent-as-tool wrapper
│   │   └── defaultTools.ts   # FinalAnswerTool, etc.
│   ├── executor/
│   │   └── LocalExecutor.ts  # vm-based code execution
│   ├── memory/
│   │   └── AgentMemory.ts    # Conversation history
│   ├── logging/
│   │   └── AgentLogger.ts    # Color-coded logging
│   └── prompts/
│       └── codeAgent.ts      # System prompt generation
├── tests/                    # Vitest unit tests
├── examples/                 # 9 runnable examples
├── dist/                     # Built output
├── package.json
├── tsconfig.json
├── tsup.config.ts
└── README.md
```

---

## Dependencies

### Runtime
- `openai` - OpenAI SDK for API calls
- `chalk` - Terminal colors
- `dotenv` - Environment variable loading

### Development
- `typescript` - Type checking
- `tsup` - Build tool
- `vitest` - Test framework
- `eslint` - Linting
- `husky` - Git hooks

---

## Conclusion

smol-js v0.2.0 provides a solid TypeScript implementation of the core smolagents functionality. The CodeAgent can solve complex tasks through iterative code generation and execution, with support for custom tools, dynamic imports, and nested agent architectures.

The main gaps compared to Python smolagents are:
- No ToolCallingAgent (function calling mode)
- No remote/Docker executors
- No built-in tool collection

These would be valuable additions for future versions.

---

*Generated: January 21, 2026*
