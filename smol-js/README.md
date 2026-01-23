# smol-js

**A TypeScript port of the [smolagents](https://github.com/huggingface/smolagents) agentic framework.**

Build AI agents that solve tasks by writing and executing JavaScript code. The agent reasons about problems, generates code, executes it in a sandbox, observes results, and iterates until it finds the answer.

[![npm version](https://img.shields.io/npm/v/@samrahimi/smol-js.svg)](https://www.npmjs.com/package/@samrahimi/smol-js)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **ReAct Framework**: Reasoning + Acting loop (Thought → Code → Observation → repeat)
- **Sandboxed Execution**: JavaScript runs in Node's vm module with state persistence
- **Tool System**: Extensible tools that agents can call as functions
- **Nested Agents**: Use agents as tools for hierarchical task delegation
- **Dynamic Imports**: Import npm packages on-the-fly via jsdelivr CDN
- **Built-in fetch()**: Agents can make HTTP requests directly in generated code
- **OpenAI-Compatible**: Works with OpenRouter, OpenAI, Azure, Anthropic, and local servers
- **Streaming**: Real-time output streaming from the LLM
- **Color-Coded Logging**: Beautiful terminal output with session logging to disk
- **Error Recovery**: Agent can recover from errors and try different approaches

## Installation

```bash
npm install @samrahimi/smol-js
```

## Quick Start

```typescript
import 'dotenv/config';
import { CodeAgent, OpenAIModel } from '@samrahimi/smol-js';

// Create the model (defaults to Claude via OpenRouter)
const model = new OpenAIModel({
  modelId: 'anthropic/claude-sonnet-4.5',
});

// Create the agent
const agent = new CodeAgent({
  model,
  maxSteps: 10,
});

// Run a task
const result = await agent.run('Calculate the first 10 prime numbers');
console.log(result.output); // [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
```

## Configuration

### Environment Variables

```bash
# API key for LLM provider (OpenRouter by default)
OPENAI_API_KEY=sk-or-v1-your-openrouter-key

# Or for OpenAI directly
OPENAI_API_KEY=sk-your-openai-key
```

### Model Configuration

```typescript
const model = new OpenAIModel({
  modelId: 'anthropic/claude-sonnet-4.5', // Model identifier
  apiKey: 'sk-...',                        // API key (or use env var)
  baseUrl: 'https://openrouter.ai/api/v1', // API endpoint (default: OpenRouter)
  maxTokens: 4096,                          // Max tokens to generate
  temperature: 0.7,                         // Generation temperature
  timeout: 120000,                          // Request timeout in ms
});
```

### Agent Configuration

```typescript
const agent = new CodeAgent({
  model,
  tools: [myTool],                          // Custom tools
  maxSteps: 20,                             // Max iterations (default: 20)
  codeExecutionDelay: 5000,                 // Safety delay before execution (default: 5000ms)
  customInstructions: '...',                // Additional system prompt instructions
  verboseLevel: LogLevel.INFO,              // Logging level (OFF, ERROR, INFO, DEBUG)
  streamOutputs: true,                      // Stream LLM output in real-time
  additionalAuthorizedImports: ['lodash'],  // npm packages the agent can import
  workingDirectory: '/path/to/dir',         // Working dir for fs operations
});
```

## Creating Tools

Tools extend the agent's capabilities. The agent sees tools as async functions it can call.

### Class-Based Tools

```typescript
import { Tool } from '@samrahimi/smol-js';
import type { ToolInputs } from '@samrahimi/smol-js';

class WeatherTool extends Tool {
  readonly name = 'get_weather';
  readonly description = 'Get current weather for a city';
  readonly inputs: ToolInputs = {
    city: {
      type: 'string',
      description: 'The city name',
      required: true,
    },
  };
  readonly outputType = 'object';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    const city = args.city as string;
    const response = await fetch(`https://api.weather.com/${city}`);
    return response.json();
  }
}

const agent = new CodeAgent({
  model,
  tools: [new WeatherTool()],
});
```

### Functional Tools

```typescript
import { createTool } from '@samrahimi/smol-js';

const calculator = createTool({
  name: 'calculate',
  description: 'Evaluate a math expression',
  inputs: {
    expression: { type: 'string', description: 'Math expression to evaluate', required: true },
  },
  outputType: 'number',
  execute: async (args) => {
    return new Function('Math', `return ${args.expression}`)(Math);
  },
});
```

## Nested Agents (Agent as Tool)

Use agents as tools for hierarchical task delegation. A "manager" agent can delegate specialized tasks to "worker" agents.

```typescript
import { CodeAgent, OpenAIModel, AgentTool, agentAsTool } from '@samrahimi/smol-js';

// Create a specialized worker agent
const mathAgent = new CodeAgent({
  model,
  tools: [calculatorTool],
  maxSteps: 5,
  verboseLevel: LogLevel.OFF, // Quiet - manager reports results
});

// Wrap it as a tool
const mathExpert = new AgentTool({
  agent: mathAgent,
  name: 'math_expert',
  description: 'Delegate math problems to a specialized math agent',
});

// Or use the helper function
const mathExpert = agentAsTool(mathAgent, {
  name: 'math_expert',
  description: 'Delegate math problems to a specialized math agent',
});

// Create manager that uses the worker
const manager = new CodeAgent({
  model,
  tools: [mathExpert, researchExpert], // Agents as tools!
  maxSteps: 10,
});

await manager.run('Research Tokyo population and calculate water consumption');
```

## Using fetch() Directly

Agents can make HTTP requests directly in their code without needing a tool:

```typescript
const agent = new CodeAgent({
  model,
  tools: [], // No tools needed!
  customInstructions: `You can use fetch() directly to make HTTP requests.
Example: const data = await fetch('https://api.example.com').then(r => r.json());`,
});

await agent.run('Fetch users from https://jsonplaceholder.typicode.com/users');
```

## Dynamic npm Imports

The agent can import npm packages dynamically:

```typescript
const agent = new CodeAgent({
  model,
  additionalAuthorizedImports: ['lodash', 'dayjs', 'uuid'],
});

// The agent can now write:
// const _ = await importPackage('lodash');
// const dayjs = await importPackage('dayjs');
```

Packages are fetched from [jsdelivr CDN](https://www.jsdelivr.com/) and cached locally in `~/.smol-js/packages/`.

## Built-in Capabilities

The agent's sandbox includes:

| Category | Available |
|----------|-----------|
| **Output** | `console.log()`, `console.error()`, `print()` |
| **HTTP** | `fetch()`, `URL`, `URLSearchParams` |
| **File System** | `fs.readFileSync()`, `fs.writeFileSync()`, `fs.existsSync()`, etc. |
| **Path** | `path.join()`, `path.resolve()`, `path.dirname()`, etc. |
| **Data** | `JSON`, `Buffer`, `TextEncoder`, `TextDecoder` |
| **Math** | `Math.*`, `parseInt()`, `parseFloat()` |
| **Types** | `Object`, `Array`, `Map`, `Set`, `Date`, `RegExp`, `Promise` |
| **Timers** | `setTimeout()`, `setInterval()` |
| **Final** | `final_answer(value)` - Return the result |

## Examples

The `examples/` folder contains complete, runnable examples:

| Example | Description |
|---------|-------------|
| **01-simple-math.ts** | Basic calculation task |
| **02-dynamic-imports.ts** | Using npm packages dynamically |
| **03-variable-persistence.ts** | Multi-step state management |
| **04-research-with-tools.ts** | Custom tools for research tasks |
| **05-error-recovery.ts** | Handling and recovering from errors |
| **06-deep-research.ts** | Real API calls with DuckDuckGo/Wikipedia |
| **07-npm-package-import.ts** | Importing from the published npm package |
| **08-fetch-agent.ts** | Agent using fetch() directly (no tools) |
| **09-nested-agents.ts** | Manager agent delegating to worker agents |

Run an example:

```bash
npx tsx examples/08-fetch-agent.ts
```

## API Reference

### CodeAgent

```typescript
class CodeAgent {
  constructor(config: CodeAgentConfig)

  // Run a task
  run(task: string, reset?: boolean): Promise<RunResult>

  // Control
  stop(): void
  reset(): void

  // Tools
  addTool(tool: Tool): void
  removeTool(name: string): boolean
  getTools(): Map<string, Tool>

  // State
  getMemory(): AgentMemory
  getExecutor(): LocalExecutor
}
```

### RunResult

```typescript
interface RunResult {
  output: unknown;        // Final answer
  steps: MemoryStep[];    // Execution history
  tokenUsage: TokenUsage; // Token counts
  duration: number;       // Total time in ms
}
```

### Tool

```typescript
abstract class Tool {
  abstract readonly name: string;
  abstract readonly description: string;
  abstract readonly inputs: ToolInputs;
  abstract readonly outputType: string;

  abstract execute(args: Record<string, unknown>): Promise<unknown>;

  setup(): Promise<void>;          // Optional async initialization
  call(args: Record<string, unknown>): Promise<unknown>;
  toCodePrompt(): string;          // Generate function signature for prompt
}
```

### AgentTool

```typescript
class AgentTool extends Tool {
  constructor(config: AgentToolConfig)
}

interface AgentToolConfig {
  agent: Agent;              // The agent to wrap
  name?: string;             // Tool name (default: 'managed_agent')
  description?: string;      // Tool description
  additionalContext?: string; // Extra context for the agent
  returnFullResult?: boolean; // Return full result vs just output
}

// Helper function
function agentAsTool(agent: Agent, options?: Omit<AgentToolConfig, 'agent'>): AgentTool
```

### LocalExecutor

```typescript
class LocalExecutor {
  constructor(config?: ExecutorConfig)

  execute(code: string): Promise<CodeExecutionOutput>
  sendTools(tools: Record<string, Tool>): void
  sendVariables(variables: Record<string, unknown>): void
  reset(): void
  getState(): Record<string, unknown>
}

interface ExecutorConfig {
  timeout?: number;              // Execution timeout (default: 30000ms)
  authorizedImports?: string[];  // Allowed npm packages
  allowFs?: boolean;             // Enable fs access (default: true)
  workingDirectory?: string;     // Working dir for fs operations
}
```

### LogLevel

```typescript
enum LogLevel {
  OFF = 0,    // No output
  ERROR = 1,  // Errors only
  INFO = 2,   // Normal output (default)
  DEBUG = 3,  // Detailed debugging
}
```

## Session Logging

All sessions are logged to `~/.smol-js/`:
- `session-<timestamp>.log` - Full session transcript
- `packages/` - Cached npm packages

## Comparison with Python smolagents

| Feature | Python smolagents | smol-js |
|---------|------------------|---------|
| Code execution | Python interpreter | Node.js vm module |
| Imports | `import` statement | `await importPackage()` |
| Tool definition | `@tool` decorator | Class extending `Tool` |
| Nested agents | `ManagedAgent` | `AgentTool` |
| Async support | Optional | All tools are async |
| HTTP requests | Requires tool | Built-in `fetch()` |
| Remote executors | E2B, Docker, etc. | Local only (for now) |
| Agent types | CodeAgent, ToolCallingAgent | CodeAgent only |
| Multi-agent | Yes | Yes (via AgentTool) |

## Security Considerations

- **Sandboxed Execution**: Code runs in Node's vm module, isolated from the main process
- **Authorized Imports**: Only explicitly allowed npm packages can be imported
- **File System Isolation**: fs operations are restricted to the configured working directory
- **Execution Delay**: Configurable delay before code execution allows user interruption (Ctrl+C)
- **Timeout Protection**: Code execution has a configurable timeout (default: 30s)

## Contributing

Contributions are welcome! Please open an issue or PR on GitHub.

```bash
# Clone and install
git clone https://github.com/samrahimi/smol-js
cd smol-js
npm install

# Run tests
npm test

# Run examples
npx tsx examples/01-simple-math.ts
```

## License

MIT

## Credits

This is a TypeScript port of [smolagents](https://github.com/huggingface/smolagents) by Hugging Face.
