# smol-js

A TypeScript port of the [smolagents](https://github.com/huggingface/smolagents) agentic framework. This library provides a CodeAgent that can solve tasks by generating and executing JavaScript code in a sandboxed environment.

## Features

- **CodeAgent**: An LLM-powered agent that generates JavaScript code to solve tasks
- **Multi-step execution**: Variables persist between steps for complex workflows
- **Tool system**: Extensible tools that the agent can use as functions
- **Dynamic imports**: Import npm packages on-the-fly via CDN
- **OpenAI-compatible API**: Works with OpenRouter, OpenAI, Azure, and local servers
- **Streaming support**: Real-time output streaming from the LLM
- **Color-coded logging**: Beautiful terminal output with syntax highlighting
- **Error recovery**: Agent can recover from errors and try different approaches

## Installation

```bash
npm install @samrahimi/smol-js
```

Or with yarn:

```bash
yarn add @samrahimi/smol-js
```

## Quick Start

```typescript
import { CodeAgent, OpenAIModel } from '@samrahimi/smol-js';

// Create the model (uses OPENAI_API_KEY env var)
const model = new OpenAIModel({
  modelId: 'anthropic/claude-sonnet-4.5', // default, via OpenRouter
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
# Required: API key for LLM provider
OPENAI_API_KEY=your-api-key

# Or use OpenRouter specifically
OPENROUTER_API_KEY=your-openrouter-key
```

### Model Configuration

```typescript
const model = new OpenAIModel({
  modelId: 'gpt-4',                    // Model identifier
  apiKey: 'sk-...',                    // API key (or use env var)
  baseUrl: 'https://api.openai.com/v1', // API endpoint
  maxTokens: 4096,                      // Max tokens to generate
  temperature: 0.7,                     // Generation temperature
  timeout: 120000,                      // Request timeout in ms
});
```

### Agent Configuration

```typescript
const agent = new CodeAgent({
  model,
  tools: [myTool],              // Custom tools
  maxSteps: 20,                 // Max iterations (default: 20)
  codeExecutionDelay: 5000,     // Delay before execution in ms (default: 5000)
  customInstructions: '...',    // Additional prompt instructions
  verboseLevel: LogLevel.INFO,  // Logging level
  streamOutputs: true,          // Stream LLM output
  additionalAuthorizedImports: ['lodash', 'dayjs'], // Allowed npm packages
  workingDirectory: '/path/to/dir', // Working dir for fs operations
});
```

## Creating Tools

Tools extend the agent's capabilities. Create a tool by extending the `Tool` class:

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
    // Fetch weather data...
    return { city, temperature: 22, condition: 'sunny' };
  }
}

// Use with agent
const agent = new CodeAgent({
  model,
  tools: [new WeatherTool()],
});
```

Or use the `createTool` helper:

```typescript
import { createTool } from '@samrahimi/smol-js';

const addNumbers = createTool({
  name: 'add',
  description: 'Adds two numbers',
  inputs: {
    a: { type: 'number', description: 'First number' },
    b: { type: 'number', description: 'Second number' },
  },
  outputType: 'number',
  execute: async (args) => (args.a as number) + (args.b as number),
});
```

## Dynamic Imports

The agent can import npm packages dynamically using `importPackage()`:

```typescript
const agent = new CodeAgent({
  model,
  additionalAuthorizedImports: ['lodash', 'dayjs', 'uuid'],
});

// The agent can now use:
// const _ = await importPackage('lodash');
// const dayjs = await importPackage('dayjs');
```

Packages are fetched from [esm.sh](https://esm.sh) CDN.

## Built-in Capabilities

The agent has access to:

- `console.log()` / `print()` - Output logging
- `fs` - File system operations (read, write, mkdir, etc.)
- `path` - Path utilities
- `fetch()` - HTTP requests
- `JSON`, `Math`, `Date` - Standard JavaScript globals
- `final_answer(value)` - Return the final result

## Log Levels

```typescript
import { LogLevel } from '@samrahimi/smol-js';

LogLevel.OFF    // No output
LogLevel.ERROR  // Errors only
LogLevel.INFO   // Normal output (default)
LogLevel.DEBUG  // Detailed debugging
```

## Session Logging

All sessions are logged to `~/.smol-js/session-<timestamp>.log`.

## Examples

See the `examples/` folder for complete examples:

1. **01-simple-math.ts** - Basic calculation task
2. **02-dynamic-imports.ts** - Using npm packages dynamically
3. **03-variable-persistence.ts** - Multi-step state management
4. **04-research-with-tools.ts** - Custom tools for research tasks
5. **05-error-recovery.ts** - Handling and recovering from errors

Run all examples:

```bash
npm run run-examples
```

Or run a specific example:

```bash
npx tsx examples/01-simple-math.ts
```

## API Reference

### CodeAgent

Main agent class that generates and executes JavaScript code.

```typescript
class CodeAgent {
  constructor(config: CodeAgentConfig)
  run(task: string, reset?: boolean): Promise<RunResult>
  stop(): void
  reset(): void
  addTool(tool: Tool): void
  removeTool(name: string): boolean
  getTools(): Map<string, Tool>
  getMemory(): AgentMemory
  getExecutor(): LocalExecutor
}
```

### RunResult

```typescript
interface RunResult {
  output: unknown;           // Final answer
  steps: MemoryStep[];       // All execution steps
  tokenUsage: TokenUsage;    // Total token usage
  duration: number;          // Total time in ms
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
  setup(): Promise<void>;
  call(args: Record<string, unknown>): Promise<unknown>;
  toCodePrompt(): string;
}
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
```

## Architectural Differences from Python smolagents

| Feature | Python smolagents | smol-js |
|---------|------------------|---------|
| Code execution | Python interpreter | Node.js vm module |
| Imports | `import` statement | `await importPackage()` |
| Tool definition | `@tool` decorator | Class extending `Tool` |
| Async support | Optional | All tools are async |
| Remote executors | E2B, Docker, etc. | Local only (for now) |
| Agent types | CodeAgent, ToolCallingAgent | CodeAgent only (for now) |

## Security Considerations

- Code executes in a sandboxed vm context
- Only authorized npm packages can be imported
- File system access is restricted to working directory
- Configurable execution delay allows user interruption

## Contributing

Contributions are welcome! Please open an issue or PR.

## License

MIT

## Credits

This is a TypeScript port of [smolagents](https://github.com/huggingface/smolagents) by Hugging Face.
