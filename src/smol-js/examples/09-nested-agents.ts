/**
 * Example 9: Nested Agents - Manager delegating to specialized workers
 *
 * This example demonstrates the hierarchical agent pattern where a "manager"
 * agent can delegate tasks to specialized "worker" agents using AgentTool.
 *
 * Architecture:
 * - Manager Agent: Coordinates and delegates tasks
 *   - Research Agent: Fetches and analyzes web data
 *   - Math Agent: Performs calculations
 *
 * The manager decides which worker to use based on the task requirements.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, Tool, AgentTool, LogLevel } from '../src/index.js';
import type { ToolInputs } from '../src/types.js';

// ============================================================================
// TOOLS FOR WORKER AGENTS
// ============================================================================

/**
 * Calculator Tool for the Math Agent
 */
class CalculatorTool extends Tool {
  readonly name = 'calculate';
  readonly description = 'Perform a mathematical calculation. Supports basic operations and functions.';
  readonly inputs: ToolInputs = {
    expression: {
      type: 'string',
      description: 'Math expression to evaluate (e.g., "2 + 2", "Math.sqrt(16)", "Math.pow(2, 10)")',
      required: true,
    },
  };
  readonly outputType = 'number';

  async execute(args: Record<string, unknown>): Promise<number> {
    const expr = args.expression as string;
    // Safe evaluation using Function constructor with Math context
    const safeEval = new Function('Math', `return ${expr}`);
    return safeEval(Math);
  }
}

/**
 * Unit Converter Tool for the Math Agent
 */
class UnitConverterTool extends Tool {
  readonly name = 'convert';
  readonly description = 'Convert between units (temperature, length, weight)';
  readonly inputs: ToolInputs = {
    value: { type: 'number', description: 'Value to convert', required: true },
    from: { type: 'string', description: 'Source unit', required: true },
    to: { type: 'string', description: 'Target unit', required: true },
  };
  readonly outputType = 'number';

  async execute(args: Record<string, unknown>): Promise<number> {
    const { value, from, to } = args as { value: number; from: string; to: string };
    const f = from.toLowerCase();
    const t = to.toLowerCase();

    // Temperature
    if (f === 'celsius' && t === 'fahrenheit') return (value * 9) / 5 + 32;
    if (f === 'fahrenheit' && t === 'celsius') return ((value - 32) * 5) / 9;
    // Length
    if (f === 'meters' && t === 'feet') return value * 3.28084;
    if (f === 'feet' && t === 'meters') return value / 3.28084;
    if (f === 'km' && t === 'miles') return value * 0.621371;
    if (f === 'miles' && t === 'km') return value / 0.621371;
    // Weight
    if (f === 'kg' && t === 'lbs') return value * 2.20462;
    if (f === 'lbs' && t === 'kg') return value / 2.20462;

    throw new Error(`Unknown conversion: ${from} to ${to}`);
  }
}

// ============================================================================
// WORKER AGENTS
// ============================================================================

function createMathAgent(model: OpenAIModel): CodeAgent {
  return new CodeAgent({
    model,
    tools: [new CalculatorTool(), new UnitConverterTool()],
    maxSteps: 5,
    codeExecutionDelay: 500,
    verboseLevel: LogLevel.OFF, // Quiet - manager will report results
    customInstructions: `You are a specialized math agent. You excel at:
- Mathematical calculations (arithmetic, algebra, trigonometry)
- Unit conversions (temperature, length, weight)
- Statistical computations

Use the calculate() and convert() tools to solve math problems.
Always show your work and explain each step briefly.
Return the final numerical answer with units if applicable.`,
  });
}

function createResearchAgent(model: OpenAIModel): CodeAgent {
  return new CodeAgent({
    model,
    tools: [], // Uses fetch directly
    maxSteps: 5,
    codeExecutionDelay: 500,
    verboseLevel: LogLevel.OFF, // Quiet - manager will report results
    customInstructions: `You are a specialized research agent. You can:
- Fetch data from APIs using fetch()
- Parse and analyze JSON responses
- Extract specific information from web data

You have access to fetch() directly. Example:
\`\`\`javascript
const response = await fetch('https://api.example.com/data');
const data = await response.json();
\`\`\`

Focus on finding accurate information and citing your sources.`,
  });
}

// ============================================================================
// MAIN - MANAGER AGENT WITH NESTED WORKERS
// ============================================================================

async function main() {
  console.log('=== Example 9: Nested Agents ===\n');
  console.log('Manager agent coordinating specialized worker agents.\n');

  // Create the model (shared by all agents)
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    maxTokens: 2048,
  });

  // Create worker agents
  const mathAgent = createMathAgent(model);
  const researchAgent = createResearchAgent(model);

  // Wrap workers as tools for the manager
  const mathTool = new AgentTool({
    agent: mathAgent,
    name: 'math_expert',
    description: `Delegate mathematical tasks to a specialized math agent.
This agent can perform complex calculations, unit conversions, and statistical analysis.
Use for: calculations, unit conversions, math problems, numerical analysis.
Pass a clear description of the math task.`,
  });

  const researchTool = new AgentTool({
    agent: researchAgent,
    name: 'research_expert',
    description: `Delegate research tasks to a specialized research agent.
This agent can fetch data from public APIs and analyze information.
Use for: looking up data, fetching API information, web research.
Pass a clear description of what information to find.`,
  });

  // Create the manager agent with workers as tools
  const manager = new CodeAgent({
    model,
    tools: [mathTool, researchTool],
    maxSteps: 8,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
    customInstructions: `You are a manager agent that coordinates specialized worker agents.

You have access to two expert agents:
1. math_expert - For calculations, unit conversions, and numerical analysis
2. research_expert - For fetching data from APIs and web research

Your job is to:
1. Analyze the user's request
2. Break it down into sub-tasks
3. Delegate each sub-task to the appropriate expert
4. Combine and synthesize the results
5. Provide a comprehensive final answer

Always delegate specialized work to the experts rather than trying to do everything yourself.
Combine insights from multiple experts when needed.`,
  });

  // Complex task requiring both agents
  const task = `I need help with a multi-part problem:

1. First, find out the current population of Tokyo using the research agent
   (use https://api.api-ninjas.com/v1/city?name=tokyo - no API key needed for basic info,
   or try https://restcountries.com/v3.1/capital/tokyo for country data)

2. Then, using the math agent:
   - If Tokyo has about 14 million people and each person uses an average of 250 liters of water per day
   - Calculate the total daily water consumption in liters
   - Convert that to gallons (1 liter = 0.264172 gallons)
   - Express the final answer in billions of gallons

Please coordinate both agents to solve this.`;

  console.log(`Task: ${task}\n`);
  console.log('=' .repeat(60) + '\n');

  const result = await manager.run(task);

  console.log('\n' + '='.repeat(60));
  console.log('FINAL RESULT');
  console.log('='.repeat(60));
  console.log('\nManager\'s Answer:');
  console.log(typeof result.output === 'string' ? result.output : JSON.stringify(result.output, null, 2));
  console.log(`\nTotal Steps: ${result.steps.filter((s) => s.type === 'action').length}`);
  console.log(`Total Duration: ${(result.duration / 1000).toFixed(2)}s`);
}

main().catch(console.error);
