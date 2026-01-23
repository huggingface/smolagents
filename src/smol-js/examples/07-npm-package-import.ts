/**
 * Example 7: Using @samrahimi/smol-js from npm
 *
 * This example demonstrates importing and using the published npm package
 * instead of the local source code. It shows a simple calculator agent
 * that can perform math operations.
 */

import 'dotenv/config';

// Import from the local source (for npm package usage, see /examples/js/)
import { CodeAgent, OpenAIModel, Tool, LogLevel } from '../src/index.js';
import type { ToolInputs } from '../src/types.js';

/**
 * Calculator Tool - Performs basic math operations
 */
class CalculatorTool extends Tool {
  readonly name = 'calculator';
  readonly description = `Perform mathematical calculations.
Supports: add, subtract, multiply, divide, power, sqrt, sin, cos, tan, log, abs, round, floor, ceil.`;
  readonly inputs: ToolInputs = {
    operation: {
      type: 'string',
      description: 'The operation to perform (add, subtract, multiply, divide, power, sqrt, etc.)',
      required: true,
    },
    a: {
      type: 'number',
      description: 'First operand',
      required: true,
    },
    b: {
      type: 'number',
      description: 'Second operand (not needed for sqrt, sin, cos, tan, log, abs)',
      required: false,
    },
  };
  readonly outputType = 'number';

  async execute(args: Record<string, unknown>): Promise<number> {
    const op = (args.operation as string).toLowerCase();
    const a = args.a as number;
    const b = args.b as number | undefined;

    switch (op) {
      case 'add':
        return a + (b ?? 0);
      case 'subtract':
        return a - (b ?? 0);
      case 'multiply':
        return a * (b ?? 1);
      case 'divide':
        if (b === 0) throw new Error('Division by zero');
        return a / (b ?? 1);
      case 'power':
        return Math.pow(a, b ?? 2);
      case 'sqrt':
        return Math.sqrt(a);
      case 'sin':
        return Math.sin(a);
      case 'cos':
        return Math.cos(a);
      case 'tan':
        return Math.tan(a);
      case 'log':
        return Math.log(a);
      case 'abs':
        return Math.abs(a);
      case 'round':
        return Math.round(a);
      case 'floor':
        return Math.floor(a);
      case 'ceil':
        return Math.ceil(a);
      default:
        throw new Error(`Unknown operation: ${op}`);
    }
  }
}

/**
 * Unit Converter Tool - Converts between common units
 */
class UnitConverterTool extends Tool {
  readonly name = 'convert_units';
  readonly description = `Convert values between different units.
Supports: temperature (celsius/fahrenheit/kelvin), length (meters/feet/inches/miles/km), weight (kg/lbs/oz).`;
  readonly inputs: ToolInputs = {
    value: {
      type: 'number',
      description: 'The value to convert',
      required: true,
    },
    from: {
      type: 'string',
      description: 'Source unit (e.g., celsius, meters, kg)',
      required: true,
    },
    to: {
      type: 'string',
      description: 'Target unit (e.g., fahrenheit, feet, lbs)',
      required: true,
    },
  };
  readonly outputType = 'number';

  async execute(args: Record<string, unknown>): Promise<number> {
    const value = args.value as number;
    const from = (args.from as string).toLowerCase();
    const to = (args.to as string).toLowerCase();

    // Temperature conversions
    if (from === 'celsius' && to === 'fahrenheit') return (value * 9) / 5 + 32;
    if (from === 'fahrenheit' && to === 'celsius') return ((value - 32) * 5) / 9;
    if (from === 'celsius' && to === 'kelvin') return value + 273.15;
    if (from === 'kelvin' && to === 'celsius') return value - 273.15;

    // Length conversions
    if (from === 'meters' && to === 'feet') return value * 3.28084;
    if (from === 'feet' && to === 'meters') return value / 3.28084;
    if (from === 'meters' && to === 'inches') return value * 39.3701;
    if (from === 'inches' && to === 'meters') return value / 39.3701;
    if (from === 'miles' && to === 'km') return value * 1.60934;
    if (from === 'km' && to === 'miles') return value / 1.60934;

    // Weight conversions
    if (from === 'kg' && to === 'lbs') return value * 2.20462;
    if (from === 'lbs' && to === 'kg') return value / 2.20462;
    if (from === 'kg' && to === 'oz') return value * 35.274;
    if (from === 'oz' && to === 'kg') return value / 35.274;

    throw new Error(`Unknown conversion: ${from} to ${to}`);
  }
}

async function main() {
  console.log('=== Example 7: Using @samrahimi/smol-js from npm ===\n');
  console.log('This example imports from the published npm package.\n');

  // Create the model (uses OpenRouter by default)
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    maxTokens: 2048,
  });

  // Create tools
  const calculator = new CalculatorTool();
  const converter = new UnitConverterTool();

  // Create the agent
  const agent = new CodeAgent({
    model,
    tools: [calculator, converter],
    maxSteps: 10,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
  });

  // Test task - a multi-step calculation with unit conversion
  const task = `I need to solve this problem:
1. Calculate the area of a circle with radius 5 meters
2. Convert that area to square feet (hint: convert meters to feet first, then square it, or convert the final area)
3. Round the result to 2 decimal places

Show your work step by step.`;

  console.log(`Task: ${task}\n`);

  const result = await agent.run(task);

  console.log('\n' + '='.repeat(60));
  console.log('RESULT');
  console.log('='.repeat(60));
  console.log('\nFinal Answer:');
  console.log(typeof result.output === 'string' ? result.output : JSON.stringify(result.output, null, 2));
  console.log(`\nSteps: ${result.steps.filter((s) => s.type === 'action').length}`);
  console.log(`Duration: ${(result.duration / 1000).toFixed(2)}s`);
}

main().catch(console.error);
