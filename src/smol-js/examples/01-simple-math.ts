/**
 * Example 1: Simple Math Task
 *
 * Demonstrates basic CodeAgent usage with a simple calculation task.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, LogLevel } from '../src/index.js';

async function main() {
  console.log('=== Example 1: Simple Math Task ===\n');

  // Create the model
  const model = new OpenAIModel({
    // Uses OPENAI_API_KEY from environment by default
    // baseUrl defaults to OpenRouter
    modelId: 'anthropic/claude-sonnet-4.5',
  });

  // Create the agent
  const agent = new CodeAgent({
    model,
    maxSteps: 5,
    codeExecutionDelay: 1000, // 1 second delay for demo
    verboseLevel: LogLevel.INFO,
  });

  // Run the task
  const result = await agent.run(
    'Calculate the sum of all prime numbers less than 50. Return the final sum.'
  );

  console.log('\n=== Result ===');
  console.log('Output:', result.output);
  console.log('Steps:', result.steps.length);
  console.log('Total tokens:', result.tokenUsage.totalTokens);
  console.log('Duration:', (result.duration / 1000).toFixed(2), 'seconds');
}

main().catch(console.error);
