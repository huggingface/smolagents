/**
 * Example 5: Error Recovery
 *
 * Demonstrates how the agent handles errors in generated code
 * and recovers by fixing the issues in subsequent steps.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, LogLevel } from '../src/index.js';

async function main() {
  console.log('=== Example 5: Error Recovery ===\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
  });

  // Create the agent
  const agent = new CodeAgent({
    model,
    maxSteps: 8,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
  });

  // Run a task that might cause errors initially
  // The agent should recover and find the correct approach
  const result = await agent.run(
    `Complete this task, but note that some approaches might fail:

     1. First, try to use a function called 'nonExistentFunction()' to process data
        (this will fail, which is expected)
     2. After the error, realize you need to implement the logic yourself
     3. Create an array of the first 10 Fibonacci numbers
     4. Return the array using final_answer

     This demonstrates error recovery - the agent should handle the initial
     failure gracefully and find a working solution.`
  );

  console.log('\n=== Result ===');
  console.log('Output:', result.output);

  // Show how many steps had errors
  const actionSteps = result.steps.filter((s) => s.type === 'action');
  const errorSteps = actionSteps.filter((s) => 'error' in s && s.error);
  console.log('Total steps:', actionSteps.length);
  console.log('Steps with errors:', errorSteps.length);
  console.log('Duration:', (result.duration / 1000).toFixed(2), 'seconds');

  // The agent should have recovered despite the initial error
  if (Array.isArray(result.output) && result.output.length === 10) {
    console.log('\n✅ Agent successfully recovered from errors!');
  }
}

main().catch(console.error);
