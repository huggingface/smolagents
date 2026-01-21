/**
 * Example 3: Variable Persistence Across Steps
 *
 * Demonstrates that variables defined in one step persist to subsequent steps.
 * This is essential for multi-step tasks where data needs to be accumulated.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, LogLevel } from '../src/index.js';

async function main() {
  console.log('=== Example 3: Variable Persistence Across Steps ===\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
  });

  // Create the agent
  const agent = new CodeAgent({
    model,
    maxSteps: 10,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
  });

  // Run a multi-step task that requires building up state
  const result = await agent.run(
    `Complete this multi-step task, using separate code blocks for each step:

     Step 1: Create an array called 'data' with 5 random numbers between 1 and 100
     Step 2: Calculate the mean of 'data' and store it in a variable called 'mean'
     Step 3: Calculate how many numbers in 'data' are above the mean
     Step 4: Return an object with { data, mean, aboveMean } using final_answer

     Important: Do ONE step per code block so we can verify variable persistence.`
  );

  console.log('\n=== Result ===');
  console.log('Output:', JSON.stringify(result.output, null, 2));
  console.log('Steps:', result.steps.length);
  console.log('Duration:', (result.duration / 1000).toFixed(2), 'seconds');
}

main().catch(console.error);
