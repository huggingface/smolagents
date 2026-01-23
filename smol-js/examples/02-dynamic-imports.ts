/**
 * Example 2: Dynamic Package Imports
 *
 * Demonstrates how the agent can dynamically import npm packages from a CDN.
 * This allows the agent to use libraries it wasn't initially configured with.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, LogLevel } from '../src/index.js';

async function main() {
  console.log('=== Example 2: Dynamic Package Imports ===\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
  });

  // Create the agent with authorized imports
  const agent = new CodeAgent({
    model,
    maxSteps: 5,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
    // Authorize specific packages that can be imported dynamically
    additionalAuthorizedImports: ['lodash', 'dayjs', 'uuid'],
  });

  // Run a task that requires using lodash
  const result = await agent.run(
    `Use lodash to:
     1. Create an array of numbers from 1 to 20
     2. Filter out odd numbers
     3. Square each remaining number
     4. Find the sum of the result

     Remember to use await importPackage('lodash') to import it.
     Return the final sum.`
  );

  console.log('\n=== Result ===');
  console.log('Output:', result.output);
  console.log('Steps:', result.steps.length);
  console.log('Duration:', (result.duration / 1000).toFixed(2), 'seconds');
}

main().catch(console.error);
