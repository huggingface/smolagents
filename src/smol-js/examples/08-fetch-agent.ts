/**
 * Example 8: Fetch Agent - Agent that makes network requests directly
 *
 * This example demonstrates an agent that uses fetch() directly in its
 * generated code to make HTTP requests. No tools needed - the agent
 * writes JavaScript that calls fetch itself.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, LogLevel } from '../src/index.js';

async function main() {
  console.log('=== Example 8: Fetch Agent ===\n');
  console.log('This agent uses fetch() directly in its code to make HTTP requests.\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    maxTokens: 2048,
  });

  // Create the agent with NO tools - it will use fetch directly
  const agent = new CodeAgent({
    model,
    tools: [], // No tools! Agent uses fetch directly
    maxSteps: 10,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
    customInstructions: `You are an agent that can make HTTP requests using the built-in fetch() function.

IMPORTANT: You have access to the standard fetch() API directly in your JavaScript code.
You do NOT need any tools - just write code that uses fetch().

Example of how to use fetch:
\`\`\`javascript
const response = await fetch('https://api.example.com/data');
const data = await response.json();
console.log(data);
\`\`\`

When making requests:
1. Always use await with fetch since it returns a Promise
2. Check response.ok to handle errors
3. Use response.json() for JSON APIs, response.text() for text
4. Log intermediate results with console.log() so you can see what you're getting
5. When done, call final_answer() with your result

Available globals: fetch, JSON, console, Math, Date, URL, URLSearchParams, Buffer, TextEncoder, TextDecoder`,
  });

  // Task: Fetch data from a public API
  const task = `Use fetch to get information about the JSONPlaceholder API:
1. Fetch the list of users from https://jsonplaceholder.typicode.com/users
2. Find all users who work for companies with names ending in "Group" or "LLC"
3. Return their names and company names as a formatted list`;

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
