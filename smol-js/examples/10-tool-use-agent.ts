/**
 * Example 10: ToolUseAgent - ReACT agent using tool calls
 *
 * Demonstrates the ToolUseAgent which uses native LLM tool calling
 * instead of code generation to perform tasks.
 */

import dotenv from 'dotenv';
dotenv.config();

import { ToolUseAgent } from '../src/agents/ToolUseAgent.js';
import { OpenAIModel } from '../src/models/OpenAIModel.js';
import { ReadFileTool } from '../src/tools/ReadFileTool.js';
import { WriteFileTool } from '../src/tools/WriteFileTool.js';
import { CurlTool } from '../src/tools/CurlTool.js';
import { LogLevel } from '../src/types.js';

async function main() {
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    baseUrl: 'https://openrouter.ai/api/v1',
  });

  const agent = new ToolUseAgent({
    model,
    tools: [
      new ReadFileTool(),
      new WriteFileTool(),
      new CurlTool(),
    ],
    maxSteps: 10,
    verboseLevel: LogLevel.INFO,
    name: 'FileAndWebAgent',
    customInstructions: 'You can read/write files and make HTTP requests.',
  });

  const result = await agent.run(
    'Read the file package.json, extract the name and version, then write a summary to /tmp/smol-js-summary.txt'
  );

  console.log('\nResult:', result.output);
}

main().catch(console.error);
