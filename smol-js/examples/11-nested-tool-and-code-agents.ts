/**
 * Example 11: Mixed Nested Agents - ToolUseAgent managing both ToolUseAgent and CodeAgent
 *
 * Demonstrates a hierarchical agent setup where a ToolUseAgent manager
 * delegates to both a ToolUseAgent (for web research) and a CodeAgent (for analysis).
 * This showcases the mixed agent hierarchy capability.
 */

import dotenv from 'dotenv';
dotenv.config();

import { ToolUseAgent } from '../src/agents/ToolUseAgent.js';
import { CodeAgent } from '../src/agents/CodeAgent.js';
import { OpenAIModel } from '../src/models/OpenAIModel.js';
import { AgentTool } from '../src/tools/AgentTool.js';
import { ExaSearchTool } from '../src/tools/ExaSearchTool.js';
import { ExaGetContentsTool } from '../src/tools/ExaGetContentsTool.js';
import { ExaResearchTool } from '../src/tools/ExaResearchTool.js';
import { ReadFileTool } from '../src/tools/ReadFileTool.js';
import { WriteFileTool } from '../src/tools/WriteFileTool.js';
import { CurlTool } from '../src/tools/CurlTool.js';
import { LogLevel } from '../src/types.js';

async function main() {
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    baseUrl: 'https://openrouter.ai/api/v1',
  });

  // Sub-agent 1: Web Researcher (ToolUseAgent)
  const researcher = new ToolUseAgent({
    model,
    tools: [
      new ExaSearchTool(),
      new ExaGetContentsTool(),
      new ExaResearchTool(),
      new CurlTool(),
    ],
    maxSteps: 8,
    name: 'Researcher',
    customInstructions: 'You are a web researcher. Use exa_research for deep research and exa_search for targeted queries. Always provide well-structured findings with source URLs.',
  });

  // Sub-agent 2: Data Analyst (CodeAgent)
  const analyst = new CodeAgent({
    model,
    tools: [new ReadFileTool(), new WriteFileTool()],
    maxSteps: 6,
    name: 'Analyst',
    codeExecutionDelay: 0,
    customInstructions: 'You are a data analyst. Process the given data using JavaScript code to extract key insights, organize findings, and compute statistics where applicable.',
  });

  // Manager Agent (ToolUseAgent)
  const manager = new ToolUseAgent({
    model,
    tools: [
      new AgentTool({
        agent: researcher,
        name: 'researcher',
        description: 'Delegates research tasks to a web researcher agent. Pass a clear research question and the agent will use Exa.ai to find and synthesize information.',
      }),
      new AgentTool({
        agent: analyst,
        name: 'analyst',
        description: 'Delegates data analysis tasks to a code execution agent. Pass data or research findings and the agent will process them using JavaScript.',
      }),
      new WriteFileTool(),
    ],
    maxSteps: 10,
    verboseLevel: LogLevel.INFO,
    name: 'Manager',
    customInstructions: `You are a project manager. To complete the task:
1. First delegate research to the researcher agent
2. Then pass findings to the analyst for structured analysis
3. Finally compile the results and save to a file using write_file`,
  });

  const result = await manager.run(
    'Research the latest developments in WebAssembly (WASM) for 2024-2025. Have the analyst organize the findings into categories, then save a structured report to /tmp/wasm-report.txt'
  );

  console.log('\nFinal output:', typeof result.output === 'string' ? result.output.slice(0, 500) : result.output);
  console.log(`\nTotal tokens used: ${result.tokenUsage.totalTokens}`);
  console.log(`Duration: ${(result.duration / 1000).toFixed(1)}s`);
}

main().catch(console.error);
