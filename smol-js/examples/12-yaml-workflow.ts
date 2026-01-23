/**
 * Example 12: YAML Workflow - Load and run a YAML-defined agent workflow
 *
 * Demonstrates loading a complex multi-agent workflow from YAML
 * and running it with the orchestrator.
 */

import dotenv from 'dotenv';
dotenv.config();

import * as path from 'path';
import { Orchestrator } from '../src/orchestrator/Orchestrator.js';

async function main() {
  const orchestrator = new Orchestrator({ verbose: true });

  // Load the research-and-write workflow
  const workflowPath = path.join(import.meta.dirname ?? __dirname, 'workflows', 'research-and-write.yaml');
  const workflow = orchestrator.loadWorkflow(workflowPath);

  // Run with a task
  const result = await orchestrator.runWorkflow(
    workflow,
    'Research the current state of AI agent frameworks in 2024-2025, focusing on open source projects and their architectures'
  );

  console.log('\nWorkflow completed successfully!');
  console.log(`Output length: ${String(result.output).length} characters`);
}

main().catch(console.error);
