/**
 * Custom Tools Demo - @samrahimi/smol-js
 *
 * This demo shows how custom tools defined in TypeScript are linked to
 * YAML workflow definitions. The key mechanism:
 *
 *   1. Define your tool as a class extending Tool (see custom-tools.ts)
 *   2. Register it with YAMLLoader via registerToolType(typeName, ToolClass)
 *   3. Reference it in YAML as: type: <typeName>
 *
 * That's it. The YAMLLoader resolves tool types from two sources:
 *   - Built-in registry: read_file, write_file, curl, exa_search, etc.
 *   - Custom registry: whatever you register with registerToolType()
 *
 * Usage:
 *   npm install
 *   npx tsx main.ts
 */

import 'dotenv/config';
import { YAMLLoader, Orchestrator } from '@samrahimi/smol-js';

// Import custom tool implementations
import { TimestampTool, TextStatsTool, SlugifyTool } from './custom-tools.js';

async function main() {
  // ─────────────────────────────────────────────────────────
  // Step 1: Create a YAMLLoader and register custom tools
  // ─────────────────────────────────────────────────────────
  //
  // This is the bridge between YAML declarations and TypeScript classes.
  // Each registerToolType() call maps a type name (used in YAML) to a
  // Tool subclass (defined in your code).

  const loader = new YAMLLoader();

  // Register custom tools by their type name
  // In workflow.yaml, these are referenced as:
  //   tools:
  //     timestamp:
  //       type: timestamp       <-- maps to TimestampTool
  //     stats:
  //       type: text_stats      <-- maps to TextStatsTool
  //     slug:
  //       type: slugify         <-- maps to SlugifyTool

  loader.registerToolType('timestamp', TimestampTool as any);
  loader.registerToolType('text_stats', TextStatsTool as any);
  loader.registerToolType('slugify', SlugifyTool as any);

  console.log('✅ Registered custom tools: timestamp, text_stats, slugify\n');

  // ─────────────────────────────────────────────────────────
  // Step 2: Load the YAML workflow
  // ─────────────────────────────────────────────────────────
  //
  // The YAML defines agents, tools, and their relationships declaratively.
  // Built-in tools (read_file, write_file, exa_search) are resolved from
  // the internal registry. Custom tools (timestamp, text_stats, slugify)
  // are resolved from what we registered above.

  const workflow = loader.loadFromFile('./workflow.yaml');

  console.log(`📋 Loaded workflow: ${workflow.name}`);
  console.log(`   Agents: ${Array.from(workflow.agents.keys()).join(', ')}`);
  console.log(`   Tools: ${Array.from(workflow.tools.keys()).join(', ')}`);
  console.log(`   Entrypoint: ${workflow.entrypointAgent.getName()}\n`);

  // ─────────────────────────────────────────────────────────
  // Step 3: Run the workflow via the Orchestrator
  // ─────────────────────────────────────────────────────────

  const orchestrator = new Orchestrator({ verbose: true });
  const task = process.argv[2] || 'Write a brief blog post about recent advances in edge computing with WebAssembly';

  console.log(`🚀 Running task: ${task}\n`);

  await orchestrator.runWorkflow(workflow, task);
}

main().catch(err => {
  console.error('Fatal error:', err.message);
  process.exit(1);
});
