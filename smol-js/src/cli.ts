/**
 * smol-js CLI - Run YAML-defined agent workflows from the command line
 *
 * Usage:
 *   smol-js <workflow.yaml> [--task "your task"]   (auto-detects yaml files)
 *   smol-js run <workflow.yaml> [--task "your task"]
 *   smol-js validate <workflow.yaml>
 */

import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';
import chalk from 'chalk';
import dotenv from 'dotenv';
import { Orchestrator } from './orchestrator/Orchestrator.js';

// Load environment variables
dotenv.config();

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0 || args[0] === '--help' || args[0] === '-h') {
    printUsage();
    process.exit(0);
  }

  const command = args[0];

  if (command === 'run') {
    await runCommand(args.slice(1));
  } else if (command === 'validate') {
    await validateCommand(args.slice(1));
  } else if (command.endsWith('.yaml') || command.endsWith('.yml')) {
    // Auto-detect: if first arg is a YAML file, treat as "run"
    await runCommand(args);
  } else {
    console.error(chalk.red(`Unknown command: ${command}`));
    printUsage();
    process.exit(1);
  }
}

function printUsage(): void {
  console.log(chalk.cyan.bold('\nsmol-js CLI - YAML Agent Orchestrator\n'));
  console.log('Usage:');
  console.log('  smol-js <workflow.yaml> [options]         Run a workflow (auto-detect)');
  console.log('  smol-js run <workflow.yaml> [options]     Run a workflow (explicit)');
  console.log('  smol-js validate <workflow.yaml>          Validate a workflow file');
  console.log('');
  console.log('Options:');
  console.log('  --task, -t <task>    Task description (prompted if not provided)');
  console.log('  --quiet, -q          Reduce output verbosity');
  console.log('  --help, -h           Show this help message');
  console.log('');
  console.log('Examples:');
  console.log('  npx @samrahimi/smol-js workflow.yaml --task "Research AI safety"');
  console.log('  smol-js research-agent.yaml -t "Write a summary of quantum computing"');
  console.log('  smol-js validate my-workflow.yaml');
}

async function runCommand(args: string[]): Promise<void> {
  if (args.length === 0) {
    console.error(chalk.red('Error: workflow file path required'));
    process.exit(1);
  }

  const filePath = args[0];
  let task = '';
  let quiet = false;

  // Parse remaining args
  for (let i = 1; i < args.length; i++) {
    if (args[i] === '--task' || args[i] === '-t') {
      task = args[i + 1] ?? '';
      i++;
    } else if (args[i] === '--quiet' || args[i] === '-q') {
      quiet = true;
    }
  }

  // Resolve file path
  const resolvedPath = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);

  if (!fs.existsSync(resolvedPath)) {
    console.error(chalk.red(`Error: file not found: ${resolvedPath}`));
    process.exit(1);
  }

  // Get task from user if not provided
  if (!task) {
    task = await promptUser('Enter your task: ');
    if (!task.trim()) {
      console.error(chalk.red('Error: task cannot be empty'));
      process.exit(1);
    }
  }

  // Create orchestrator and run
  const orchestrator = new Orchestrator({ verbose: !quiet });

  try {
    console.log(chalk.gray(`\nLoading workflow from: ${resolvedPath}\n`));
    const workflow = orchestrator.loadWorkflow(resolvedPath);

    await orchestrator.runWorkflow(workflow, task);

    // Exit with success
    process.exit(0);
  } catch (error) {
    console.error(chalk.red(`\nError: ${(error as Error).message}`));
    if (process.env.DEBUG) {
      console.error((error as Error).stack);
    }
    process.exit(1);
  }
}

async function validateCommand(args: string[]): Promise<void> {
  if (args.length === 0) {
    console.error(chalk.red('Error: workflow file path required'));
    process.exit(1);
  }

  const filePath = args[0];
  const resolvedPath = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);

  if (!fs.existsSync(resolvedPath)) {
    console.error(chalk.red(`Error: file not found: ${resolvedPath}`));
    process.exit(1);
  }

  const orchestrator = new Orchestrator({ verbose: false });

  try {
    const workflow = orchestrator.loadWorkflow(resolvedPath);
    console.log(chalk.green.bold('✅ Workflow is valid'));
    console.log(chalk.green(`  Name: ${workflow.name}`));
    console.log(chalk.green(`  Agents: ${Array.from(workflow.agents.keys()).join(', ')}`));
    console.log(chalk.green(`  Tools: ${Array.from(workflow.tools.keys()).join(', ') || '(using defaults)'}`));
    console.log(chalk.green(`  Entrypoint: ${workflow.entrypointAgent.getName()}`));
    process.exit(0);
  } catch (error) {
    console.error(chalk.red(`❌ Validation failed: ${(error as Error).message}`));
    process.exit(1);
  }
}

function promptUser(question: string): Promise<string> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  return new Promise((resolve) => {
    rl.question(chalk.cyan(question), (answer) => {
      rl.close();
      resolve(answer);
    });
  });
}

main().catch((error) => {
  console.error(chalk.red(`Fatal error: ${error.message}`));
  process.exit(1);
});
