/**
 * Orchestrator - Loads, runs, and provides real-time visibility into agent execution
 */

import chalk from 'chalk';
import type { OrchestratorEvent, RunResult } from '../types.js';
import { Agent } from '../agents/Agent.js';
import { YAMLLoader, LoadedWorkflow } from './YAMLLoader.js';

export interface OrchestratorConfig {
  /** Whether to display real-time output (default: true) */
  verbose?: boolean;
  /** Callback for orchestrator events */
  onEvent?: (event: OrchestratorEvent) => void;
}

export class Orchestrator {
  private loader: YAMLLoader;
  private config: OrchestratorConfig;
  private activeAgents: Map<string, { agent: Agent; depth: number }> = new Map();
  private eventLog: OrchestratorEvent[] = [];

  constructor(config: OrchestratorConfig = {}) {
    this.loader = new YAMLLoader();
    this.config = {
      verbose: config.verbose ?? true,
      onEvent: config.onEvent,
    };
  }

  /**
   * Load a workflow from a YAML file.
   */
  loadWorkflow(filePath: string): LoadedWorkflow {
    const workflow = this.loader.loadFromFile(filePath);
    this.displayWorkflowInfo(workflow);
    return workflow;
  }

  /**
   * Load a workflow from YAML string.
   */
  loadWorkflowFromString(yamlContent: string): LoadedWorkflow {
    const workflow = this.loader.loadFromString(yamlContent);
    this.displayWorkflowInfo(workflow);
    return workflow;
  }

  /**
   * Run a loaded workflow with a task.
   */
  async runWorkflow(workflow: LoadedWorkflow, task: string): Promise<RunResult> {
    this.displayRunStart(workflow.name, task);

    // Set up event tracking on the entrypoint agent
    this.instrumentAgent(workflow.entrypointAgent, workflow.entrypointAgent.getName(), 0);

    // Also instrument sub-agents
    for (const [name, agent] of workflow.agents) {
      if (agent !== workflow.entrypointAgent) {
        this.instrumentAgent(agent, name, 1);
      }
    }

    try {
      const result = await workflow.entrypointAgent.run(task);
      this.displayRunEnd(result);
      return result;
    } catch (error) {
      this.displayError(error as Error);
      throw error;
    }
  }

  /**
   * Run a standalone agent with a task.
   */
  async runAgent(agent: Agent, task: string): Promise<RunResult> {
    this.instrumentAgent(agent, agent.getName(), 0);
    const result = await agent.run(task);
    return result;
  }

  /**
   * Instrument an agent with orchestrator event tracking.
   */
  private instrumentAgent(agent: Agent, name: string, depth: number): void {
    this.activeAgents.set(name, { agent, depth });

    // The Agent base class supports onEvent callback through config
    // We'll add event tracking through the logging output
  }

  /**
   * Display workflow info at startup.
   */
  private displayWorkflowInfo(workflow: LoadedWorkflow): void {
    if (!this.config.verbose) return;

    const line = '═'.repeat(70);
    console.log(chalk.cyan(line));
    console.log(chalk.cyan.bold(`  Workflow: ${workflow.name}`));
    if (workflow.description) {
      console.log(chalk.cyan(`  ${workflow.description}`));
    }
    console.log(chalk.cyan(`  Agents: ${Array.from(workflow.agents.keys()).join(', ')}`));
    console.log(chalk.cyan(`  Tools: ${Array.from(workflow.tools.keys()).join(', ') || '(none defined at workflow level)'}`));
    console.log(chalk.cyan(`  Entrypoint: ${workflow.entrypointAgent.getName()}`));
    console.log(chalk.cyan(line));
    console.log();
  }

  /**
   * Display run start info.
   */
  private displayRunStart(workflowName: string, task: string): void {
    if (!this.config.verbose) return;
    console.log(chalk.green.bold(`\n▶ Running workflow "${workflowName}"`));
    console.log(chalk.green(`  Task: ${task}`));
    console.log(chalk.gray('─'.repeat(70)));
  }

  /**
   * Display run completion info.
   */
  private displayRunEnd(result: RunResult): void {
    if (!this.config.verbose) return;
    console.log(chalk.gray('\n' + '─'.repeat(70)));
    console.log(chalk.green.bold(`\n✅ Workflow complete`));
    console.log(chalk.green(`  Duration: ${(result.duration / 1000).toFixed(2)}s`));
    console.log(chalk.green(`  Tokens: ${result.tokenUsage.totalTokens}`));
    console.log(chalk.green(`  Steps: ${result.steps.length}`));

    const outputStr = typeof result.output === 'string'
      ? result.output
      : JSON.stringify(result.output, null, 2);
    console.log(chalk.magenta.bold('\n  Final Output:'));
    // Indent output
    const indentedOutput = outputStr.split('\n').map(line => `  ${line}`).join('\n');
    console.log(chalk.magenta(indentedOutput));
    console.log();
  }

  /**
   * Display an error.
   */
  private displayError(error: Error): void {
    if (!this.config.verbose) return;
    console.error(chalk.red.bold(`\n❌ Workflow failed: ${error.message}`));
    if (error.stack) {
      console.error(chalk.red.dim(error.stack));
    }
  }

  /**
   * Log an orchestration event.
   */
  logEvent(event: OrchestratorEvent): void {
    this.eventLog.push(event);
    if (this.config.onEvent) {
      this.config.onEvent(event);
    }
  }

  /**
   * Get the event log.
   */
  getEventLog(): OrchestratorEvent[] {
    return [...this.eventLog];
  }

  /**
   * Get the YAML loader for registering custom tools.
   */
  getLoader(): YAMLLoader {
    return this.loader;
  }
}
