/**
 * AgentTool - Wraps a CodeAgent as a Tool for use by other agents
 *
 * This enables nested/hierarchical agent architectures where a "manager" agent
 * can delegate tasks to specialized "worker" agents.
 *
 * Based on the ManagedAgent pattern from Python smolagents.
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';
import type { Agent } from '../agents/Agent.js';

export interface AgentToolConfig {
  /**
   * The agent to wrap as a tool
   */
  agent: Agent;

  /**
   * Name for the tool (defaults to agent's name or "managed_agent")
   */
  name?: string;

  /**
   * Description of what this agent does (used in parent agent's prompt)
   */
  description?: string;

  /**
   * Additional context to provide to the agent with each task
   */
  additionalContext?: string;

  /**
   * Whether to provide the full result object or just the output
   * @default false
   */
  returnFullResult?: boolean;
}

export class AgentTool extends Tool {
  readonly name: string;
  readonly description: string;
  readonly inputs: ToolInputs = {
    task: {
      type: 'string',
      description: 'The task or question to delegate to this agent',
      required: true,
    },
  };
  readonly outputType = 'string';

  private agent: Agent;
  private additionalContext?: string;
  private returnFullResult: boolean;

  constructor(config: AgentToolConfig) {
    super();
    this.agent = config.agent;
    this.name = config.name ?? 'managed_agent';
    this.additionalContext = config.additionalContext;
    this.returnFullResult = config.returnFullResult ?? false;

    // Build description from config or generate default
    this.description = config.description ?? this.generateDescription();
  }

  /**
   * Generate a default description based on the agent's configuration
   */
  private generateDescription(): string {
    return `Delegates a task to a specialized agent.
This agent can help with complex sub-tasks that require multiple steps.
Pass a clear, specific task description and the agent will work autonomously to solve it.
Returns the agent's final answer as a string.`;
  }

  /**
   * Execute the agent with the given task
   */
  async execute(args: Record<string, unknown>): Promise<unknown> {
    let task = args.task as string;

    // Add additional context if provided
    if (this.additionalContext) {
      task = `${this.additionalContext}\n\nTask: ${task}`;
    }

    // Run the agent
    const result = await this.agent.run(task, true);

    // Return full result or just output
    if (this.returnFullResult) {
      return {
        output: result.output,
        steps: result.steps.length,
        duration: result.duration,
      };
    }

    // Convert output to string if needed
    const output = result.output;
    if (typeof output === 'string') {
      return output;
    }
    return JSON.stringify(output, null, 2);
  }

  /**
   * Override toCodePrompt to provide a cleaner signature for nested agents
   */
  toCodePrompt(): string {
    return `
/**
 * ${this.description}
 *
 * @param task - The task or question to delegate to this specialized agent
 * @returns The agent's answer as a string
 */
async function ${this.name}(task: string): Promise<string> { ... }
`.trim();
  }
}

/**
 * Helper function to quickly wrap an agent as a tool
 */
export function agentAsTool(
  agent: Agent,
  options?: Omit<AgentToolConfig, 'agent'>
): AgentTool {
  return new AgentTool({ agent, ...options });
}
