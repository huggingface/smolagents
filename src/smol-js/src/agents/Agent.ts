/**
 * Agent - Abstract base class for all agents
 *
 * Provides the foundation for multi-step agents that follow the ReAct framework.
 * Extend this class to create specific agent implementations.
 */

import type {
  ActionStep,
  RunResult,
  LogLevel,
  ActionOutput,
} from '../types.js';
import { Tool } from '../tools/Tool.js';
import { Model } from '../models/Model.js';
import { AgentMemory } from '../memory/AgentMemory.js';
import { AgentLogger } from '../logging/AgentLogger.js';
import { LogLevel as LogLevelEnum } from '../types.js';

export interface AgentConfig {
  /**
   * The LLM model to use for generation
   */
  model: Model;

  /**
   * Tools available to the agent
   */
  tools?: Tool[];

  /**
   * Maximum number of steps before stopping
   * @default 20
   */
  maxSteps?: number;

  /**
   * Delay in milliseconds before executing code (for user interruption)
   * @default 5000
   */
  codeExecutionDelay?: number;

  /**
   * Custom system prompt (will be merged with generated prompt)
   */
  customInstructions?: string;

  /**
   * Log level for output
   * @default LogLevel.INFO
   */
  verboseLevel?: LogLevel;

  /**
   * Whether to stream model outputs
   * @default true
   */
  streamOutputs?: boolean;
}

export abstract class Agent {
  /**
   * The LLM model for generation
   */
  protected model: Model;

  /**
   * Available tools mapped by name
   */
  protected tools: Map<string, Tool> = new Map();

  /**
   * Agent memory tracking all steps
   */
  protected memory!: AgentMemory;

  /**
   * Logger for formatted output
   */
  protected logger: AgentLogger;

  /**
   * Configuration options
   */
  protected config: Required<Omit<AgentConfig, 'model' | 'tools'>>;

  /**
   * Current step number
   */
  protected currentStep: number = 0;

  /**
   * Whether the agent is currently running
   */
  protected isRunning: boolean = false;

  constructor(config: AgentConfig) {
    this.model = config.model;
    this.logger = new AgentLogger(config.verboseLevel ?? LogLevelEnum.INFO);

    // Set default config values
    this.config = {
      maxSteps: config.maxSteps ?? 20,
      codeExecutionDelay: config.codeExecutionDelay ?? 5000,
      customInstructions: config.customInstructions ?? '',
      verboseLevel: config.verboseLevel ?? LogLevelEnum.INFO,
      streamOutputs: config.streamOutputs ?? true,
    };

    // Register tools
    if (config.tools) {
      for (const tool of config.tools) {
        this.tools.set(tool.name, tool);
      }
    }
  }

  /**
   * Initialize the system prompt for the agent.
   * Must be implemented by subclasses.
   */
  protected abstract initializeSystemPrompt(): string;

  /**
   * Execute a single step in the agent loop.
   * Must be implemented by subclasses.
   *
   * @param memoryStep - The memory step to populate with execution results
   * @returns The action output from this step
   */
  protected abstract executeStep(memoryStep: ActionStep): Promise<ActionOutput>;

  /**
   * Run the agent on a task.
   *
   * @param task - The task description
   * @param reset - Whether to reset memory before running
   * @returns The final result
   */
  async run(task: string, reset: boolean = true): Promise<RunResult> {
    const startTime = Date.now();

    // Reset if requested or this is a new run
    if (reset || !this.memory) {
      const systemPrompt = this.initializeSystemPrompt();
      this.memory = new AgentMemory(systemPrompt);
      this.currentStep = 0;
    }

    // Add task to memory
    this.memory.addTask(task);

    this.isRunning = true;
    this.logger.header(`🚀 Starting Agent: ${task.slice(0, 50)}${task.length > 50 ? '...' : ''}`);

    let finalOutput: unknown = null;
    let isFinalAnswer = false;

    try {
      // Main agent loop
      while (this.currentStep < this.config.maxSteps && this.isRunning) {
        this.currentStep++;
        this.logger.stepProgress(this.currentStep, this.config.maxSteps);

        // Create memory step
        const memoryStep = this.memory.createActionStep(this.currentStep);

        try {
          // Execute the step
          const actionOutput = await this.executeStep(memoryStep);

          // Update memory step
          memoryStep.timing.endTime = Date.now();
          memoryStep.timing.duration = memoryStep.timing.endTime - memoryStep.timing.startTime;
          memoryStep.actionOutput = actionOutput;
          memoryStep.isFinalAnswer = actionOutput.isFinalAnswer;

          if (actionOutput.isFinalAnswer) {
            finalOutput = actionOutput.output;
            isFinalAnswer = true;
            this.logger.finalAnswer(finalOutput);
            break;
          }
        } catch (error) {
          // Store error in memory step
          memoryStep.error = error as Error;
          memoryStep.timing.endTime = Date.now();
          memoryStep.timing.duration = memoryStep.timing.endTime - memoryStep.timing.startTime;

          this.logger.error('Step execution failed', error as Error);

          // Error will be passed to LLM in next iteration for recovery
        }
      }

      // If we hit max steps without a final answer
      if (!isFinalAnswer && this.currentStep >= this.config.maxSteps) {
        this.logger.warn(`Max steps (${this.config.maxSteps}) reached without final answer`);
        finalOutput = await this.provideFinalAnswer(task);
      }
    } finally {
      this.isRunning = false;
    }

    // Calculate total duration and token usage
    const duration = Date.now() - startTime;
    const tokenUsage = this.memory.getTotalTokenUsage();

    // Add final answer to memory
    this.memory.addFinalAnswer(finalOutput);

    this.logger.info(`\n⏱️ Total time: ${(duration / 1000).toFixed(2)}s`);
    this.logger.info(`📊 Total tokens: ${tokenUsage.totalTokens}`);

    const logPath = this.logger.getLogPath();
    if (logPath) {
      this.logger.info(`📁 Log file: ${logPath}`);
    }

    return {
      output: finalOutput,
      steps: this.memory.steps,
      tokenUsage,
      duration,
    };
  }

  /**
   * Generate a final answer when max steps is reached.
   */
  protected async provideFinalAnswer(task: string): Promise<unknown> {
    this.logger.subheader('Generating final answer from accumulated context');

    const messages = this.memory.toMessages();

    // Add prompt for final answer
    messages.push({
      role: 'user',
      content: `You have reached the maximum number of steps. Based on your work so far, provide the best answer you can for the original task: "${task}"

Summarize what you accomplished and provide a final answer. Call final_answer() with your response.`,
    });

    const response = await this.model.generate(messages);

    // Extract final answer from response
    // This is a simplified extraction - subclasses may override
    return response.content;
  }

  /**
   * Stop the agent.
   */
  stop(): void {
    this.isRunning = false;
    this.logger.info('Agent stopped by user');
  }

  /**
   * Get the current memory.
   */
  getMemory(): AgentMemory {
    return this.memory;
  }

  /**
   * Get registered tools.
   */
  getTools(): Map<string, Tool> {
    return this.tools;
  }

  /**
   * Add a tool to the agent.
   */
  addTool(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  /**
   * Remove a tool from the agent.
   */
  removeTool(name: string): boolean {
    return this.tools.delete(name);
  }

  /**
   * Sleep for a specified duration.
   */
  protected sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
