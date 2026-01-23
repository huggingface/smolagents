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
  MemoryStrategy,
} from '../types.js';
import { Tool } from '../tools/Tool.js';
import { Model } from '../models/Model.js';
import { AgentMemory } from '../memory/AgentMemory.js';
import { AgentLogger } from '../logging/AgentLogger.js';
import { LogLevel as LogLevelEnum } from '../types.js';

// Default global max context length (in tokens, estimated)
const DEFAULT_MAX_CONTEXT_LENGTH = 100000;

export interface AgentConfig {
  /** The LLM model to use for generation */
  model: Model;

  /** Tools available to the agent */
  tools?: Tool[];

  /** Maximum number of steps before stopping (default: 20) */
  maxSteps?: number;

  /** Delay in ms before executing code (default: 5000) */
  codeExecutionDelay?: number;

  /** Custom instructions appended to system prompt */
  customInstructions?: string;

  /** Log level for output (default: INFO) */
  verboseLevel?: LogLevel;

  /** Whether to stream model outputs (default: true) */
  streamOutputs?: boolean;

  /** Whether the agent retains memory between run() calls (default: false) */
  persistent?: boolean;

  /** Max context length in tokens (default: 100000) */
  maxContextLength?: number;

  /** Memory management strategy when context is exceeded (default: 'truncate') */
  memoryStrategy?: MemoryStrategy;

  /** Max tokens for generation (passed to model if set) */
  maxTokens?: number;

  /** Temperature for generation (passed to model if set) */
  temperature?: number;

  /** Agent name for logging */
  name?: string;

  /** Callback for orchestration events */
  onEvent?: (event: { type: string; data: unknown }) => void;
}

export abstract class Agent {
  /** The LLM model for generation */
  protected model: Model;

  /** Available tools mapped by name */
  protected tools: Map<string, Tool> = new Map();

  /** Agent memory tracking all steps */
  protected memory!: AgentMemory;

  /** Logger for formatted output */
  protected logger: AgentLogger;

  /** Configuration options */
  protected config: {
    maxSteps: number;
    codeExecutionDelay: number;
    customInstructions: string;
    verboseLevel: LogLevel;
    streamOutputs: boolean;
    persistent: boolean;
    maxContextLength: number;
    memoryStrategy: MemoryStrategy;
    maxTokens?: number;
    temperature?: number;
    name: string;
    onEvent?: (event: { type: string; data: unknown }) => void;
  };

  /** Current step number */
  protected currentStep: number = 0;

  /** Whether the agent is currently running */
  protected isRunning: boolean = false;

  /** Whether the agent has been initialized at least once */
  private initialized: boolean = false;

  constructor(config: AgentConfig) {
    this.model = config.model;
    this.logger = new AgentLogger(config.verboseLevel ?? LogLevelEnum.INFO);

    this.config = {
      maxSteps: config.maxSteps ?? 20,
      codeExecutionDelay: config.codeExecutionDelay ?? 5000,
      customInstructions: config.customInstructions ?? '',
      verboseLevel: config.verboseLevel ?? LogLevelEnum.INFO,
      streamOutputs: config.streamOutputs ?? true,
      persistent: config.persistent ?? false,
      maxContextLength: config.maxContextLength ?? DEFAULT_MAX_CONTEXT_LENGTH,
      memoryStrategy: config.memoryStrategy ?? 'truncate',
      maxTokens: config.maxTokens,
      temperature: config.temperature,
      name: config.name ?? 'Agent',
      onEvent: config.onEvent,
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
   */
  protected abstract executeStep(memoryStep: ActionStep): Promise<ActionOutput>;

  /**
   * Run the agent on a task.
   */
  async run(task: string, reset: boolean = true): Promise<RunResult> {
    const startTime = Date.now();

    // For persistent agents, only reset if explicitly requested or first run
    const shouldReset = !this.config.persistent ? (reset || !this.memory) : (!this.initialized);

    if (shouldReset) {
      const systemPrompt = this.initializeSystemPrompt();
      this.memory = new AgentMemory(systemPrompt, {
        maxContextLength: this.config.maxContextLength,
        memoryStrategy: this.config.memoryStrategy,
        model: this.model,
      });
      this.currentStep = 0;
      this.initialized = true;
    }

    // Add task to memory
    this.memory.addTask(task);

    this.isRunning = true;
    this.emitEvent('agent_start', { task, name: this.config.name });
    this.logger.header(`Starting ${this.config.name}: ${task.slice(0, 80)}${task.length > 80 ? '...' : ''}`);

    let finalOutput: unknown = null;
    let isFinalAnswer = false;

    try {
      while (this.currentStep < this.config.maxSteps && this.isRunning) {
        this.currentStep++;
        this.logger.stepProgress(this.currentStep, this.config.maxSteps);
        this.emitEvent('agent_step', { step: this.currentStep, maxSteps: this.config.maxSteps });

        const memoryStep = this.memory.createActionStep(this.currentStep);

        try {
          const actionOutput = await this.executeStep(memoryStep);

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
          memoryStep.error = error as Error;
          memoryStep.timing.endTime = Date.now();
          memoryStep.timing.duration = memoryStep.timing.endTime - memoryStep.timing.startTime;
          this.logger.error('Step execution failed', error as Error);
          this.emitEvent('agent_error', { error: (error as Error).message, step: this.currentStep });
        }

        // Check and manage context length after each step
        await this.memory.manageContext();
      }

      if (!isFinalAnswer && this.currentStep >= this.config.maxSteps) {
        this.logger.warn(`Max steps (${this.config.maxSteps}) reached without final answer`);
        finalOutput = await this.provideFinalAnswer(task);
      }
    } finally {
      this.isRunning = false;
    }

    const duration = Date.now() - startTime;
    const tokenUsage = this.memory.getTotalTokenUsage();

    this.memory.addFinalAnswer(finalOutput);
    this.emitEvent('agent_end', { output: finalOutput, duration, tokenUsage });

    this.logger.info(`\nTotal time: ${(duration / 1000).toFixed(2)}s`);
    this.logger.info(`Total tokens: ${tokenUsage.totalTokens}`);

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
    messages.push({
      role: 'user',
      content: `You have reached the maximum number of steps. Based on your work so far, provide the best answer you can for the original task: "${task}". Summarize what you accomplished and provide a final answer.`,
    });

    const response = await this.model.generate(messages, {
      maxTokens: this.config.maxTokens,
      temperature: this.config.temperature,
    });

    return response.content;
  }

  /** Emit an orchestration event */
  protected emitEvent(type: string, data: unknown): void {
    if (this.config.onEvent) {
      this.config.onEvent({ type, data });
    }
  }

  /** Stop the agent */
  stop(): void {
    this.isRunning = false;
    this.logger.info('Agent stopped by user');
  }

  /** Get the current memory */
  getMemory(): AgentMemory {
    return this.memory;
  }

  /** Get registered tools */
  getTools(): Map<string, Tool> {
    return this.tools;
  }

  /** Add a tool to the agent */
  addTool(tool: Tool): void {
    this.tools.set(tool.name, tool);
  }

  /** Remove a tool from the agent */
  removeTool(name: string): boolean {
    return this.tools.delete(name);
  }

  /** Get agent name */
  getName(): string {
    return this.config.name;
  }

  /** Sleep for a specified duration */
  protected sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
