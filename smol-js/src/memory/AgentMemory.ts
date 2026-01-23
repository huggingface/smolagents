/**
 * AgentMemory - Tracks agent execution history with context management
 */

import type {
  MemoryStep,
  SystemPromptStep,
  TaskStep,
  ActionStep,
  FinalAnswerStep,
  ChatMessage,
  TokenUsage,
  MemoryStrategy,
  Model,
} from '../types.js';

export interface MemoryConfig {
  maxContextLength?: number;
  memoryStrategy?: MemoryStrategy;
  model?: Model;
}

// Rough token estimation: ~4 chars per token on average
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessagesTokens(messages: ChatMessage[]): number {
  let total = 0;
  for (const msg of messages) {
    total += estimateTokens(msg.content ?? '');
    if (msg.toolCalls) {
      total += estimateTokens(JSON.stringify(msg.toolCalls));
    }
    total += 4; // overhead per message (role, etc.)
  }
  return total;
}

export class AgentMemory {
  /** System prompt step (always first) */
  systemPrompt: SystemPromptStep;

  /** All execution steps */
  steps: (TaskStep | ActionStep | FinalAnswerStep)[] = [];

  private maxContextLength: number;
  private memoryStrategy: MemoryStrategy;
  private model?: Model;

  constructor(systemPrompt: string, config?: MemoryConfig) {
    this.systemPrompt = {
      type: 'system',
      content: systemPrompt,
      timestamp: Date.now(),
    };
    this.maxContextLength = config?.maxContextLength ?? 100000;
    this.memoryStrategy = config?.memoryStrategy ?? 'truncate';
    this.model = config?.model;
  }

  /** Reset memory, keeping only the system prompt */
  reset(): void {
    this.steps = [];
  }

  /** Add a task step */
  addTask(task: string): TaskStep {
    const step: TaskStep = {
      type: 'task',
      task,
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /** Create a new action step */
  createActionStep(stepNumber: number): ActionStep {
    const step: ActionStep = {
      type: 'action',
      stepNumber,
      timing: { startTime: Date.now() },
      modelInputMessages: [],
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /** Add a final answer step */
  addFinalAnswer(answer: unknown): FinalAnswerStep {
    const step: FinalAnswerStep = {
      type: 'final',
      answer,
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /** Get the last step */
  getLastStep(): MemoryStep | undefined {
    return this.steps[this.steps.length - 1];
  }

  /** Get all action steps */
  getActionSteps(): ActionStep[] {
    return this.steps.filter((s): s is ActionStep => s.type === 'action');
  }

  /**
   * Convert memory to messages for LLM context.
   * Handles both CodeAgent (observation-based) and ToolUseAgent (tool_call-based) patterns.
   */
  toMessages(): ChatMessage[] {
    const messages: ChatMessage[] = [];

    // System prompt
    messages.push({
      role: 'system',
      content: this.systemPrompt.content,
    });

    for (const step of this.steps) {
      switch (step.type) {
        case 'task':
          messages.push({
            role: 'user',
            content: `Task: ${step.task}`,
          });
          break;

        case 'action':
          // Assistant response with tool calls (ToolUseAgent)
          if (step.toolCalls && step.toolCalls.length > 0) {
            messages.push({
              role: 'assistant',
              content: step.modelOutputMessage?.content ?? null,
              toolCalls: step.toolCalls,
            });

            // Add tool results
            if (step.toolResults) {
              for (const result of step.toolResults) {
                messages.push({
                  role: 'tool',
                  content: result.error
                    ? `Error: ${result.error}`
                    : typeof result.result === 'string'
                      ? result.result
                      : JSON.stringify(result.result, null, 2),
                  toolCallId: result.toolCallId,
                });
              }
            }
          } else {
            // CodeAgent pattern: assistant message + observation
            if (step.modelOutputMessage) {
              messages.push({
                role: 'assistant',
                content: step.modelOutputMessage.content,
              });
            }

            if (step.observation) {
              messages.push({
                role: 'user',
                content: step.observation,
              });
            }

            if (step.error && !step.observation) {
              messages.push({
                role: 'user',
                content: `Error: ${step.error.message}`,
              });
            }
          }
          break;

        case 'final':
          break;
      }
    }

    return messages;
  }

  /**
   * Manage context length - truncate or compact if exceeded.
   */
  async manageContext(): Promise<void> {
    const messages = this.toMessages();
    const tokenCount = estimateMessagesTokens(messages);

    if (tokenCount <= this.maxContextLength) {
      return;
    }

    if (this.memoryStrategy === 'truncate') {
      this.truncateOlderMessages();
    } else if (this.memoryStrategy === 'compact') {
      await this.compactMessages();
    }
  }

  /**
   * Truncate older action steps to fit within context.
   */
  private truncateOlderMessages(): void {
    // Keep system prompt, task, and recent steps
    const actionSteps = this.getActionSteps();
    if (actionSteps.length <= 2) return;

    // Remove oldest action steps until we fit
    const targetTokens = this.maxContextLength * 0.75;
    let currentTokens = estimateMessagesTokens(this.toMessages());

    while (currentTokens > targetTokens && this.steps.length > 2) {
      // Find first action step and remove it
      const idx = this.steps.findIndex(s => s.type === 'action');
      if (idx === -1) break;
      this.steps.splice(idx, 1);
      currentTokens = estimateMessagesTokens(this.toMessages());
    }
  }

  /**
   * Compact older messages into a summary.
   */
  private async compactMessages(): Promise<void> {
    if (!this.model) {
      // Fall back to truncation if no model available
      this.truncateOlderMessages();
      return;
    }

    const actionSteps = this.getActionSteps();
    if (actionSteps.length <= 2) return;

    // Summarize older steps
    const stepsToSummarize = actionSteps.slice(0, -2);
    const summaryContent = stepsToSummarize.map(step => {
      const parts: string[] = [];
      if (step.modelOutputMessage?.content) {
        parts.push(`Action: ${step.modelOutputMessage.content.slice(0, 200)}`);
      }
      if (step.observation) {
        parts.push(`Observation: ${step.observation.slice(0, 200)}`);
      }
      if (step.toolResults) {
        for (const r of step.toolResults) {
          const resultStr = typeof r.result === 'string' ? r.result : JSON.stringify(r.result);
          parts.push(`Tool ${r.toolName}: ${resultStr.slice(0, 200)}`);
        }
      }
      return parts.join('\n');
    }).join('\n---\n');

    try {
      const summaryResponse = await this.model.generate([
        {
          role: 'system',
          content: 'Summarize the following agent execution history concisely, preserving key findings and results. Be brief but complete.',
        },
        {
          role: 'user',
          content: summaryContent,
        },
      ]);

      // Remove the summarized steps and replace with a summary task step
      const recentSteps = this.steps.filter(s =>
        s.type === 'task' || s.type === 'final' ||
        (s.type === 'action' && actionSteps.indexOf(s) >= actionSteps.length - 2)
      );

      this.steps = [
        {
          type: 'task' as const,
          task: `[Context Summary from previous steps]\n${summaryResponse.content}`,
          timestamp: Date.now(),
        },
        ...recentSteps,
      ];
    } catch {
      // Fall back to truncation
      this.truncateOlderMessages();
    }
  }

  /** Get total token usage across all steps */
  getTotalTokenUsage(): TokenUsage {
    let inputTokens = 0;
    let outputTokens = 0;

    for (const step of this.steps) {
      if (step.type === 'action' && step.tokenUsage) {
        inputTokens += step.tokenUsage.inputTokens;
        outputTokens += step.tokenUsage.outputTokens;
      }
    }

    return {
      inputTokens,
      outputTokens,
      totalTokens: inputTokens + outputTokens,
    };
  }

  /** Get current estimated token count */
  getEstimatedTokenCount(): number {
    return estimateMessagesTokens(this.toMessages());
  }

  /** Get a summary of the memory for logging */
  getSummary(): string {
    const actionSteps = this.getActionSteps();
    const lines = [
      `System Prompt: ${this.systemPrompt.content.slice(0, 100)}...`,
      `Total Steps: ${this.steps.length}`,
      `Action Steps: ${actionSteps.length}`,
    ];

    const tokenUsage = this.getTotalTokenUsage();
    if (tokenUsage.totalTokens > 0) {
      lines.push(`Total Tokens: ${tokenUsage.totalTokens}`);
    }

    return lines.join('\n');
  }

  /** Serialize memory to JSON */
  toJSON(): Record<string, unknown> {
    return {
      systemPrompt: this.systemPrompt,
      steps: this.steps,
    };
  }
}
