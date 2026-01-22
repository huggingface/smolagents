/**
 * AgentMemory - Tracks agent execution history
 *
 * Stores all steps taken by the agent and converts them to messages
 * that can be sent to the LLM for context.
 */

import type {
  MemoryStep,
  SystemPromptStep,
  TaskStep,
  ActionStep,
  FinalAnswerStep,
  ChatMessage,
  TokenUsage,
} from '../types.js';

export class AgentMemory {
  /**
   * System prompt step (always first)
   */
  systemPrompt: SystemPromptStep;

  /**
   * All execution steps
   */
  steps: (TaskStep | ActionStep | FinalAnswerStep)[] = [];

  constructor(systemPrompt: string) {
    this.systemPrompt = {
      type: 'system',
      content: systemPrompt,
      timestamp: Date.now(),
    };
  }

  /**
   * Reset memory, keeping only the system prompt.
   */
  reset(): void {
    this.steps = [];
  }

  /**
   * Add a task step.
   */
  addTask(task: string): TaskStep {
    const step: TaskStep = {
      type: 'task',
      task,
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /**
   * Create a new action step.
   */
  createActionStep(stepNumber: number): ActionStep {
    const step: ActionStep = {
      type: 'action',
      stepNumber,
      timing: {
        startTime: Date.now(),
      },
      modelInputMessages: [],
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /**
   * Add a final answer step.
   */
  addFinalAnswer(answer: unknown): FinalAnswerStep {
    const step: FinalAnswerStep = {
      type: 'final',
      answer,
      timestamp: Date.now(),
    };
    this.steps.push(step);
    return step;
  }

  /**
   * Get the last step.
   */
  getLastStep(): MemoryStep | undefined {
    return this.steps[this.steps.length - 1];
  }

  /**
   * Get all action steps.
   */
  getActionSteps(): ActionStep[] {
    return this.steps.filter((s): s is ActionStep => s.type === 'action');
  }

  /**
   * Convert memory to messages for LLM context.
   */
  toMessages(): ChatMessage[] {
    const messages: ChatMessage[] = [];

    // Add system prompt
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
          // Add assistant's response (reasoning + code)
          if (step.modelOutputMessage) {
            messages.push({
              role: 'assistant',
              content: step.modelOutputMessage.content,
            });
          }

          // Add observation as user message
          if (step.observation) {
            messages.push({
              role: 'user',
              content: step.observation,
            });
          }

          // Add error as user message
          if (step.error) {
            messages.push({
              role: 'user',
              content: `Error: ${step.error.message}`,
            });
          }
          break;

        case 'final':
          // Final answer doesn't need to be in messages
          break;
      }
    }

    return messages;
  }

  /**
   * Get total token usage across all steps.
   */
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

  /**
   * Get a summary of the memory for logging.
   */
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

  /**
   * Serialize memory to JSON.
   */
  toJSON(): Record<string, unknown> {
    return {
      systemPrompt: this.systemPrompt,
      steps: this.steps,
    };
  }
}
