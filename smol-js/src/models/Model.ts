/**
 * Model base class for smol-js
 *
 * Models are responsible for generating text responses from LLMs.
 * Extend this class to support different LLM providers.
 */

import type { ChatMessage, GenerateOptions, TokenUsage } from '../types.js';

export abstract class Model {
  /**
   * Model identifier (e.g., "gpt-4", "claude-3-sonnet")
   */
  abstract readonly modelId: string;

  /**
   * Generate a response from the model.
   */
  abstract generate(
    messages: ChatMessage[],
    options?: GenerateOptions
  ): Promise<ChatMessage>;

  /**
   * Optional streaming generation.
   * Yields content chunks and returns the final message.
   */
  generateStream?(
    messages: ChatMessage[],
    options?: GenerateOptions
  ): AsyncGenerator<string, ChatMessage, undefined>;

  /**
   * Check if the model supports streaming.
   */
  supportsStreaming(): boolean {
    return typeof this.generateStream === 'function';
  }

  /**
   * Extract token usage from a response message.
   */
  protected extractTokenUsage(_response: unknown): TokenUsage | undefined {
    // Override in subclasses to extract token usage from API responses
    return undefined;
  }

  /**
   * Convert messages to the format expected by the model's API.
   */
  protected formatMessages(messages: ChatMessage[]): unknown[] {
    // Default implementation - override for specific API formats
    return messages.map((msg) => ({
      role: msg.role,
      content: msg.content,
      ...(msg.name && { name: msg.name }),
      ...(msg.toolCallId && { tool_call_id: msg.toolCallId }),
    }));
  }
}
