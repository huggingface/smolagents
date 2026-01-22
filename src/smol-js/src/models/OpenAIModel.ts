/**
 * OpenAI-compatible Model implementation
 *
 * Supports any API that follows the OpenAI chat completions format,
 * including OpenRouter, Azure OpenAI, local servers, etc.
 */

import OpenAI from 'openai';
import { Model } from './Model.js';
import type { ChatMessage, GenerateOptions, TokenUsage, MessageRole } from '../types.js';

export interface OpenAIModelConfig {
  /**
   * Model identifier (e.g., "gpt-4", "anthropic/claude-sonnet-4.5")
   */
  modelId?: string;

  /**
   * API key for authentication
   */
  apiKey?: string;

  /**
   * Base URL for the API endpoint
   * @default "https://openrouter.ai/api/v1"
   */
  baseUrl?: string;

  /**
   * Maximum tokens to generate
   */
  maxTokens?: number;

  /**
   * Temperature for generation (0-2)
   */
  temperature?: number;

  /**
   * Request timeout in milliseconds
   */
  timeout?: number;

  /**
   * Default headers to include in requests
   */
  defaultHeaders?: Record<string, string>;
}

// Default configuration uses OpenRouter with Claude Sonnet
const DEFAULT_CONFIG: Required<Pick<OpenAIModelConfig, 'modelId' | 'baseUrl' | 'maxTokens' | 'temperature' | 'timeout'>> = {
  modelId: 'anthropic/claude-sonnet-4.5',
  baseUrl: 'https://openrouter.ai/api/v1',
  maxTokens: 65000,
  temperature: 1,
  timeout: 120000,
};

export class OpenAIModel extends Model {
  readonly modelId: string;
  private client: OpenAI;
  private config: OpenAIModelConfig;

  constructor(config: OpenAIModelConfig = {}) {
    super();

    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
    };

    this.modelId = this.config.modelId ?? DEFAULT_CONFIG.modelId;

    // Get API key from config or environment
    const apiKey = this.config.apiKey ?? process.env.OPENAI_API_KEY ?? process.env.OPENROUTER_API_KEY;

    if (!apiKey) {
      throw new Error(
        'API key is required. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable, or pass apiKey in config.'
      );
    }

    this.client = new OpenAI({
      apiKey,
      baseURL: this.config.baseUrl,
      timeout: this.config.timeout,
      defaultHeaders: this.config.defaultHeaders,
    });
  }

  /**
   * Generate a response from the model.
   */
  async generate(messages: ChatMessage[], options: GenerateOptions = {}): Promise<ChatMessage> {
    const formattedMessages = this.formatMessages(messages);

    const response = await this.client.chat.completions.create({
      model: this.modelId,
      messages: formattedMessages as OpenAI.Chat.ChatCompletionMessageParam[],
      max_tokens: options.maxTokens ?? this.config.maxTokens,
      temperature: options.temperature ?? this.config.temperature,
      ...(options.stopSequences && { stop: options.stopSequences }),
    });

    const choice = response.choices[0];
    const message = choice?.message;

    if (!message) {
      throw new Error('No response from model');
    }

    const tokenUsage: TokenUsage | undefined = response.usage
      ? {
          inputTokens: response.usage.prompt_tokens,
          outputTokens: response.usage.completion_tokens,
          totalTokens: response.usage.total_tokens,
        }
      : undefined;

    return {
      role: 'assistant' as MessageRole,
      content: message.content ?? '',
      tokenUsage,
    };
  }

  /**
   * Generate a streaming response from the model.
   */
  async *generateStream(
    messages: ChatMessage[],
    options: GenerateOptions = {}
  ): AsyncGenerator<string, ChatMessage, undefined> {
    const formattedMessages = this.formatMessages(messages);

    const stream = await this.client.chat.completions.create({
      model: this.modelId,
      messages: formattedMessages as OpenAI.Chat.ChatCompletionMessageParam[],
      max_tokens: options.maxTokens ?? this.config.maxTokens,
      temperature: options.temperature ?? this.config.temperature,
      ...(options.stopSequences && { stop: options.stopSequences }),
      stream: true,
    });

    let fullContent = '';

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta;
      if (delta?.content) {
        fullContent += delta.content;
        yield delta.content;
      }
    }

    return {
      role: 'assistant' as MessageRole,
      content: fullContent,
    };
  }

  /**
   * Format messages for the OpenAI API.
   */
  protected formatMessages(messages: ChatMessage[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    return messages.map((msg) => {
      // Handle tool responses
      if (msg.role === 'tool') {
        return {
          role: 'user' as const,
          content: msg.content ?? '',
        };
      }

      return {
        role: msg.role as 'system' | 'user' | 'assistant',
        content: msg.content ?? '',
      };
    });
  }
}
