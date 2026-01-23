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
   * Maximum tokens to generate (omitted from requests by default)
   */
  maxTokens?: number;

  /**
   * Temperature for generation (omitted from requests by default)
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

const DEFAULT_MODEL_ID = 'anthropic/claude-sonnet-4.5';
const DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1';
const DEFAULT_TIMEOUT = 120000;

export class OpenAIModel extends Model {
  readonly modelId: string;
  private client: OpenAI;
  private config: OpenAIModelConfig;

  constructor(config: OpenAIModelConfig = {}) {
    super();

    this.config = config;
    this.modelId = config.modelId ?? DEFAULT_MODEL_ID;

    // Get API key from config or environment
    const apiKey = config.apiKey ?? process.env.OPENAI_API_KEY ?? process.env.OPENROUTER_API_KEY;

    if (!apiKey) {
      throw new Error(
        'API key is required. Set OPENAI_API_KEY or OPENROUTER_API_KEY environment variable, or pass apiKey in config.'
      );
    }

    this.client = new OpenAI({
      apiKey,
      baseURL: config.baseUrl ?? DEFAULT_BASE_URL,
      timeout: config.timeout ?? DEFAULT_TIMEOUT,
      defaultHeaders: config.defaultHeaders,
    });
  }

  /**
   * Generate a response from the model (supports tool calling).
   */
  async generate(messages: ChatMessage[], options: GenerateOptions = {}): Promise<ChatMessage> {
    const formattedMessages = this.formatMessages(messages);

    const requestParams: Record<string, unknown> = {
      model: this.modelId,
      messages: formattedMessages,
    };

    // Only include maxTokens if specified at request or model config level
    const maxTokens = options.maxTokens ?? this.config.maxTokens;
    if (maxTokens !== undefined) {
      requestParams.max_tokens = maxTokens;
    }

    // Only include temperature if specified at request or model config level
    const temperature = options.temperature ?? this.config.temperature;
    if (temperature !== undefined) {
      requestParams.temperature = temperature;
    }

    if (options.stopSequences) {
      requestParams.stop = options.stopSequences;
    }

    // Add tool definitions for function calling
    if (options.toolDefinitions && options.toolDefinitions.length > 0) {
      requestParams.tools = options.toolDefinitions;
    } else if (options.tools && options.tools.length > 0) {
      requestParams.tools = options.tools.map(t => t.toOpenAITool());
    }

    const response = await this.client.chat.completions.create({
      ...requestParams,
      model: this.modelId,
      messages: formattedMessages as OpenAI.Chat.ChatCompletionMessageParam[],
    } as OpenAI.Chat.ChatCompletionCreateParamsNonStreaming);

    const choice = response.choices[0];
    const message = choice?.message;

    if (!message) {
      throw new Error('No response from model');
    }

    const tokenUsage: TokenUsage | undefined = response.usage
      ? {
          inputTokens: response.usage.prompt_tokens,
          outputTokens: response.usage.completion_tokens ?? 0,
          totalTokens: response.usage.total_tokens,
        }
      : undefined;

    // Map tool_calls from OpenAI format
    const toolCalls = message.tool_calls?.map((tc: OpenAI.Chat.ChatCompletionMessageToolCall) => ({
      id: tc.id,
      type: 'function' as const,
      function: {
        name: tc.function.name,
        arguments: tc.function.arguments,
      },
    }));

    return {
      role: 'assistant' as MessageRole,
      content: message.content ?? '',
      toolCalls,
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

    const requestParams: Record<string, unknown> = {
      model: this.modelId,
      messages: formattedMessages,
      stream: true,
    };

    const maxTokens = options.maxTokens ?? this.config.maxTokens;
    if (maxTokens !== undefined) {
      requestParams.max_tokens = maxTokens;
    }

    const temperature = options.temperature ?? this.config.temperature;
    if (temperature !== undefined) {
      requestParams.temperature = temperature;
    }

    if (options.stopSequences) {
      requestParams.stop = options.stopSequences;
    }

    const stream = await this.client.chat.completions.create({
      ...requestParams,
      model: this.modelId,
      messages: formattedMessages as OpenAI.Chat.ChatCompletionMessageParam[],
      stream: true,
    } as OpenAI.Chat.ChatCompletionCreateParamsStreaming);

    let fullContent = '';

    for await (const chunk of stream as AsyncIterable<OpenAI.Chat.ChatCompletionChunk>) {
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
   * Format messages for the OpenAI API, including tool call/response messages.
   */
  protected formatMessages(messages: ChatMessage[]): OpenAI.Chat.ChatCompletionMessageParam[] {
    return messages.map((msg) => {
      // Handle tool response messages
      if (msg.role === 'tool' && msg.toolCallId) {
        return {
          role: 'tool' as const,
          content: msg.content ?? '',
          tool_call_id: msg.toolCallId,
        };
      }

      // Handle tool responses without toolCallId (legacy format)
      if (msg.role === 'tool') {
        return {
          role: 'user' as const,
          content: msg.content ?? '',
        };
      }

      // Handle assistant messages with tool calls
      if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
        return {
          role: 'assistant' as const,
          content: msg.content || null,
          tool_calls: msg.toolCalls.map(tc => ({
            id: tc.id,
            type: 'function' as const,
            function: {
              name: tc.function.name,
              arguments: typeof tc.function.arguments === 'string'
                ? tc.function.arguments
                : JSON.stringify(tc.function.arguments),
            },
          })),
        };
      }

      return {
        role: msg.role as 'system' | 'user' | 'assistant',
        content: msg.content ?? '',
      };
    });
  }
}
