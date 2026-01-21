/**
 * Core types for smol-js
 */

// Message roles following OpenAI chat completions format
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

// Token usage tracking
export interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

// Timing information for steps
export interface Timing {
  startTime: number;
  endTime?: number;
  duration?: number;
}

// Chat message structure
export interface ChatMessage {
  role: MessageRole;
  content: string | null;
  name?: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
  tokenUsage?: TokenUsage;
}

// Tool call structure
export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string | Record<string, unknown>;
  };
}

// Tool input schema type
export type ToolInputType =
  | 'string'
  | 'number'
  | 'boolean'
  | 'array'
  | 'object'
  | 'any';

// Tool input definition
export interface ToolInput {
  type: ToolInputType;
  description: string;
  required?: boolean;
  default?: unknown;
}

// Tool inputs schema
export interface ToolInputs {
  [key: string]: ToolInput;
}

// Code execution output
export interface CodeExecutionOutput {
  output: unknown;
  logs: string;
  isFinalAnswer: boolean;
  error?: Error;
}

// Action output from a step
export interface ActionOutput {
  output: unknown;
  isFinalAnswer: boolean;
}

// Agent configuration
export interface AgentConfig {
  model: Model;
  tools?: Tool[];
  maxSteps?: number;
  codeExecutionDelay?: number;
  systemPrompt?: string;
  additionalAuthorizedImports?: string[];
  streamOutputs?: boolean;
  verboseLevel?: LogLevel;
}

// Model configuration
export interface ModelConfig {
  modelId?: string;
  apiKey?: string;
  baseUrl?: string;
  maxTokens?: number;
  temperature?: number;
  timeout?: number;
}

// Log levels
export enum LogLevel {
  OFF = -1,
  ERROR = 0,
  INFO = 1,
  DEBUG = 2,
}

// Step types for memory
export type StepType = 'system' | 'task' | 'action' | 'planning' | 'final';

// Base memory step
export interface MemoryStep {
  type: StepType;
  timestamp: number;
}

// System prompt step
export interface SystemPromptStep extends MemoryStep {
  type: 'system';
  content: string;
}

// Task step
export interface TaskStep extends MemoryStep {
  type: 'task';
  task: string;
}

// Action step - main execution step
export interface ActionStep extends MemoryStep {
  type: 'action';
  stepNumber: number;
  timing: Timing;
  modelInputMessages: ChatMessage[];
  modelOutputMessage?: ChatMessage;
  codeAction?: string;
  observation?: string;
  actionOutput?: ActionOutput;
  tokenUsage?: TokenUsage;
  error?: Error;
  isFinalAnswer?: boolean;
}

// Final answer step
export interface FinalAnswerStep extends MemoryStep {
  type: 'final';
  answer: unknown;
}

// Stream event types
export interface StreamEvent {
  type: 'delta' | 'toolCall' | 'observation' | 'step' | 'final' | 'error';
  data: unknown;
}

// Run result
export interface RunResult {
  output: unknown;
  steps: MemoryStep[];
  tokenUsage: TokenUsage;
  duration: number;
}

// Forward declarations for circular deps
export interface Tool {
  name: string;
  description: string;
  inputs: ToolInputs;
  outputType: string;
  execute: (args: Record<string, unknown>) => Promise<unknown>;
  toCodePrompt: () => string;
}

export interface Model {
  modelId: string;
  generate: (
    messages: ChatMessage[],
    options?: GenerateOptions
  ) => Promise<ChatMessage>;
  generateStream?: (
    messages: ChatMessage[],
    options?: GenerateOptions
  ) => AsyncGenerator<string, ChatMessage, undefined>;
}

// Generation options
export interface GenerateOptions {
  stopSequences?: string[];
  maxTokens?: number;
  temperature?: number;
  tools?: Tool[];
}
