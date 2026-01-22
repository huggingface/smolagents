/**
 * smol-js - A TypeScript port of the smolagents agentic framework
 *
 * @packageDocumentation
 */

// Core types
export type {
  MessageRole,
  TokenUsage,
  Timing,
  ChatMessage,
  ToolCall,
  ToolInputType,
  ToolInput,
  ToolInputs,
  CodeExecutionOutput,
  ActionOutput,
  AgentConfig as AgentConfigType,
  ModelConfig,
  MemoryStep,
  SystemPromptStep,
  TaskStep,
  ActionStep,
  FinalAnswerStep,
  StreamEvent,
  RunResult,
  GenerateOptions,
} from './types.js';

export { LogLevel } from './types.js';

// Agents
export { Agent, type AgentConfig } from './agents/Agent.js';
export { CodeAgent, type CodeAgentConfig } from './agents/CodeAgent.js';

// Models
export { Model } from './models/Model.js';
export { OpenAIModel, type OpenAIModelConfig } from './models/OpenAIModel.js';

// Tools
export { Tool, createTool } from './tools/Tool.js';
export { FinalAnswerTool, UserInputTool, finalAnswerTool } from './tools/defaultTools.js';
export { AgentTool, agentAsTool, type AgentToolConfig } from './tools/AgentTool.js';

// Executor
export { LocalExecutor, type ExecutorConfig } from './executor/LocalExecutor.js';

// Memory
export { AgentMemory } from './memory/AgentMemory.js';

// Logging
export { AgentLogger } from './logging/AgentLogger.js';

// Prompts
export {
  generateSystemPrompt,
  FINAL_ANSWER_PROMPT,
  getErrorRecoveryPrompt,
  type PromptVariables,
} from './prompts/codeAgent.js';
