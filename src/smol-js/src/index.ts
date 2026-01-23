/**
 * smol-js - A TypeScript agentic framework supporting CodeAgent and ToolUseAgent
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
  ToolCallResult,
  ToolInputType,
  ToolInput,
  ToolInputs,
  OpenAIToolDefinition,
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
  MemoryStrategy,
  YAMLAgentDefinition,
  YAMLModelDefinition,
  YAMLWorkflowDefinition,
  YAMLToolDefinition,
  OrchestratorEvent,
} from './types.js';

export { LogLevel } from './types.js';

// Agents
export { Agent, type AgentConfig } from './agents/Agent.js';
export { CodeAgent, type CodeAgentConfig } from './agents/CodeAgent.js';
export { ToolUseAgent, type ToolUseAgentConfig } from './agents/ToolUseAgent.js';

// Models
export { Model } from './models/Model.js';
export { OpenAIModel, type OpenAIModelConfig } from './models/OpenAIModel.js';

// Tools
export { Tool, createTool } from './tools/Tool.js';
export { FinalAnswerTool, UserInputTool, finalAnswerTool } from './tools/defaultTools.js';
export { AgentTool, agentAsTool, type AgentToolConfig } from './tools/AgentTool.js';
export { ReadFileTool } from './tools/ReadFileTool.js';
export { WriteFileTool } from './tools/WriteFileTool.js';
export { CurlTool } from './tools/CurlTool.js';
export { ExaSearchTool } from './tools/ExaSearchTool.js';
export { ExaGetContentsTool } from './tools/ExaGetContentsTool.js';
export { ExaResearchTool } from './tools/ExaResearchTool.js';

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

export {
  generateToolUseSystemPrompt,
  formatToolDescriptions,
  type ToolUsePromptVariables,
} from './prompts/toolUseAgent.js';

// Orchestrator
export { YAMLLoader, type LoadedWorkflow } from './orchestrator/YAMLLoader.js';
export { Orchestrator, type OrchestratorConfig } from './orchestrator/Orchestrator.js';
