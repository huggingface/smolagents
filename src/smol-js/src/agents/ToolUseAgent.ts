/**
 * ToolUseAgent - Executes tasks using standard OpenAI-style tool calls
 *
 * Unlike CodeAgent which generates and executes JavaScript code,
 * ToolUseAgent operates by making tool calls through the LLM's native
 * function calling capabilities, following the ReACT pattern:
 * Think -> Act (tool call) -> Observe (result) -> repeat
 */

import { Agent, AgentConfig } from './Agent.js';
import { FinalAnswerTool } from '../tools/defaultTools.js';
import { Tool } from '../tools/Tool.js';
import { generateToolUseSystemPrompt, formatToolDescriptions } from '../prompts/toolUseAgent.js';
import type { ActionStep, ActionOutput, ToolCall, ToolCallResult } from '../types.js';

export interface ToolUseAgentConfig extends AgentConfig {
  /** Whether to run independent tool calls in parallel (default: true) */
  parallelToolCalls?: boolean;
}

export class ToolUseAgent extends Agent {
  private parallelToolCalls: boolean;

  constructor(config: ToolUseAgentConfig) {
    super(config);
    this.parallelToolCalls = config.parallelToolCalls ?? true;

    // Always add final_answer tool
    if (!this.tools.has('final_answer')) {
      this.tools.set('final_answer', new FinalAnswerTool());
    }
  }

  /**
   * Initialize the system prompt with tool descriptions.
   */
  protected initializeSystemPrompt(): string {
    const toolList = Array.from(this.tools.values());
    const toolDescriptions = formatToolDescriptions(
      toolList.map(t => ({
        name: t.name,
        description: t.description,
        inputs: t.inputs,
      }))
    );

    return generateToolUseSystemPrompt({
      tools: toolDescriptions,
      customInstructions: this.config.customInstructions,
    });
  }

  /**
   * Execute a single step: send messages with tool definitions, process tool calls.
   */
  protected async executeStep(memoryStep: ActionStep): Promise<ActionOutput> {
    const messages = this.memory.toMessages();
    memoryStep.modelInputMessages = [...messages];

    // Handle error recovery from previous step
    const actionSteps = this.memory.getActionSteps();
    const prevStep = actionSteps.length >= 2 ? actionSteps[actionSteps.length - 2] : undefined;
    if (prevStep?.error) {
      messages.push({
        role: 'user',
        content: `Your previous action encountered an error: ${prevStep.error.message}\nPlease try a different approach.`,
      });
    }

    // Get tool definitions for the API call
    const toolDefinitions = Array.from(this.tools.values()).map(t => t.toOpenAITool());

    // Generate response with tool calling
    this.logger.subheader('Agent thinking...');
    const response = await this.model.generate(messages, {
      toolDefinitions,
      maxTokens: this.config.maxTokens,
      temperature: this.config.temperature,
    });

    memoryStep.modelOutputMessage = response;
    memoryStep.tokenUsage = response.tokenUsage;

    // Log reasoning (text content)
    if (response.content && response.content.trim()) {
      this.logger.reasoning(response.content.trim());
    }

    // Check if model made tool calls
    if (!response.toolCalls || response.toolCalls.length === 0) {
      // No tool calls - the model just responded with text
      this.logger.warn('No tool calls in response. Prompting model to use tools.');
      memoryStep.observation = 'You must use tools to complete the task. Please call the appropriate tool(s). When you have the final answer, call the final_answer tool.';
      return { output: null, isFinalAnswer: false };
    }

    // Process tool calls
    memoryStep.toolCalls = response.toolCalls;
    const toolResults = await this.processToolCalls(response.toolCalls);
    memoryStep.toolResults = toolResults;

    // Check if final_answer was called
    for (const result of toolResults) {
      if (result.toolName === 'final_answer') {
        return { output: result.result, isFinalAnswer: true };
      }
    }

    // Log tool results as observations
    for (const result of toolResults) {
      if (result.error) {
        this.logger.error(`Tool ${result.toolName} failed: ${result.error}`);
        this.emitEvent('agent_error', { tool: result.toolName, error: result.error });
      } else {
        const resultStr = typeof result.result === 'string'
          ? result.result
          : JSON.stringify(result.result, null, 2);
        this.logger.output(`[${result.toolName}]: ${resultStr.slice(0, 500)}${resultStr.length > 500 ? '...' : ''}`);
        this.emitEvent('agent_observation', { tool: result.toolName, result: resultStr.slice(0, 500) });
      }
    }

    return { output: null, isFinalAnswer: false };
  }

  /**
   * Process tool calls from the model response.
   */
  private async processToolCalls(toolCalls: ToolCall[]): Promise<ToolCallResult[]> {
    const results: ToolCallResult[] = [];

    // Execute tool calls
    const executeTool = async (tc: ToolCall): Promise<ToolCallResult> => {
      const toolName = tc.function.name;
      const tool = this.tools.get(toolName);

      if (!tool) {
        return {
          toolCallId: tc.id,
          toolName,
          result: null,
          error: `Unknown tool: ${toolName}. Available tools: ${Array.from(this.tools.keys()).join(', ')}`,
        };
      }

      // Parse arguments
      let args: Record<string, unknown>;
      try {
        args = typeof tc.function.arguments === 'string'
          ? JSON.parse(tc.function.arguments)
          : tc.function.arguments as Record<string, unknown>;
      } catch {
        return {
          toolCallId: tc.id,
          toolName,
          result: null,
          error: `Failed to parse tool arguments: ${tc.function.arguments}`,
        };
      }

      this.logger.info(`  Calling tool: ${toolName}(${JSON.stringify(args).slice(0, 100)}...)`);
      this.emitEvent('agent_tool_call', { tool: toolName, args });

      try {
        const result = await tool.call(args);
        return {
          toolCallId: tc.id,
          toolName,
          result,
        };
      } catch (error) {
        return {
          toolCallId: tc.id,
          toolName,
          result: null,
          error: `Tool execution error: ${(error as Error).message}`,
        };
      }
    };

    if (this.parallelToolCalls) {
      const promises = toolCalls.map(tc => executeTool(tc));
      const resolvedResults = await Promise.all(promises);
      results.push(...resolvedResults);
    } else {
      for (const tc of toolCalls) {
        results.push(await executeTool(tc));
      }
    }

    return results;
  }

  /**
   * Override provideFinalAnswer to use tool calling format.
   */
  protected async provideFinalAnswer(task: string): Promise<unknown> {
    this.logger.subheader('Generating final answer from accumulated context');

    const messages = this.memory.toMessages();
    messages.push({
      role: 'user',
      content: `You have reached the maximum number of steps. Based on your work so far, provide the best answer for the task: "${task}". Call the final_answer tool with your response.`,
    });

    const toolDefinitions = [new FinalAnswerTool().toOpenAITool()];
    const response = await this.model.generate(messages, {
      toolDefinitions,
      maxTokens: this.config.maxTokens,
      temperature: this.config.temperature,
    });

    // Try to extract from tool call
    if (response.toolCalls && response.toolCalls.length > 0) {
      const tc = response.toolCalls[0];
      try {
        const args = typeof tc.function.arguments === 'string'
          ? JSON.parse(tc.function.arguments)
          : tc.function.arguments;
        return (args as Record<string, unknown>).answer;
      } catch {
        return response.content;
      }
    }

    return response.content;
  }

  /**
   * Add a tool, which can also be an Agent instance (auto-wraps with AgentTool).
   */
  addTool(tool: Tool): void {
    super.addTool(tool);
  }
}
