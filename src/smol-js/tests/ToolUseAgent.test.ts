/**
 * Unit tests for ToolUseAgent
 */

import { describe, it, expect, vi } from 'vitest';
import { ToolUseAgent } from '../src/agents/ToolUseAgent.js';
import { createTool } from '../src/tools/Tool.js';
import { LogLevel } from '../src/types.js';

// Mock model that returns tool calls
function createMockModel(responses: Array<{
  content?: string;
  toolCalls?: Array<{
    id: string;
    name: string;
    arguments: Record<string, unknown>;
  }>;
}>) {
  let callIndex = 0;

  return {
    modelId: 'mock-model',
    generate: vi.fn(async () => {
      const response = responses[callIndex] ?? responses[responses.length - 1];
      callIndex++;

      return {
        role: 'assistant' as const,
        content: response.content ?? '',
        toolCalls: response.toolCalls?.map(tc => ({
          id: tc.id,
          type: 'function' as const,
          function: {
            name: tc.name,
            arguments: JSON.stringify(tc.arguments),
          },
        })),
        tokenUsage: { inputTokens: 100, outputTokens: 50, totalTokens: 150 },
      };
    }),
    supportsStreaming: () => false,
  };
}

describe('ToolUseAgent', () => {
  describe('initialization', () => {
    it('should create with tools', () => {
      const model = createMockModel([]);
      const tool = createTool({
        name: 'test',
        description: 'Test',
        inputs: {},
        outputType: 'string',
        execute: async () => 'ok',
      });

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [tool],
        name: 'TestAgent',
        verboseLevel: LogLevel.OFF,
      });

      expect(agent.getName()).toBe('TestAgent');
      expect(agent.getTools().has('test')).toBe(true);
      expect(agent.getTools().has('final_answer')).toBe(true);
    });
  });

  describe('run', () => {
    it('should execute tool calls and return final answer', async () => {
      const model = createMockModel([
        {
          content: 'Let me add those numbers.',
          toolCalls: [{ id: 'call_1', name: 'add', arguments: { a: 2, b: 3 } }],
        },
        {
          content: 'The result is 5.',
          toolCalls: [{ id: 'call_2', name: 'final_answer', arguments: { answer: 5 } }],
        },
      ]);

      const addTool = createTool({
        name: 'add',
        description: 'Add two numbers',
        inputs: {
          a: { type: 'number', description: 'First' },
          b: { type: 'number', description: 'Second' },
        },
        outputType: 'number',
        execute: async (args) => (args.a as number) + (args.b as number),
      });

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [addTool],
        maxSteps: 5,
        verboseLevel: LogLevel.OFF,
      });

      const result = await agent.run('Add 2 and 3');

      expect(result.output).toBe(5);
      expect(model.generate).toHaveBeenCalledTimes(2);
    });

    it('should handle tool errors gracefully', async () => {
      const model = createMockModel([
        {
          content: 'Let me try this.',
          toolCalls: [{ id: 'call_1', name: 'failing_tool', arguments: {} }],
        },
        {
          content: 'The tool failed, returning error info.',
          toolCalls: [{ id: 'call_2', name: 'final_answer', arguments: { answer: 'Error occurred' } }],
        },
      ]);

      const failingTool = createTool({
        name: 'failing_tool',
        description: 'A tool that fails',
        inputs: {},
        outputType: 'string',
        execute: async () => { throw new Error('Tool failed!'); },
      });

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [failingTool],
        maxSteps: 5,
        verboseLevel: LogLevel.OFF,
      });

      const result = await agent.run('Try the failing tool');

      expect(result.output).toBe('Error occurred');
    });

    it('should handle unknown tool calls', async () => {
      const model = createMockModel([
        {
          content: 'Using unknown tool.',
          toolCalls: [{ id: 'call_1', name: 'nonexistent', arguments: {} }],
        },
        {
          content: 'Falling back.',
          toolCalls: [{ id: 'call_2', name: 'final_answer', arguments: { answer: 'done' } }],
        },
      ]);

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [],
        maxSteps: 5,
        verboseLevel: LogLevel.OFF,
      });

      const result = await agent.run('Test unknown tool');

      expect(result.output).toBe('done');
    });

    it('should prompt for tool use when model responds without tools', async () => {
      const model = createMockModel([
        { content: 'I think the answer is...' },
        {
          content: 'Let me use the tool.',
          toolCalls: [{ id: 'call_1', name: 'final_answer', arguments: { answer: 42 } }],
        },
      ]);

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [],
        maxSteps: 5,
        verboseLevel: LogLevel.OFF,
      });

      const result = await agent.run('What is the answer?');

      expect(result.output).toBe(42);
    });

    it('should respect maxSteps', async () => {
      const model = createMockModel([
        {
          content: 'Still working...',
          toolCalls: [{ id: 'call_1', name: 'test', arguments: {} }],
        },
      ]);

      const testTool = createTool({
        name: 'test',
        description: 'Test',
        inputs: {},
        outputType: 'string',
        execute: async () => 'result',
      });

      const agent = new ToolUseAgent({
        model: model as any,
        tools: [testTool],
        maxSteps: 2,
        verboseLevel: LogLevel.OFF,
      });

      const result = await agent.run('Do something forever');

      // Should stop after maxSteps
      expect(result).toBeDefined();
    });
  });
});
