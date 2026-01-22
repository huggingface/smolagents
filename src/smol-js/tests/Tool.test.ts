/**
 * Unit tests for Tool class
 */

import { describe, it, expect, vi } from 'vitest';
import { Tool, createTool } from '../src/tools/Tool.js';
import type { ToolInputs } from '../src/types.js';

// Test tool implementation
class TestTool extends Tool {
  readonly name = 'test_tool';
  readonly description = 'A test tool for unit testing';
  readonly inputs: ToolInputs = {
    message: {
      type: 'string',
      description: 'The message to process',
      required: true,
    },
    count: {
      type: 'number',
      description: 'Number of times to repeat',
      required: false,
    },
  };
  readonly outputType = 'string';

  async execute(args: Record<string, unknown>): Promise<string> {
    const message = args.message as string;
    const count = (args.count as number) ?? 1;
    return message.repeat(count);
  }
}

describe('Tool', () => {
  describe('basic functionality', () => {
    it('should create a tool with required properties', () => {
      const tool = new TestTool();

      expect(tool.name).toBe('test_tool');
      expect(tool.description).toBe('A test tool for unit testing');
      expect(tool.inputs).toHaveProperty('message');
      expect(tool.outputType).toBe('string');
    });

    it('should execute with valid arguments', async () => {
      const tool = new TestTool();

      const result = await tool.call({ message: 'hello' });

      expect(result).toBe('hello');
    });

    it('should execute with optional arguments', async () => {
      const tool = new TestTool();

      const result = await tool.call({ message: 'hi', count: 3 });

      expect(result).toBe('hihihi');
    });

    it('should throw on missing required arguments', async () => {
      const tool = new TestTool();

      await expect(tool.call({})).rejects.toThrow('Missing required argument: message');
    });

    it('should throw on invalid argument type', async () => {
      const tool = new TestTool();

      await expect(tool.call({ message: 123 })).rejects.toThrow(
        "Argument 'message' has invalid type"
      );
    });
  });

  describe('setup lifecycle', () => {
    it('should call setup before first execution', async () => {
      class SetupTool extends Tool {
        readonly name = 'setup_tool';
        readonly description = 'Test setup';
        readonly inputs: ToolInputs = {};
        readonly outputType = 'boolean';

        setupCalled = false;

        async setup(): Promise<void> {
          await super.setup();
          this.setupCalled = true;
        }

        async execute(): Promise<boolean> {
          return this.setupCalled;
        }
      }

      const tool = new SetupTool();
      expect(tool.setupCalled).toBe(false);

      const result = await tool.call({});

      expect(result).toBe(true);
      expect(tool.setupCalled).toBe(true);
    });
  });

  describe('toCodePrompt', () => {
    it('should generate valid code prompt', () => {
      const tool = new TestTool();

      const prompt = tool.toCodePrompt();

      expect(prompt).toContain('function test_tool');
      expect(prompt).toContain('message: string');
      expect(prompt).toContain('count?: number');
      expect(prompt).toContain('A test tool for unit testing');
      expect(prompt).toContain('@param message');
      expect(prompt).toContain('@returns string');
    });
  });

  describe('toJSON', () => {
    it('should serialize tool to JSON', () => {
      const tool = new TestTool();

      const json = tool.toJSON();

      expect(json).toEqual({
        name: 'test_tool',
        description: 'A test tool for unit testing',
        inputs: tool.inputs,
        outputType: 'string',
      });
    });
  });
});

describe('createTool', () => {
  it('should create a tool from configuration', async () => {
    const tool = createTool({
      name: 'add_numbers',
      description: 'Adds two numbers',
      inputs: {
        a: { type: 'number', description: 'First number' },
        b: { type: 'number', description: 'Second number' },
      },
      outputType: 'number',
      execute: async (args) => (args.a as number) + (args.b as number),
    });

    expect(tool.name).toBe('add_numbers');

    const result = await tool.call({ a: 2, b: 3 });
    expect(result).toBe(5);
  });
});
