/**
 * Unit tests for Tool class
 */

import { describe, it, expect } from 'vitest';
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

  describe('toOpenAITool', () => {
    it('should generate valid OpenAI tool definition', () => {
      const tool = new TestTool();

      const def = tool.toOpenAITool();

      expect(def.type).toBe('function');
      expect(def.function.name).toBe('test_tool');
      expect(def.function.description).toBe('A test tool for unit testing');
      expect(def.function.parameters.type).toBe('object');
      expect(def.function.parameters.properties).toHaveProperty('message');
      expect(def.function.parameters.properties).toHaveProperty('count');
      expect(def.function.parameters.required).toContain('message');
      expect(def.function.parameters.required).not.toContain('count');
    });

    it('should include enum values', () => {
      class EnumTool extends Tool {
        readonly name = 'enum_tool';
        readonly description = 'Tool with enum';
        readonly inputs: ToolInputs = {
          mode: {
            type: 'string',
            description: 'Mode',
            enum: ['fast', 'slow'],
          },
        };
        readonly outputType = 'string';
        async execute(): Promise<string> { return ''; }
      }

      const tool = new EnumTool();
      const def = tool.toOpenAITool();

      expect((def.function.parameters.properties.mode as Record<string, unknown>).enum).toEqual(['fast', 'slow']);
    });

    it('should map types to JSON Schema types', () => {
      class TypesTool extends Tool {
        readonly name = 'types_tool';
        readonly description = 'Test types';
        readonly inputs: ToolInputs = {
          str: { type: 'string', description: 's' },
          num: { type: 'number', description: 'n' },
          bool: { type: 'boolean', description: 'b' },
          arr: { type: 'array', description: 'a' },
          obj: { type: 'object', description: 'o' },
          any: { type: 'any', description: 'x' },
        };
        readonly outputType = 'string';
        async execute(): Promise<string> { return ''; }
      }

      const tool = new TypesTool();
      const def = tool.toOpenAITool();
      const props = def.function.parameters.properties;

      expect((props.str as Record<string, unknown>).type).toBe('string');
      expect((props.num as Record<string, unknown>).type).toBe('number');
      expect((props.bool as Record<string, unknown>).type).toBe('boolean');
      expect((props.arr as Record<string, unknown>).type).toBe('array');
      expect((props.obj as Record<string, unknown>).type).toBe('object');
      expect((props.any as Record<string, unknown>).type).toBe('string'); // 'any' maps to 'string' in JSON Schema
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

  it('should generate OpenAI tool definition from createTool', () => {
    const tool = createTool({
      name: 'test',
      description: 'Test tool',
      inputs: { x: { type: 'string', description: 'Input' } },
      outputType: 'string',
      execute: async () => 'ok',
    });

    const def = tool.toOpenAITool();
    expect(def.function.name).toBe('test');
    expect(def.function.parameters.required).toEqual(['x']);
  });
});
