/**
 * Unit tests for LocalExecutor
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LocalExecutor } from '../src/executor/LocalExecutor.js';
import { createTool } from '../src/tools/Tool.js';

describe('LocalExecutor', () => {
  let executor: LocalExecutor;

  beforeEach(() => {
    executor = new LocalExecutor({
      timeout: 5000,
      authorizedImports: ['lodash'],
    });
  });

  describe('basic execution', () => {
    it('should execute simple code', async () => {
      const result = await executor.execute('1 + 1');

      expect(result.error).toBeUndefined();
      expect(result.output).toBe(2);
      expect(result.isFinalAnswer).toBe(false);
    });

    it('should capture console.log output', async () => {
      const result = await executor.execute('console.log("hello"); console.log("world");');

      expect(result.logs).toContain('hello');
      expect(result.logs).toContain('world');
    });

    it('should capture print output', async () => {
      const result = await executor.execute('print("test message")');

      expect(result.logs).toContain('test message');
    });

    it('should handle async code', async () => {
      const result = await executor.execute(`
        const delay = (ms) => new Promise(r => setTimeout(r, ms));
        await delay(10);
        "done"
      `);

      expect(result.output).toBe('done');
    });
  });

  describe('state persistence', () => {
    it('should persist variables between executions', async () => {
      await executor.execute('let counter = 0');
      await executor.execute('counter++');
      const result = await executor.execute('counter');

      expect(result.output).toBe(1);
    });

    it('should persist complex objects', async () => {
      await executor.execute('const data = { items: [] }');
      await executor.execute('data.items.push("a", "b", "c")');
      const result = await executor.execute('data.items.length');

      expect(result.output).toBe(3);
    });

    it('should reset state on reset()', async () => {
      await executor.execute('let x = 42');
      executor.reset();

      const result = await executor.execute('typeof x');

      expect(result.output).toBe('undefined');
    });
  });

  describe('final_answer', () => {
    it('should detect final_answer call', async () => {
      const result = await executor.execute('final_answer("the answer")');

      expect(result.isFinalAnswer).toBe(true);
      expect(result.output).toBe('the answer');
    });

    it('should pass complex objects to final_answer', async () => {
      const result = await executor.execute('final_answer({ result: 42, items: [1, 2, 3] })');

      expect(result.isFinalAnswer).toBe(true);
      expect(result.output).toEqual({ result: 42, items: [1, 2, 3] });
    });
  });

  describe('error handling', () => {
    it('should capture syntax errors', async () => {
      const result = await executor.execute('const x = {');

      expect(result.error).toBeDefined();
      expect(result.error?.message).toContain('Unexpected');
    });

    it('should capture runtime errors', async () => {
      const result = await executor.execute('throw new Error("test error")');

      expect(result.error).toBeDefined();
      expect(result.error?.message).toContain('test error');
    });

    it('should capture reference errors', async () => {
      const result = await executor.execute('nonexistentVariable.property');

      expect(result.error).toBeDefined();
    });
  });

  describe('tools', () => {
    it('should make tools available as functions', async () => {
      const tool = createTool({
        name: 'multiply',
        description: 'Multiplies two numbers',
        inputs: {
          a: { type: 'number', description: 'First number' },
          b: { type: 'number', description: 'Second number' },
        },
        outputType: 'number',
        execute: async (args) => (args.a as number) * (args.b as number),
      });

      executor.sendTools({ multiply: tool });

      const result = await executor.execute('await multiply({ a: 6, b: 7 })');

      expect(result.output).toBe(42);
    });

    it('should support positional arguments for tools', async () => {
      const tool = createTool({
        name: 'greet',
        description: 'Greets a person',
        inputs: {
          name: { type: 'string', description: 'Name to greet' },
        },
        outputType: 'string',
        execute: async (args) => `Hello, ${args.name}!`,
      });

      executor.sendTools({ greet: tool });

      const result = await executor.execute('await greet("World")');

      expect(result.output).toBe('Hello, World!');
    });
  });

  describe('built-in capabilities', () => {
    it('should have access to Math', async () => {
      const result = await executor.execute('Math.sqrt(16)');

      expect(result.output).toBe(4);
    });

    it('should have access to JSON', async () => {
      const result = await executor.execute('JSON.parse(\'{"a": 1}\')');

      expect(result.output).toEqual({ a: 1 });
    });

    it('should have access to Array methods', async () => {
      const result = await executor.execute('[1, 2, 3].map(x => x * 2)');

      expect(result.output).toEqual([2, 4, 6]);
    });

    it('should have access to Promise', async () => {
      const result = await executor.execute(`
        await Promise.resolve(42)
      `);

      expect(result.output).toBe(42);
    });
  });

  describe('sendVariables', () => {
    it('should inject variables into context', async () => {
      executor.sendVariables({ myValue: 123, myString: 'test' });

      const result = await executor.execute('myValue + myString.length');

      expect(result.output).toBe(127);
    });
  });

  describe('getState', () => {
    it('should return current state', async () => {
      await executor.execute('const x = 1; const y = 2;');

      const state = executor.getState();

      expect(state.x).toBe(1);
      expect(state.y).toBe(2);
    });
  });
});
