/**
 * Unit tests for AgentMemory
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { AgentMemory } from '../src/memory/AgentMemory.js';

describe('AgentMemory', () => {
  let memory: AgentMemory;

  beforeEach(() => {
    memory = new AgentMemory('You are a helpful assistant.');
  });

  describe('initialization', () => {
    it('should store system prompt', () => {
      expect(memory.systemPrompt.content).toBe('You are a helpful assistant.');
      expect(memory.systemPrompt.type).toBe('system');
    });

    it('should start with empty steps', () => {
      expect(memory.steps).toHaveLength(0);
    });

    it('should accept config options', () => {
      const mem = new AgentMemory('test', {
        maxContextLength: 50000,
        memoryStrategy: 'compact',
      });
      expect(mem.systemPrompt.content).toBe('test');
    });
  });

  describe('addTask', () => {
    it('should add a task step', () => {
      const step = memory.addTask('Calculate 2 + 2');

      expect(step.type).toBe('task');
      expect(step.task).toBe('Calculate 2 + 2');
      expect(memory.steps).toHaveLength(1);
    });
  });

  describe('createActionStep', () => {
    it('should create an action step with step number', () => {
      const step = memory.createActionStep(1);

      expect(step.type).toBe('action');
      expect(step.stepNumber).toBe(1);
      expect(step.timing.startTime).toBeDefined();
      expect(memory.steps).toHaveLength(1);
    });
  });

  describe('addFinalAnswer', () => {
    it('should add a final answer step', () => {
      const step = memory.addFinalAnswer('The answer is 42');

      expect(step.type).toBe('final');
      expect(step.answer).toBe('The answer is 42');
    });

    it('should handle complex answer objects', () => {
      const answer = { result: 42, items: [1, 2, 3] };
      const step = memory.addFinalAnswer(answer);

      expect(step.answer).toEqual(answer);
    });
  });

  describe('toMessages', () => {
    it('should convert memory to chat messages (CodeAgent pattern)', () => {
      memory.addTask('Test task');
      const actionStep = memory.createActionStep(1);
      actionStep.modelOutputMessage = {
        role: 'assistant',
        content: 'Thinking...\n```javascript\nconst x = 1;\n```',
      };
      actionStep.observation = 'Observation: Output: 1';

      const messages = memory.toMessages();

      expect(messages).toHaveLength(4); // system, task, assistant, observation
      expect(messages[0].role).toBe('system');
      expect(messages[1].role).toBe('user');
      expect(messages[1].content).toContain('Test task');
      expect(messages[2].role).toBe('assistant');
      expect(messages[3].role).toBe('user');
      expect(messages[3].content).toContain('Observation');
    });

    it('should convert memory to chat messages (ToolUseAgent pattern)', () => {
      memory.addTask('Test task');
      const actionStep = memory.createActionStep(1);
      actionStep.modelOutputMessage = {
        role: 'assistant',
        content: 'Let me search for that.',
      };
      actionStep.toolCalls = [{
        id: 'call_123',
        type: 'function',
        function: {
          name: 'web_search',
          arguments: '{"query":"test"}',
        },
      }];
      actionStep.toolResults = [{
        toolCallId: 'call_123',
        toolName: 'web_search',
        result: 'Search results here',
      }];

      const messages = memory.toMessages();

      expect(messages).toHaveLength(4); // system, task, assistant+toolCalls, tool result
      expect(messages[2].role).toBe('assistant');
      expect(messages[2].toolCalls).toHaveLength(1);
      expect(messages[2].toolCalls![0].function.name).toBe('web_search');
      expect(messages[3].role).toBe('tool');
      expect(messages[3].toolCallId).toBe('call_123');
      expect(messages[3].content).toBe('Search results here');
    });

    it('should include error messages', () => {
      memory.addTask('Test');
      const actionStep = memory.createActionStep(1);
      actionStep.error = new Error('Something went wrong');

      const messages = memory.toMessages();

      const errorMessage = messages.find((m) => m.content?.includes('Error:'));
      expect(errorMessage).toBeDefined();
      expect(errorMessage?.content).toContain('Something went wrong');
    });
  });

  describe('manageContext', () => {
    it('should not truncate when within limit', async () => {
      const mem = new AgentMemory('Short prompt', { maxContextLength: 100000 });
      mem.addTask('Short task');
      mem.createActionStep(1);

      await mem.manageContext();

      expect(mem.steps).toHaveLength(2);
    });

    it('should truncate older steps when exceeding context length', async () => {
      const mem = new AgentMemory('System prompt', {
        maxContextLength: 100, // Very low limit
        memoryStrategy: 'truncate',
      });
      mem.addTask('Task');

      // Add many action steps with long content
      for (let i = 0; i < 10; i++) {
        const step = mem.createActionStep(i + 1);
        step.modelOutputMessage = {
          role: 'assistant',
          content: 'A'.repeat(200),
        };
        step.observation = 'B'.repeat(200);
      }

      await mem.manageContext();

      // Some steps should have been removed
      const actionSteps = mem.getActionSteps();
      expect(actionSteps.length).toBeLessThan(10);
    });
  });

  describe('reset', () => {
    it('should clear steps but keep system prompt', () => {
      memory.addTask('Task 1');
      memory.createActionStep(1);
      memory.addFinalAnswer('Answer');

      expect(memory.steps.length).toBeGreaterThan(0);

      memory.reset();

      expect(memory.steps).toHaveLength(0);
      expect(memory.systemPrompt.content).toBe('You are a helpful assistant.');
    });
  });

  describe('getActionSteps', () => {
    it('should return only action steps', () => {
      memory.addTask('Task');
      memory.createActionStep(1);
      memory.createActionStep(2);
      memory.addFinalAnswer('Answer');

      const actionSteps = memory.getActionSteps();

      expect(actionSteps).toHaveLength(2);
      expect(actionSteps[0].stepNumber).toBe(1);
      expect(actionSteps[1].stepNumber).toBe(2);
    });
  });

  describe('getTotalTokenUsage', () => {
    it('should sum token usage across steps', () => {
      const step1 = memory.createActionStep(1);
      step1.tokenUsage = { inputTokens: 100, outputTokens: 50, totalTokens: 150 };

      const step2 = memory.createActionStep(2);
      step2.tokenUsage = { inputTokens: 200, outputTokens: 100, totalTokens: 300 };

      const total = memory.getTotalTokenUsage();

      expect(total.inputTokens).toBe(300);
      expect(total.outputTokens).toBe(150);
      expect(total.totalTokens).toBe(450);
    });

    it('should return zeros if no token usage', () => {
      memory.createActionStep(1);

      const total = memory.getTotalTokenUsage();

      expect(total.inputTokens).toBe(0);
      expect(total.outputTokens).toBe(0);
      expect(total.totalTokens).toBe(0);
    });
  });

  describe('getEstimatedTokenCount', () => {
    it('should return an estimated token count', () => {
      memory.addTask('This is a test task');
      const tokenCount = memory.getEstimatedTokenCount();
      expect(tokenCount).toBeGreaterThan(0);
    });
  });

  describe('getLastStep', () => {
    it('should return the last step', () => {
      memory.addTask('Task');
      memory.createActionStep(1);
      memory.createActionStep(2);

      const lastStep = memory.getLastStep();

      expect(lastStep?.type).toBe('action');
      expect((lastStep as { stepNumber: number }).stepNumber).toBe(2);
    });

    it('should return undefined if no steps', () => {
      expect(memory.getLastStep()).toBeUndefined();
    });
  });

  describe('toJSON', () => {
    it('should serialize memory', () => {
      memory.addTask('Test task');
      memory.createActionStep(1);

      const json = memory.toJSON();

      expect(json.systemPrompt).toBeDefined();
      expect(json.steps).toHaveLength(2);
    });
  });
});
