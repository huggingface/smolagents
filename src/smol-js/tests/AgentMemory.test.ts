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
    it('should convert memory to chat messages', () => {
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
