/**
 * Unit tests for OpenAIModel
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenAIModel } from '../src/models/OpenAIModel.js';

// Mock the OpenAI client
vi.mock('openai', () => {
  return {
    default: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    })),
  };
});

describe('OpenAIModel', () => {
  beforeEach(() => {
    // Set environment variable for tests
    process.env.OPENAI_API_KEY = 'test-api-key';
    vi.clearAllMocks();
  });

  describe('initialization', () => {
    it('should create model with default config', () => {
      const model = new OpenAIModel();

      expect(model.modelId).toBe('anthropic/claude-sonnet-4.5');
    });

    it('should accept custom model ID', () => {
      const model = new OpenAIModel({ modelId: 'gpt-4' });

      expect(model.modelId).toBe('gpt-4');
    });

    it('should throw if no API key', () => {
      delete process.env.OPENAI_API_KEY;
      delete process.env.OPENROUTER_API_KEY;

      expect(() => new OpenAIModel()).toThrow('API key is required');
    });

    it('should accept API key from config', () => {
      delete process.env.OPENAI_API_KEY;

      const model = new OpenAIModel({ apiKey: 'config-api-key' });

      expect(model.modelId).toBeDefined();
    });
  });

  describe('supportsStreaming', () => {
    it('should support streaming', () => {
      const model = new OpenAIModel();

      expect(model.supportsStreaming()).toBe(true);
    });
  });

  describe('formatMessages', () => {
    it('should format standard messages', () => {
      const model = new OpenAIModel();

      // Access protected method through type assertion
      const formatted = (model as unknown as { formatMessages: typeof model['formatMessages'] }).formatMessages([
        { role: 'system', content: 'You are helpful' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
      ]);

      expect(formatted).toEqual([
        { role: 'system', content: 'You are helpful' },
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: 'Hi there' },
      ]);
    });

    it('should convert tool role to user', () => {
      const model = new OpenAIModel();

      const formatted = (model as unknown as { formatMessages: typeof model['formatMessages'] }).formatMessages([
        { role: 'tool', content: 'Tool output' },
      ]);

      expect(formatted[0].role).toBe('user');
    });
  });
});
