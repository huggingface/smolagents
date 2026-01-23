/**
 * Unit tests for YAMLLoader
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { YAMLLoader } from '../src/orchestrator/YAMLLoader.js';

// Mock environment for tests
process.env.OPENAI_API_KEY = 'test-key';

describe('YAMLLoader', () => {
  let loader: YAMLLoader;

  beforeEach(() => {
    loader = new YAMLLoader();
  });

  describe('loadFromString', () => {
    it('should load a simple workflow', () => {
      const yaml = `
name: test-workflow
description: A test workflow
model:
  modelId: gpt-4
  baseUrl: https://api.openai.com/v1
  apiKey: test-key
agents:
  main_agent:
    type: ToolUseAgent
    tools:
      - read_file
    maxSteps: 5
entrypoint: main_agent
`;

      const workflow = loader.loadFromString(yaml);

      expect(workflow.name).toBe('test-workflow');
      expect(workflow.description).toBe('A test workflow');
      expect(workflow.agents.has('main_agent')).toBe(true);
      expect(workflow.entrypointAgent).toBeDefined();
      expect(workflow.entrypointAgent.getName()).toBe('main_agent');
    });

    it('should load a workflow with multiple agents', () => {
      const yaml = `
name: multi-agent
model:
  modelId: gpt-4
  apiKey: test-key
tools:
  search:
    type: exa_search
  reader:
    type: read_file
agents:
  researcher:
    type: ToolUseAgent
    tools:
      - search
    maxSteps: 5
  writer:
    type: ToolUseAgent
    tools:
      - reader
    maxSteps: 3
  manager:
    type: ToolUseAgent
    agents:
      - researcher
      - writer
    maxSteps: 8
entrypoint: manager
`;

      const workflow = loader.loadFromString(yaml);

      expect(workflow.agents.size).toBe(3);
      expect(workflow.tools.size).toBe(2);
      expect(workflow.entrypointAgent.getName()).toBe('manager');
    });

    it('should load a workflow with CodeAgent', () => {
      const yaml = `
name: code-workflow
model:
  modelId: gpt-4
  apiKey: test-key
agents:
  coder:
    type: CodeAgent
    maxSteps: 10
    customInstructions: Use ES2020 syntax
entrypoint: coder
`;

      const workflow = loader.loadFromString(yaml);

      expect(workflow.agents.has('coder')).toBe(true);
    });

    it('should handle agent-level model config', () => {
      const yaml = `
name: model-test
model:
  modelId: gpt-4
  apiKey: test-key
agents:
  agent1:
    type: ToolUseAgent
    model:
      modelId: gpt-3.5-turbo
      apiKey: test-key
    maxSteps: 3
entrypoint: agent1
`;

      const workflow = loader.loadFromString(yaml);
      expect(workflow.agents.has('agent1')).toBe(true);
    });

    it('should throw on missing name', () => {
      const yaml = `
agents:
  agent:
    type: ToolUseAgent
entrypoint: agent
`;

      expect(() => loader.loadFromString(yaml)).toThrow('must have a name');
    });

    it('should throw on missing entrypoint', () => {
      const yaml = `
name: test
agents:
  agent:
    type: ToolUseAgent
`;

      expect(() => loader.loadFromString(yaml)).toThrow('must have an entrypoint');
    });

    it('should throw on unknown tool type', () => {
      const yaml = `
name: test
model:
  apiKey: test-key
tools:
  bad_tool:
    type: nonexistent_tool
agents:
  agent:
    type: ToolUseAgent
    tools:
      - bad_tool
entrypoint: agent
`;

      expect(() => loader.loadFromString(yaml)).toThrow('Unknown tool type');
    });

    it('should throw on invalid entrypoint reference', () => {
      const yaml = `
name: test
model:
  apiKey: test-key
agents:
  agent:
    type: ToolUseAgent
entrypoint: nonexistent
`;

      expect(() => loader.loadFromString(yaml)).toThrow('not found');
    });

    it('should handle global max context length', () => {
      const yaml = `
name: test
model:
  apiKey: test-key
agents:
  agent:
    type: ToolUseAgent
entrypoint: agent
globalMaxContextLength: 50000
`;

      const workflow = loader.loadFromString(yaml);
      expect(workflow).toBeDefined();
    });

    it('should support all built-in tool types', () => {
      const yaml = `
name: all-tools
model:
  apiKey: test-key
tools:
  search:
    type: exa_search
  contents:
    type: exa_get_contents
  research:
    type: exa_research
  read:
    type: read_file
  write:
    type: write_file
  http:
    type: curl
agents:
  agent:
    type: ToolUseAgent
    tools:
      - search
      - contents
      - research
      - read
      - write
      - http
entrypoint: agent
`;

      const workflow = loader.loadFromString(yaml);
      expect(workflow.tools.size).toBe(6);
    });
  });

  describe('loadFromFile', () => {
    it('should throw on non-existent file', () => {
      expect(() => loader.loadFromFile('/nonexistent/workflow.yaml')).toThrow('not found');
    });
  });

  describe('registerToolType', () => {
    it('should allow registering custom tools', () => {
      // This just tests that the registration method exists
      // Actual custom tool would need to implement Tool interface
      expect(typeof loader.registerToolType).toBe('function');
    });
  });
});
