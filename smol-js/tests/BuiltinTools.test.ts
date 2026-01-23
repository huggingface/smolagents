/**
 * Unit tests for built-in tools
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { ReadFileTool } from '../src/tools/ReadFileTool.js';
import { WriteFileTool } from '../src/tools/WriteFileTool.js';
import { CurlTool } from '../src/tools/CurlTool.js';

const TEST_DIR = path.join(os.tmpdir(), 'smol-js-test-' + Date.now());

describe('ReadFileTool', () => {
  beforeEach(() => {
    fs.mkdirSync(TEST_DIR, { recursive: true });
    fs.writeFileSync(path.join(TEST_DIR, 'test.txt'), 'Hello, World!');
  });

  afterEach(() => {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('should have correct metadata', () => {
    const tool = new ReadFileTool();
    expect(tool.name).toBe('read_file');
    expect(tool.inputs.path.type).toBe('string');
    expect(tool.outputType).toBe('string');
  });

  it('should read a file', async () => {
    const tool = new ReadFileTool({ workingDirectory: TEST_DIR });
    const content = await tool.call({ path: 'test.txt' });
    expect(content).toBe('Hello, World!');
  });

  it('should read with absolute path', async () => {
    const tool = new ReadFileTool();
    const content = await tool.call({ path: path.join(TEST_DIR, 'test.txt') });
    expect(content).toBe('Hello, World!');
  });

  it('should throw on non-existent file', async () => {
    const tool = new ReadFileTool();
    await expect(tool.call({ path: '/nonexistent/file.txt' })).rejects.toThrow('File not found');
  });

  it('should throw on directory path', async () => {
    const tool = new ReadFileTool();
    await expect(tool.call({ path: TEST_DIR })).rejects.toThrow('Path is a directory');
  });

  it('should generate valid OpenAI tool definition', () => {
    const tool = new ReadFileTool();
    const def = tool.toOpenAITool();
    expect(def.function.name).toBe('read_file');
    expect(def.function.parameters.required).toContain('path');
  });
});

describe('WriteFileTool', () => {
  beforeEach(() => {
    fs.mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it('should have correct metadata', () => {
    const tool = new WriteFileTool();
    expect(tool.name).toBe('write_file');
    expect(tool.inputs.content.type).toBe('string');
  });

  it('should write a file', async () => {
    const tool = new WriteFileTool({ workingDirectory: TEST_DIR });
    const result = await tool.call({ path: 'output.txt', content: 'Test content' });

    expect(result).toContain('Successfully wrote');
    const content = fs.readFileSync(path.join(TEST_DIR, 'output.txt'), 'utf-8');
    expect(content).toBe('Test content');
  });

  it('should append to a file', async () => {
    const tool = new WriteFileTool({ workingDirectory: TEST_DIR });
    fs.writeFileSync(path.join(TEST_DIR, 'append.txt'), 'First ');

    await tool.call({ path: 'append.txt', content: 'Second', append: true });

    const content = fs.readFileSync(path.join(TEST_DIR, 'append.txt'), 'utf-8');
    expect(content).toBe('First Second');
  });

  it('should create parent directories', async () => {
    const tool = new WriteFileTool({ workingDirectory: TEST_DIR });
    await tool.call({ path: 'sub/dir/file.txt', content: 'nested' });

    const content = fs.readFileSync(path.join(TEST_DIR, 'sub', 'dir', 'file.txt'), 'utf-8');
    expect(content).toBe('nested');
  });
});

describe('CurlTool', () => {
  it('should have correct metadata', () => {
    const tool = new CurlTool();
    expect(tool.name).toBe('curl');
    expect(tool.inputs.url.type).toBe('string');
    expect(tool.inputs.method.type).toBe('string');
  });

  it('should generate valid OpenAI tool definition', () => {
    const tool = new CurlTool();
    const def = tool.toOpenAITool();
    expect(def.function.name).toBe('curl');
    expect(def.function.parameters.required).toContain('url');
    expect(def.function.parameters.required).not.toContain('method');
  });
});
