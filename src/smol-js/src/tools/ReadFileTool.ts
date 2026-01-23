/**
 * ReadFileTool - Read contents from a file
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';
import * as fs from 'fs';
import * as path from 'path';

export class ReadFileTool extends Tool {
  readonly name = 'read_file';
  readonly description = 'Read the contents of a file at the specified path. Returns the file content as a string.';
  readonly inputs: ToolInputs = {
    path: {
      type: 'string',
      description: 'The file path to read (absolute or relative to working directory)',
      required: true,
    },
    encoding: {
      type: 'string',
      description: 'File encoding (default: utf-8)',
      required: false,
      default: 'utf-8',
    },
  };
  readonly outputType = 'string';

  private workingDirectory: string;

  constructor(config?: { workingDirectory?: string }) {
    super();
    this.workingDirectory = config?.workingDirectory ?? process.cwd();
  }

  async execute(args: Record<string, unknown>): Promise<string> {
    const filePath = args.path as string;
    const encoding = (args.encoding as BufferEncoding) ?? 'utf-8';

    const resolvedPath = path.isAbsolute(filePath)
      ? filePath
      : path.resolve(this.workingDirectory, filePath);

    if (!fs.existsSync(resolvedPath)) {
      throw new Error(`File not found: ${resolvedPath}`);
    }

    const stat = fs.statSync(resolvedPath);
    if (stat.isDirectory()) {
      throw new Error(`Path is a directory, not a file: ${resolvedPath}`);
    }

    const content = fs.readFileSync(resolvedPath, encoding);

    // Truncate very large files
    const maxLength = 100000;
    if (content.length > maxLength) {
      return content.slice(0, maxLength) + `\n\n[... truncated, file is ${content.length} characters total]`;
    }

    return content;
  }
}
