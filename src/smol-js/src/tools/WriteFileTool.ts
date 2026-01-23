/**
 * WriteFileTool - Write content to a file
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';
import * as fs from 'fs';
import * as path from 'path';

export class WriteFileTool extends Tool {
  readonly name = 'write_file';
  readonly description = 'Write content to a file at the specified path. Creates the file if it does not exist, and creates parent directories as needed. Overwrites existing content by default.';
  readonly inputs: ToolInputs = {
    path: {
      type: 'string',
      description: 'The file path to write to (absolute or relative to working directory)',
      required: true,
    },
    content: {
      type: 'string',
      description: 'The content to write to the file',
      required: true,
    },
    append: {
      type: 'boolean',
      description: 'If true, append to the file instead of overwriting (default: false)',
      required: false,
      default: false,
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
    const content = args.content as string;
    const append = (args.append as boolean) ?? false;

    const resolvedPath = path.isAbsolute(filePath)
      ? filePath
      : path.resolve(this.workingDirectory, filePath);

    // Create parent directories if needed
    const dir = path.dirname(resolvedPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    if (append) {
      fs.appendFileSync(resolvedPath, content, 'utf-8');
      return `Successfully appended ${content.length} characters to ${resolvedPath}`;
    } else {
      fs.writeFileSync(resolvedPath, content, 'utf-8');
      return `Successfully wrote ${content.length} characters to ${resolvedPath}`;
    }
  }
}
