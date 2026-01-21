/**
 * LocalExecutor - JavaScript code execution engine using Node's vm module
 *
 * Executes JavaScript code chunks in an isolated context with:
 * - State persistence between steps (variables carry forward)
 * - Tool injection (tools available as async functions)
 * - Dynamic imports via CDN (esm.sh)
 * - Print capture and logging
 * - Safety timeouts
 */

import * as vm from 'vm';
import * as fs from 'fs';
import * as path from 'path';
import type { CodeExecutionOutput } from '../types.js';
import { Tool as ToolClass } from '../tools/Tool.js';

// Default timeout for code execution (30 seconds)
const DEFAULT_TIMEOUT_MS = 30000;

// Maximum length of captured output
const MAX_OUTPUT_LENGTH = 50000;

// CDN base URL for dynamic imports
const ESM_CDN = 'https://esm.sh';

export interface ExecutorConfig {
  /**
   * Maximum execution time in milliseconds
   * @default 30000
   */
  timeout?: number;

  /**
   * Additional authorized imports (npm packages to allow)
   */
  authorizedImports?: string[];

  /**
   * Whether to allow fs module access
   * @default true
   */
  allowFs?: boolean;

  /**
   * Working directory for fs operations
   */
  workingDirectory?: string;
}

export class LocalExecutor {
  private context: vm.Context;
  private state: Record<string, unknown> = {};
  private tools: Map<string, ToolClass> = new Map();
  private config: ExecutorConfig;
  private capturedLogs: string[] = [];

  constructor(config: ExecutorConfig = {}) {
    this.config = {
      timeout: DEFAULT_TIMEOUT_MS,
      allowFs: true,
      workingDirectory: process.cwd(),
      ...config,
    };

    this.context = this.createContext();
  }

  /**
   * Create the VM context with available globals.
   */
  private createContext(): vm.Context {
    // Create a proxy for console to capture logs
    const consoleProxy = {
      log: (...args: unknown[]) => {
        const output = args.map((arg) => this.stringify(arg)).join(' ');
        this.capturedLogs.push(output);
      },
      error: (...args: unknown[]) => {
        const output = args.map((arg) => this.stringify(arg)).join(' ');
        this.capturedLogs.push(`[ERROR] ${output}`);
      },
      warn: (...args: unknown[]) => {
        const output = args.map((arg) => this.stringify(arg)).join(' ');
        this.capturedLogs.push(`[WARN] ${output}`);
      },
      info: (...args: unknown[]) => {
        const output = args.map((arg) => this.stringify(arg)).join(' ');
        this.capturedLogs.push(output);
      },
      debug: (...args: unknown[]) => {
        const output = args.map((arg) => this.stringify(arg)).join(' ');
        this.capturedLogs.push(`[DEBUG] ${output}`);
      },
    };

    // Create print function (alias for console.log)
    const print = (...args: unknown[]) => consoleProxy.log(...args);

    // Create dynamic import function for npm packages
    const dynamicImport = async (packageName: string): Promise<unknown> => {
      // Check if it's an authorized import
      const authorized = this.config.authorizedImports ?? [];
      const basePackage = packageName.split('/')[0];

      if (!authorized.includes(basePackage) && !authorized.includes(packageName)) {
        throw new Error(
          `Import not authorized: ${packageName}. Add it to authorizedImports to allow.`
        );
      }

      try {
        // Try to import from CDN using dynamic import
        const cdnUrl = `${ESM_CDN}/${packageName}`;
        const module = await import(cdnUrl);
        return module.default ?? module;
      } catch (error) {
        throw new Error(`Failed to import ${packageName}: ${(error as Error).message}`);
      }
    };

    // Build context object
    const contextObj: Record<string, unknown> = {
      // Console and print
      console: consoleProxy,
      print,

      // Built-in objects
      Object,
      Array,
      String,
      Number,
      Boolean,
      Date,
      Math,
      JSON,
      RegExp,
      Error,
      Map,
      Set,
      WeakMap,
      WeakSet,
      Promise,
      Symbol,
      Proxy,
      Reflect,

      // Type checking
      parseInt,
      parseFloat,
      isNaN,
      isFinite,
      typeof: (v: unknown) => typeof v,

      // Timers (promisified for async support)
      setTimeout: global.setTimeout,
      clearTimeout: global.clearTimeout,
      setInterval: global.setInterval,
      clearInterval: global.clearInterval,

      // Async utilities
      fetch: global.fetch,

      // Dynamic import for npm packages
      importPackage: dynamicImport,

      // URL handling
      URL,
      URLSearchParams,

      // Text encoding
      TextEncoder,
      TextDecoder,

      // Buffer (useful for many operations)
      Buffer,

      // State reference (variables persist here)
      __state__: this.state,

      // Final answer marker
      __final_answer__: null as unknown,
      __is_final_answer__: false,
    };

    // Add fs module if allowed
    if (this.config.allowFs) {
      contextObj.fs = {
        readFileSync: (filePath: string, encoding?: BufferEncoding) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), filePath);
          return fs.readFileSync(resolvedPath, encoding ?? 'utf-8');
        },
        writeFileSync: (filePath: string, data: string | Buffer) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), filePath);
          return fs.writeFileSync(resolvedPath, data);
        },
        existsSync: (filePath: string) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), filePath);
          return fs.existsSync(resolvedPath);
        },
        readdirSync: (dirPath: string) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), dirPath);
          return fs.readdirSync(resolvedPath);
        },
        mkdirSync: (dirPath: string, options?: fs.MakeDirectoryOptions) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), dirPath);
          return fs.mkdirSync(resolvedPath, options);
        },
        unlinkSync: (filePath: string) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), filePath);
          return fs.unlinkSync(resolvedPath);
        },
        statSync: (filePath: string) => {
          const resolvedPath = path.resolve(this.config.workingDirectory ?? process.cwd(), filePath);
          return fs.statSync(resolvedPath);
        },
      };

      contextObj.path = {
        join: path.join,
        resolve: (...paths: string[]) =>
          path.resolve(this.config.workingDirectory ?? process.cwd(), ...paths),
        dirname: path.dirname,
        basename: path.basename,
        extname: path.extname,
      };
    }

    return vm.createContext(contextObj);
  }

  /**
   * Add tools to the executor context.
   */
  sendTools(tools: Record<string, ToolClass>): void {
    for (const [name, tool] of Object.entries(tools)) {
      this.tools.set(name, tool);

      // Add tool as async function in context
      this.context[name] = async (...args: unknown[]) => {
        // Handle both positional and named arguments
        let callArgs: Record<string, unknown>;

        if (args.length === 1 && typeof args[0] === 'object' && args[0] !== null) {
          // Called with named arguments object
          callArgs = args[0] as Record<string, unknown>;
        } else {
          // Called with positional arguments - map to input names
          const inputNames = Object.keys(tool.inputs);
          callArgs = {};
          args.forEach((arg, i) => {
            if (i < inputNames.length) {
              callArgs[inputNames[i]] = arg;
            }
          });
        }

        return tool.call(callArgs);
      };
    }
  }

  /**
   * Send variables to the executor state.
   */
  sendVariables(variables: Record<string, unknown>): void {
    Object.assign(this.state, variables);
    Object.assign(this.context, variables);
  }

  /**
   * Execute JavaScript code and return the result.
   */
  async execute(code: string): Promise<CodeExecutionOutput> {
    // Reset captured logs
    this.capturedLogs = [];

    // Reset final answer flag
    this.context.__is_final_answer__ = false;
    this.context.__final_answer__ = null;

    // Sync state to context
    Object.assign(this.context, this.state);

    // Wrap code to handle async and capture the last expression
    const wrappedCode = this.wrapCode(code);

    try {
      // Create and run the script
      const script = new vm.Script(wrappedCode, {
        filename: 'agent-code.js',
      });

      // Run with timeout
      const result = await script.runInContext(this.context, {
        timeout: this.config.timeout,
        displayErrors: true,
      });

      // Wait for the result if it's a promise
      const output = result instanceof Promise ? await result : result;

      // Update state with any new variables from context
      this.updateStateFromContext();

      // Check if final_answer was called
      const isFinalAnswer = this.context.__is_final_answer__ as boolean;
      const finalOutput = isFinalAnswer ? this.context.__final_answer__ : output;

      // Truncate logs if too long
      const logs = this.capturedLogs.join('\n').slice(0, MAX_OUTPUT_LENGTH);

      return {
        output: finalOutput,
        logs,
        isFinalAnswer,
      };
    } catch (error) {
      // Truncate logs if too long
      const logs = this.capturedLogs.join('\n').slice(0, MAX_OUTPUT_LENGTH);

      return {
        output: null,
        logs,
        isFinalAnswer: false,
        error: error as Error,
      };
    }
  }

  /**
   * Wrap code to handle async execution and final_answer calls.
   */
  private wrapCode(code: string): string {
    // Add final_answer function that sets the flag
    const finalAnswerFunc = `
      function final_answer(answer) {
        __is_final_answer__ = true;
        __final_answer__ = answer;
        return answer;
      }
    `;

    // Wrap in async IIFE to support await
    // Store result of last expression in __last_result__
    // We use a Function constructor approach to capture the last expression value
    return `
      ${finalAnswerFunc}
      (async () => {
        let __last_result__;
        ${this.instrumentCode(code)}
        return __last_result__;
      })()
    `;
  }

  /**
   * Instrument code to capture the last expression value and convert
   * let/const/var declarations to global assignments for state persistence.
   */
  private instrumentCode(code: string): string {
    // Split code into lines and find statements
    const lines = code.trim().split('\n');

    if (lines.length === 0) {
      return code;
    }

    // Process the code to capture expression results
    const processedLines = lines.map((line, index) => {
      const trimmed = line.trim();

      // Skip empty lines and comments
      if (!trimmed ||
          trimmed.startsWith('//') ||
          trimmed.startsWith('/*') ||
          trimmed.startsWith('*') ||
          trimmed.endsWith('*/')) {
        return line;
      }

      // Transform variable declarations to global assignments for persistence
      // This allows variables to persist across script executions
      // Handle multiple declarations on the same line using global replacement
      let transformed = line;
      let hasDeclaration = false;

      // Replace all let/const/var declarations with global assignments
      transformed = transformed.replace(
        /\b(let|const|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=/g,
        (_match, _keyword, varName) => {
          hasDeclaration = true;
          return `${varName} =`;
        }
      );

      // Handle declarations without initialization
      transformed = transformed.replace(
        /\b(let|const|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*(?=[;,]|$)/g,
        (_match, _keyword, varName) => {
          hasDeclaration = true;
          return `${varName} = undefined`;
        }
      );

      if (hasDeclaration) {
        return transformed;
      }

      // Skip control flow statements for last-result capture
      if (trimmed.startsWith('if') ||
          trimmed.startsWith('else') ||
          trimmed.startsWith('for') ||
          trimmed.startsWith('while') ||
          trimmed.startsWith('do') ||
          trimmed.startsWith('switch') ||
          trimmed.startsWith('case') ||
          trimmed.startsWith('default') ||
          trimmed.startsWith('try') ||
          trimmed.startsWith('catch') ||
          trimmed.startsWith('finally') ||
          trimmed.startsWith('return') ||
          trimmed.startsWith('throw') ||
          trimmed.startsWith('break') ||
          trimmed.startsWith('continue') ||
          trimmed.startsWith('function') ||
          trimmed.startsWith('class') ||
          trimmed.startsWith('import') ||
          trimmed.startsWith('export') ||
          trimmed === '{' ||
          trimmed === '}' ||
          trimmed.endsWith('{') ||
          trimmed.endsWith('}')) {
        return line;
      }

      // For the last meaningful line, try to capture the expression value
      if (index === lines.length - 1 || this.isLastMeaningfulLine(lines, index)) {
        // If line doesn't end with semicolon, try to capture it
        if (!trimmed.endsWith(';')) {
          return `__last_result__ = ${line}`;
        } else {
          // Remove semicolon, capture, and add it back
          const withoutSemi = trimmed.slice(0, -1);
          // Check if it looks like an expression (not ending with closing brace)
          if (!withoutSemi.endsWith('}') && !withoutSemi.endsWith(')')) {
            return `__last_result__ = ${withoutSemi};`;
          }
        }
      }

      return line;
    });

    return processedLines.join('\n');
  }

  /**
   * Check if this is the last meaningful line of code.
   */
  private isLastMeaningfulLine(lines: string[], currentIndex: number): boolean {
    for (let i = currentIndex + 1; i < lines.length; i++) {
      const trimmed = lines[i].trim();
      if (trimmed && !trimmed.startsWith('//') && !trimmed.startsWith('/*')) {
        return false;
      }
    }
    return true;
  }

  /**
   * Update internal state from context after execution.
   */
  private updateStateFromContext(): void {
    // List of keys to exclude from state (builtins and internals)
    const excludeKeys = new Set([
      'console',
      'print',
      'Object',
      'Array',
      'String',
      'Number',
      'Boolean',
      'Date',
      'Math',
      'JSON',
      'RegExp',
      'Error',
      'Map',
      'Set',
      'WeakMap',
      'WeakSet',
      'Promise',
      'Symbol',
      'Proxy',
      'Reflect',
      'parseInt',
      'parseFloat',
      'isNaN',
      'isFinite',
      'typeof',
      'setTimeout',
      'clearTimeout',
      'setInterval',
      'clearInterval',
      'fetch',
      'importPackage',
      'URL',
      'URLSearchParams',
      'TextEncoder',
      'TextDecoder',
      'Buffer',
      '__state__',
      '__final_answer__',
      '__is_final_answer__',
      'fs',
      'path',
      'final_answer',
      // Exclude tools
      ...this.tools.keys(),
    ]);

    // Copy non-builtin keys to state
    for (const key of Object.keys(this.context)) {
      if (!excludeKeys.has(key)) {
        this.state[key] = this.context[key];
      }
    }
  }

  /**
   * Stringify a value for logging.
   */
  private stringify(value: unknown): string {
    if (value === undefined) return 'undefined';
    if (value === null) return 'null';
    if (typeof value === 'string') return value;
    if (typeof value === 'function') return `[Function: ${value.name || 'anonymous'}]`;

    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  /**
   * Reset the executor state.
   */
  reset(): void {
    this.state = {};
    this.capturedLogs = [];
    this.context = this.createContext();

    // Re-add tools
    const tools = Object.fromEntries(this.tools);
    this.sendTools(tools);
  }

  /**
   * Get the current state.
   */
  getState(): Record<string, unknown> {
    return { ...this.state };
  }
}
