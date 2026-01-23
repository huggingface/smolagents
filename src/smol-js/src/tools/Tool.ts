/**
 * Tool base class for smol-js
 *
 * Tools are the primary way for agents to interact with the outside world.
 * Extend this class and implement the execute() method to create custom tools.
 */

import type { ToolInputs, ToolInputType, OpenAIToolDefinition } from '../types.js';

export abstract class Tool {
  /**
   * Unique identifier for the tool
   */
  abstract readonly name: string;

  /**
   * Human-readable description of what the tool does
   */
  abstract readonly description: string;

  /**
   * Input parameter schema
   */
  abstract readonly inputs: ToolInputs;

  /**
   * Output type description
   */
  abstract readonly outputType: string;

  /**
   * Whether the tool has been set up
   */
  protected isSetup: boolean = false;

  /**
   * Optional setup method called before first use.
   * Override this for expensive initialization (loading models, etc.)
   */
  async setup(): Promise<void> {
    this.isSetup = true;
  }

  /**
   * Execute the tool with the given arguments.
   * Must be implemented by subclasses.
   */
  abstract execute(args: Record<string, unknown>): Promise<unknown>;

  /**
   * Call the tool, ensuring setup is complete and validating arguments.
   */
  async call(args: Record<string, unknown>): Promise<unknown> {
    if (!this.isSetup) {
      await this.setup();
    }

    // Validate arguments
    this.validateArguments(args);

    // Execute and return result
    return this.execute(args);
  }

  /**
   * Validate that provided arguments match the input schema.
   */
  protected validateArguments(args: Record<string, unknown>): void {
    const providedKeys = new Set(Object.keys(args));

    for (const [key, input] of Object.entries(this.inputs)) {
      // Check required arguments
      if (input.required !== false && !providedKeys.has(key)) {
        throw new Error(`Missing required argument: ${key}`);
      }

      // Check type if argument is provided
      if (providedKeys.has(key) && args[key] !== undefined && args[key] !== null) {
        const value = args[key];
        if (!this.checkType(value, input.type)) {
          throw new Error(
            `Argument '${key}' has invalid type. Expected ${input.type}, got ${typeof value}`
          );
        }
      }

      providedKeys.delete(key);
    }
  }

  /**
   * Check if a value matches the expected type.
   */
  protected checkType(value: unknown, expectedType: ToolInputType): boolean {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number';
      case 'boolean':
        return typeof value === 'boolean';
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && value !== null && !Array.isArray(value);
      case 'any':
        return true;
      default:
        return false;
    }
  }

  /**
   * Generate a code-friendly prompt representation of this tool.
   * Used in the CodeAgent system prompt.
   */
  toCodePrompt(): string {
    const argsSignature = Object.entries(this.inputs)
      .map(([name, input]) => {
        const optional = input.required === false ? '?' : '';
        return `${name}${optional}: ${this.typeToJsType(input.type)}`;
      })
      .join(', ');

    const argsDoc = Object.entries(this.inputs)
      .map(([name, input]) => `   * @param ${name} - ${input.description}`)
      .join('\n');

    return `
/**
 * ${this.description}
 *
${argsDoc}
 * @returns ${this.outputType}
 */
async function ${this.name}(${argsSignature}): Promise<${this.typeToJsType(this.outputType as ToolInputType)}> { ... }
`.trim();
  }

  /**
   * Generate an OpenAI-compatible tool definition for function calling.
   */
  toOpenAITool(): OpenAIToolDefinition {
    const properties: Record<string, unknown> = {};
    const required: string[] = [];

    for (const [key, input] of Object.entries(this.inputs)) {
      const prop: Record<string, unknown> = {
        type: this.typeToJsonSchemaType(input.type),
        description: input.description,
      };

      if (input.enum) {
        prop.enum = input.enum;
      }

      if (input.default !== undefined) {
        prop.default = input.default;
      }

      properties[key] = prop;

      if (input.required !== false) {
        required.push(key);
      }
    }

    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: {
          type: 'object',
          properties,
          ...(required.length > 0 && { required }),
        },
      },
    };
  }

  /**
   * Convert tool input type to JSON Schema type.
   */
  protected typeToJsonSchemaType(type: ToolInputType): string {
    switch (type) {
      case 'string':
        return 'string';
      case 'number':
        return 'number';
      case 'boolean':
        return 'boolean';
      case 'array':
        return 'array';
      case 'object':
        return 'object';
      case 'any':
        return 'string';
      default:
        return 'string';
    }
  }

  /**
   * Convert tool input type to JS/TS type string.
   */
  protected typeToJsType(type: ToolInputType | string): string {
    switch (type) {
      case 'string':
        return 'string';
      case 'number':
        return 'number';
      case 'boolean':
        return 'boolean';
      case 'array':
        return 'any[]';
      case 'object':
        return 'Record<string, any>';
      case 'any':
        return 'any';
      default:
        return type;
    }
  }

  /**
   * Serialize the tool to a JSON-compatible object.
   */
  toJSON(): Record<string, unknown> {
    return {
      name: this.name,
      description: this.description,
      inputs: this.inputs,
      outputType: this.outputType,
    };
  }
}

/**
 * Helper function to create a tool from a function.
 * This is an alternative to extending the Tool class.
 */
export function createTool(config: {
  name: string;
  description: string;
  inputs: ToolInputs;
  outputType: string;
  execute: (args: Record<string, unknown>) => Promise<unknown>;
}): Tool {
  return new (class extends Tool {
    readonly name = config.name;
    readonly description = config.description;
    readonly inputs = config.inputs;
    readonly outputType = config.outputType;

    async execute(args: Record<string, unknown>): Promise<unknown> {
      return config.execute(args);
    }
  })();
}
