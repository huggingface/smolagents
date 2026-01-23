/**
 * Default tools provided to all agents
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';

/**
 * FinalAnswerTool - Used by the agent to return the final answer.
 * This is always available to CodeAgent.
 */
export class FinalAnswerTool extends Tool {
  readonly name = 'final_answer';
  readonly description = 'Returns the final answer to the user query. Use this when you have completed the task and have the final result.';
  readonly inputs: ToolInputs = {
    answer: {
      type: 'any',
      description: 'The final answer to return. Can be any type (string, number, object, etc.)',
      required: true,
    },
  };
  readonly outputType = 'any';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    return args.answer;
  }
}

/**
 * UserInputTool - Allows the agent to ask the user for input.
 */
export class UserInputTool extends Tool {
  readonly name = 'user_input';
  readonly description = 'Asks the user for additional input or clarification.';
  readonly inputs: ToolInputs = {
    question: {
      type: 'string',
      description: 'The question to ask the user',
      required: true,
    },
  };
  readonly outputType = 'string';

  private inputHandler?: (question: string) => Promise<string>;

  constructor(inputHandler?: (question: string) => Promise<string>) {
    super();
    this.inputHandler = inputHandler;
  }

  async execute(args: Record<string, unknown>): Promise<string> {
    const question = args.question as string;

    if (this.inputHandler) {
      return this.inputHandler(question);
    }

    // Default: use readline for terminal input
    const readline = await import('readline');
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });

    return new Promise((resolve) => {
      rl.question(`\n[Agent asks]: ${question}\nYour response: `, (answer) => {
        rl.close();
        resolve(answer);
      });
    });
  }
}

// Export singleton instances
export const finalAnswerTool = new FinalAnswerTool();
