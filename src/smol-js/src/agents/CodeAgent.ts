/**
 * CodeAgent - Executes tasks by generating and running JavaScript code
 *
 * This is the main agent implementation for smol-js. It follows the ReAct pattern:
 * 1. Receives a task
 * 2. Generates reasoning and code
 * 3. Executes the code in a sandboxed environment
 * 4. Observes the result and continues or returns final answer
 */

import { Agent, AgentConfig } from './Agent.js';
import { LocalExecutor, ExecutorConfig } from '../executor/LocalExecutor.js';
import { FinalAnswerTool } from '../tools/defaultTools.js';
import { Tool } from '../tools/Tool.js';
import { generateSystemPrompt, getErrorRecoveryPrompt } from '../prompts/codeAgent.js';
import type { ActionStep, ActionOutput, ChatMessage } from '../types.js';

export interface CodeAgentConfig extends AgentConfig {
  /**
   * Additional npm packages that can be imported dynamically
   */
  additionalAuthorizedImports?: string[];

  /**
   * Executor configuration
   */
  executorConfig?: ExecutorConfig;

  /**
   * Working directory for file operations
   */
  workingDirectory?: string;
}

// Regex patterns for extracting code from LLM output
const CODE_BLOCK_REGEX = /```(?:javascript|js)?\n([\s\S]*?)```/;
const THOUGHT_REGEX = /(?:Thought|Reasoning):\s*([\s\S]*?)(?=```|$)/i;

export class CodeAgent extends Agent {
  /**
   * The JavaScript code executor
   */
  private executor: LocalExecutor;

  /**
   * Authorized imports for dynamic npm package loading
   */
  private authorizedImports: string[];

  constructor(config: CodeAgentConfig) {
    super(config);

    // Set authorized imports
    this.authorizedImports = config.additionalAuthorizedImports ?? [];

    // Initialize executor
    this.executor = new LocalExecutor({
      ...config.executorConfig,
      authorizedImports: this.authorizedImports,
      workingDirectory: config.workingDirectory,
    });

    // Always add final_answer tool
    if (!this.tools.has('final_answer')) {
      this.tools.set('final_answer', new FinalAnswerTool());
    }

    // Send tools to executor
    this.executor.sendTools(Object.fromEntries(this.tools));
  }

  /**
   * Initialize the system prompt with tool definitions.
   */
  protected initializeSystemPrompt(): string {
    // Generate tool documentation
    const toolDocs = Array.from(this.tools.values())
      .filter((tool) => tool.name !== 'final_answer') // final_answer is documented separately
      .map((tool) => tool.toCodePrompt())
      .join('\n\n');

    // Add final_answer documentation
    const finalAnswerDoc = `
/**
 * Returns the final answer to the user. Call this when you have completed the task.
 * @param answer - The final answer (can be any type)
 */
function final_answer(answer: any): void { ... }
`.trim();

    const allTools = toolDocs ? `${toolDocs}\n\n${finalAnswerDoc}` : finalAnswerDoc;

    // Format authorized imports
    const importsDoc = this.authorizedImports.length > 0
      ? this.authorizedImports.map((pkg) => `- ${pkg}`).join('\n')
      : 'None (use built-in capabilities only)';

    return generateSystemPrompt({
      tools: allTools,
      authorizedImports: importsDoc,
      customInstructions: this.config.customInstructions,
    });
  }

  /**
   * Execute a single step: get LLM response, extract code, execute it.
   */
  protected async executeStep(memoryStep: ActionStep): Promise<ActionOutput> {
    // Get messages for LLM
    const messages = this.memory.toMessages();
    memoryStep.modelInputMessages = [...messages];

    // Check if last step had an error - add recovery prompt
    const lastStep = this.memory.getActionSteps().slice(-2)[0]; // Get step before current
    if (lastStep?.error) {
      messages.push({
        role: 'user',
        content: getErrorRecoveryPrompt(lastStep.error.message),
      });
    }

    // Generate response from LLM
    const response = await this.generateResponse(messages);
    memoryStep.modelOutputMessage = response;
    memoryStep.tokenUsage = response.tokenUsage;

    const content = response.content ?? '';

    // Extract thought/reasoning
    const thoughtMatch = content.match(THOUGHT_REGEX);
    if (thoughtMatch) {
      this.logger.reasoning(thoughtMatch[1].trim());
    }

    // Extract code block
    const codeMatch = content.match(CODE_BLOCK_REGEX);

    if (!codeMatch) {
      // No code block found - this might be just reasoning
      // Feed back to LLM to generate code
      this.logger.warn('No code block found in response');
      memoryStep.observation = 'No code block was found in your response. Please provide JavaScript code in a ```javascript code block.';

      return {
        output: null,
        isFinalAnswer: false,
      };
    }

    const code = codeMatch[1].trim();
    memoryStep.codeAction = code;

    this.logger.code(code);

    // Wait before execution (allows user to interrupt)
    if (this.config.codeExecutionDelay > 0) {
      this.logger.waiting(this.config.codeExecutionDelay / 1000);
      await this.sleep(this.config.codeExecutionDelay);
    }

    // Execute the code
    this.logger.subheader('Executing code...');
    const result = await this.executor.execute(code);

    // Log execution logs
    if (result.logs) {
      this.logger.logs(result.logs);
    }

    // Handle execution error
    if (result.error) {
      this.logger.error('Code execution error', result.error);

      memoryStep.error = result.error;
      memoryStep.observation = `Error during code execution:\n${result.error.message}`;

      return {
        output: null,
        isFinalAnswer: false,
      };
    }

    // Format observation
    const outputStr = this.formatOutput(result.output);
    this.logger.output(outputStr);

    memoryStep.observation = this.formatObservation(result.logs, outputStr);

    return {
      output: result.output,
      isFinalAnswer: result.isFinalAnswer,
    };
  }

  /**
   * Generate response from the LLM, optionally streaming.
   */
  private async generateResponse(messages: ChatMessage[]): Promise<ChatMessage> {
    if (this.config.streamOutputs && this.model.supportsStreaming() && this.model.generateStream) {
      // Stream the response
      this.logger.subheader('Agent thinking...');

      let fullContent = '';
      const generator = this.model.generateStream(messages, {
        stopSequences: ['Observation:', 'Observation:\n'],
      });

      for await (const chunk of generator) {
        this.logger.streamChar(chunk);
        fullContent += chunk;
      }

      this.logger.streamEnd();

      return {
        role: 'assistant',
        content: fullContent,
      };
    } else {
      // Non-streaming response
      this.logger.subheader('Agent thinking...');

      return this.model.generate(messages, {
        stopSequences: ['Observation:', 'Observation:\n'],
      });
    }
  }

  /**
   * Format output for display.
   */
  private formatOutput(output: unknown): string {
    if (output === undefined || output === null) {
      return '(no output)';
    }

    if (typeof output === 'string') {
      return output;
    }

    try {
      return JSON.stringify(output, null, 2);
    } catch {
      return String(output);
    }
  }

  /**
   * Format the observation to send back to the LLM.
   */
  private formatObservation(logs: string, output: string): string {
    const parts: string[] = [];

    if (logs.trim()) {
      parts.push(`Execution logs:\n${logs}`);
    }

    parts.push(`Last output:\n${output}`);

    return `Observation:\n${parts.join('\n\n')}`;
  }

  /**
   * Reset the agent and executor state.
   */
  reset(): void {
    this.executor.reset();
    this.currentStep = 0;
  }

  /**
   * Get the executor instance.
   */
  getExecutor(): LocalExecutor {
    return this.executor;
  }

  /**
   * Override addTool to also register with executor.
   */
  addTool(tool: Tool): void {
    super.addTool(tool);
    this.executor.sendTools({ [tool.name]: tool });
  }
}
