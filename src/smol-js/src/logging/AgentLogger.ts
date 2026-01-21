/**
 * AgentLogger - Color-coded console logging for agent execution
 *
 * Provides formatted output with different colors for:
 * - Headers (cyan)
 * - Reasoning/Thoughts (yellow)
 * - Code blocks (green)
 * - Output/Results (blue)
 * - Errors (red)
 */

import chalk from 'chalk';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { LogLevel } from '../types.js';

// Log file directory
const LOG_DIR = path.join(os.homedir(), '.smol-js/logs');

export class AgentLogger {
  private level: LogLevel;
  private logFile?: fs.WriteStream;
  private sessionId: string;

  constructor(level: LogLevel = LogLevel.INFO) {
    this.level = level;
    this.sessionId = this.generateSessionId();

    // Initialize log file if logging is enabled
    if (level > LogLevel.OFF) {
      this.initLogFile();
    }
  }

  /**
   * Generate a unique session ID.
   */
  private generateSessionId(): string {
    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-');
    return `session-${timestamp}`;
  }

  /**
   * Initialize the log file.
   */
  private initLogFile(): void {
    try {
      // Create log directory if it doesn't exist
      if (!fs.existsSync(LOG_DIR)) {
        fs.mkdirSync(LOG_DIR, { recursive: true });
      }

      const logPath = path.join(LOG_DIR, `${this.sessionId}.log`);
      this.logFile = fs.createWriteStream(logPath, { flags: 'a' });

      this.writeToFile(`=== Session Started: ${new Date().toISOString()} ===\n\n`);
    } catch (error) {
      console.warn('Could not create log file:', (error as Error).message);
    }
  }

  /**
   * Write to the log file.
   */
  private writeToFile(content: string): void {
    if (this.logFile) {
      // Strip ANSI codes for file output
      // eslint-disable-next-line no-control-regex
      const cleanContent = content.replace(/\x1b\[[0-9;]*m/g, '');
      this.logFile.write(cleanContent);
    }
  }

  /**
   * Set the log level.
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Log a header (task start, step start, etc.)
   */
  header(message: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const line = '═'.repeat(60);
    const output = `\n${chalk.cyan(line)}\n${chalk.cyan.bold(message)}\n${chalk.cyan(line)}\n`;

    console.log(output);
    this.writeToFile(`\n${'═'.repeat(60)}\n${message}\n${'═'.repeat(60)}\n`);
  }

  /**
   * Log a subheader.
   */
  subheader(message: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `\n${chalk.cyan('─'.repeat(40))}\n${chalk.cyan(message)}\n`;

    console.log(output);
    this.writeToFile(`\n${'─'.repeat(40)}\n${message}\n`);
  }

  /**
   * Log reasoning/thought from the agent.
   */
  reasoning(content: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.yellow.bold('💭 Reasoning:')}\n${chalk.yellow(content)}\n`;

    console.log(output);
    this.writeToFile(`\n💭 Reasoning:\n${content}\n`);
  }

  /**
   * Log code block.
   */
  code(content: string, language: string = 'javascript', level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.green.bold('📝 Code:')}\n${chalk.green('```' + language)}\n${chalk.green(content)}\n${chalk.green('```')}\n`;

    console.log(output);
    this.writeToFile(`\n📝 Code:\n\`\`\`${language}\n${content}\n\`\`\`\n`);
  }

  /**
   * Log execution output.
   */
  output(content: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.blue.bold('📤 Output:')}\n${chalk.blue(content)}\n`;

    console.log(output);
    this.writeToFile(`\n📤 Output:\n${content}\n`);
  }

  /**
   * Log execution logs (print statements).
   */
  logs(content: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level || !content.trim()) return;

    const output = `${chalk.gray.bold('📋 Logs:')}\n${chalk.gray(content)}\n`;

    console.log(output);
    this.writeToFile(`\n📋 Logs:\n${content}\n`);
  }

  /**
   * Log an error.
   */
  error(message: string, error?: Error, level: LogLevel = LogLevel.ERROR): void {
    if (this.level < level) return;

    const errorMessage = error ? `${message}: ${error.message}` : message;
    const output = `${chalk.red.bold('❌ Error:')}\n${chalk.red(errorMessage)}\n`;

    console.error(output);
    this.writeToFile(`\n❌ Error:\n${errorMessage}\n`);

    if (error?.stack && this.level >= LogLevel.DEBUG) {
      console.error(chalk.red.dim(error.stack));
      this.writeToFile(`Stack: ${error.stack}\n`);
    }
  }

  /**
   * Log a warning.
   */
  warn(message: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.yellow.bold('⚠️ Warning:')} ${chalk.yellow(message)}\n`;

    console.warn(output);
    this.writeToFile(`\n⚠️ Warning: ${message}\n`);
  }

  /**
   * Log general info.
   */
  info(message: string, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.white(message)}`;

    console.log(output);
    this.writeToFile(`${message}\n`);
  }

  /**
   * Log debug info.
   */
  debug(message: string): void {
    if (this.level < LogLevel.DEBUG) return;

    const output = `${chalk.dim('[DEBUG]')} ${chalk.dim(message)}`;

    console.log(output);
    this.writeToFile(`[DEBUG] ${message}\n`);
  }

  /**
   * Log final answer.
   */
  finalAnswer(answer: unknown, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const line = '═'.repeat(60);
    const answerStr = typeof answer === 'string' ? answer : JSON.stringify(answer, null, 2);
    const output = `\n${chalk.magenta(line)}\n${chalk.magenta.bold('✅ Final Answer:')}\n${chalk.magenta(answerStr)}\n${chalk.magenta(line)}\n`;

    console.log(output);
    this.writeToFile(`\n${'═'.repeat(60)}\n✅ Final Answer:\n${answerStr}\n${'═'.repeat(60)}\n`);
  }

  /**
   * Log step progress.
   */
  stepProgress(current: number, max: number, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.cyan.bold(`\n🔄 Step ${current}/${max}`)}\n`;

    console.log(output);
    this.writeToFile(`\n🔄 Step ${current}/${max}\n`);
  }

  /**
   * Log waiting message for code execution delay.
   */
  waiting(seconds: number, level: LogLevel = LogLevel.INFO): void {
    if (this.level < level) return;

    const output = `${chalk.yellow(`⏳ Waiting ${seconds}s before code execution (Ctrl+C to abort)...`)}`;

    console.log(output);
    this.writeToFile(`⏳ Waiting ${seconds}s before code execution...\n`);
  }

  /**
   * Stream content character by character.
   */
  streamChar(char: string): void {
    if (this.level < LogLevel.INFO) return;
    process.stdout.write(chalk.yellow(char));
  }

  /**
   * End streaming (add newline).
   */
  streamEnd(): void {
    if (this.level < LogLevel.INFO) return;
    console.log();
  }

  /**
   * Close the log file.
   */
  close(): void {
    if (this.logFile) {
      this.writeToFile(`\n=== Session Ended: ${new Date().toISOString()} ===\n`);
      this.logFile.end();
    }
  }

  /**
   * Get the log file path.
   */
  getLogPath(): string | undefined {
    return this.logFile ? path.join(LOG_DIR, `${this.sessionId}.log`) : undefined;
  }
}
