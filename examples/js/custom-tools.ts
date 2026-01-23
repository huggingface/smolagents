/**
 * Custom tool implementations for the YAML workflow demo.
 *
 * These tools are defined here in code, then registered with the YAMLLoader
 * so the YAML can reference them by type name. This is how you bridge the
 * declarative YAML world with imperative TypeScript tool implementations.
 */

import { Tool } from '@samrahimi/smol-js';
import type { ToolInputs } from '@samrahimi/smol-js';

/**
 * TimestampTool - Returns the current date/time in a specified format.
 *
 * In YAML, this is referenced as:
 *   tools:
 *     my_timestamp:
 *       type: timestamp    <-- this string maps to the class via registerToolType()
 */
export class TimestampTool extends Tool {
  readonly name = 'timestamp';
  readonly description = 'Returns the current date and time. Can return ISO format, unix epoch, or a human-readable string.';
  readonly inputs: ToolInputs = {
    format: {
      type: 'string',
      description: 'Output format: "iso", "unix", "human", or "date_only"',
      required: false,
      default: 'iso',
    },
  };
  readonly outputType = 'string';

  async execute(args: Record<string, unknown>): Promise<string> {
    const format = (args.format as string) || 'iso';
    const now = new Date();

    switch (format) {
      case 'unix':
        return String(Math.floor(now.getTime() / 1000));
      case 'human':
        return now.toLocaleString('en-US', {
          weekday: 'long', year: 'numeric', month: 'long', day: 'numeric',
          hour: '2-digit', minute: '2-digit', second: '2-digit',
        });
      case 'date_only':
        return now.toISOString().split('T')[0];
      case 'iso':
      default:
        return now.toISOString();
    }
  }
}

/**
 * TextStatsTool - Analyzes text and returns word count, sentence count, etc.
 *
 * In YAML, referenced as:
 *   tools:
 *     stats:
 *       type: text_stats
 */
export class TextStatsTool extends Tool {
  readonly name = 'text_stats';
  readonly description = 'Analyzes text and returns statistics: word count, sentence count, paragraph count, character count, and estimated reading time.';
  readonly inputs: ToolInputs = {
    text: {
      type: 'string',
      description: 'The text to analyze',
      required: true,
    },
  };
  readonly outputType = 'string';

  async execute(args: Record<string, unknown>): Promise<string> {
    const text = args.text as string;

    const words = text.split(/\s+/).filter(w => w.length > 0).length;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0).length;
    const characters = text.length;
    const readingTimeMinutes = Math.max(1, Math.round(words / 200));

    return JSON.stringify({
      words,
      sentences,
      paragraphs,
      characters,
      estimatedReadingTime: `${readingTimeMinutes} min`,
    }, null, 2);
  }
}

/**
 * SlugifyTool - Converts text to a URL-friendly slug.
 *
 * In YAML, referenced as:
 *   tools:
 *     slug:
 *       type: slugify
 */
export class SlugifyTool extends Tool {
  readonly name = 'slugify';
  readonly description = 'Converts a title or text string into a URL-friendly slug (lowercase, hyphens, no special chars).';
  readonly inputs: ToolInputs = {
    text: {
      type: 'string',
      description: 'The text to convert to a slug',
      required: true,
    },
  };
  readonly outputType = 'string';

  async execute(args: Record<string, unknown>): Promise<string> {
    const text = args.text as string;
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '')
      .replace(/[\s_]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '');
  }
}
