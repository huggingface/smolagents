/**
 * ExaGetContentsTool - Get webpage contents using Exa.ai API
 *
 * Fetches and extracts clean text content from web pages.
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';

export interface ExaGetContentsConfig {
  apiKey?: string;
}

export class ExaGetContentsTool extends Tool {
  readonly name = 'exa_get_contents';
  readonly description = 'Get the full text content of one or more web pages using Exa.ai. Returns cleaned, readable text extracted from the HTML. Use this to read articles, documentation, or any web page content.';
  readonly inputs: ToolInputs = {
    urls: {
      type: 'array',
      description: 'Array of URLs to fetch content from (max 10)',
      required: true,
    },
    maxCharacters: {
      type: 'number',
      description: 'Maximum characters of content to return per page (default: 10000)',
      required: false,
      default: 10000,
    },
    livecrawl: {
      type: 'string',
      description: 'Crawl strategy: "fallback" (use cache, fetch live if unavailable), "always" (always fetch live), "never" (cache only). Default: "fallback"',
      required: false,
      default: 'fallback',
      enum: ['fallback', 'always', 'never'],
    },
  };
  readonly outputType = 'string';

  private apiKey: string;

  constructor(config?: ExaGetContentsConfig) {
    super();
    this.apiKey = config?.apiKey ?? process.env.EXA_API_KEY ?? '';
  }

  async setup(): Promise<void> {
    if (!this.apiKey) {
      throw new Error('EXA_API_KEY is required. Set it as an environment variable or pass it in the config.');
    }
    this.isSetup = true;
  }

  async execute(args: Record<string, unknown>): Promise<string> {
    const urls = (args.urls as string[]).slice(0, 10);
    const maxCharacters = (args.maxCharacters as number) ?? 10000;
    const livecrawl = (args.livecrawl as string) ?? 'fallback';

    const requestBody = {
      urls,
      text: { maxCharacters },
      livecrawl,
    };

    const response = await fetch('https://api.exa.ai/contents', {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Exa get contents failed (${response.status}): ${errorText}`);
    }

    const data = await response.json() as {
      results: Array<{
        url: string;
        title?: string;
        author?: string;
        publishedDate?: string;
        text?: string;
      }>;
    };

    if (!data.results || data.results.length === 0) {
      return 'No content could be retrieved from the provided URLs.';
    }

    const formattedResults = data.results.map((result) => {
      const parts = [`## ${result.title ?? result.url}`];
      parts.push(`URL: ${result.url}`);
      if (result.author) parts.push(`Author: ${result.author}`);
      if (result.publishedDate) parts.push(`Date: ${result.publishedDate}`);
      parts.push('');
      if (result.text) {
        parts.push(result.text);
      } else {
        parts.push('[No text content available]');
      }
      return parts.join('\n');
    }).join('\n\n---\n\n');

    return formattedResults;
  }
}
