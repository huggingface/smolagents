/**
 * ExaSearchTool - Web search using the Exa.ai API
 *
 * Uses Exa's embeddings-based search for semantically intelligent results.
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';

export interface ExaSearchConfig {
  apiKey?: string;
}

export class ExaSearchTool extends Tool {
  readonly name = 'exa_search';
  readonly description = 'Search the web using Exa.ai semantic search. Returns relevant web pages with titles, URLs, and optionally content snippets. Use this for finding information, research, and fact-checking.';
  readonly inputs: ToolInputs = {
    query: {
      type: 'string',
      description: 'The search query. Be specific and descriptive for best results.',
      required: true,
    },
    numResults: {
      type: 'number',
      description: 'Number of results to return (default: 10, max: 30)',
      required: false,
      default: 10,
    },
    type: {
      type: 'string',
      description: 'Search type: "auto" (default), "neural" (embeddings-based), or "keyword"',
      required: false,
      default: 'auto',
      enum: ['auto', 'neural', 'keyword'],
    },
    category: {
      type: 'string',
      description: 'Optional category filter: "research paper", "news", "pdf", "github", "tweet", "company", "blog"',
      required: false,
    },
    includeDomains: {
      type: 'array',
      description: 'Only include results from these domains (e.g., ["arxiv.org", "github.com"])',
      required: false,
    },
    excludeDomains: {
      type: 'array',
      description: 'Exclude results from these domains',
      required: false,
    },
    startPublishedDate: {
      type: 'string',
      description: 'Filter results published after this ISO 8601 date (e.g., "2024-01-01")',
      required: false,
    },
  };
  readonly outputType = 'string';

  private apiKey: string;

  constructor(config?: ExaSearchConfig) {
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
    const query = args.query as string;
    const numResults = Math.min((args.numResults as number) ?? 10, 30);
    const type = (args.type as string) ?? 'auto';

    const requestBody: Record<string, unknown> = {
      query,
      numResults,
      type,
      text: { maxCharacters: 1000 },
    };

    if (args.category) requestBody.category = args.category;
    if (args.includeDomains) requestBody.includeDomains = args.includeDomains;
    if (args.excludeDomains) requestBody.excludeDomains = args.excludeDomains;
    if (args.startPublishedDate) requestBody.startPublishedDate = args.startPublishedDate;

    const response = await fetch('https://api.exa.ai/search', {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Exa search failed (${response.status}): ${errorText}`);
    }

    const data = await response.json() as {
      results: Array<{
        title?: string;
        url: string;
        publishedDate?: string;
        author?: string;
        text?: string;
        score?: number;
      }>;
    };

    if (!data.results || data.results.length === 0) {
      return 'No results found for the query.';
    }

    // Format results for the agent
    const formattedResults = data.results.map((result, i) => {
      const parts = [`[${i + 1}] ${result.title ?? 'Untitled'}`];
      parts.push(`    URL: ${result.url}`);
      if (result.publishedDate) parts.push(`    Date: ${result.publishedDate}`);
      if (result.author) parts.push(`    Author: ${result.author}`);
      if (result.text) {
        const snippet = result.text.slice(0, 300).trim();
        parts.push(`    Snippet: ${snippet}${result.text.length > 300 ? '...' : ''}`);
      }
      return parts.join('\n');
    }).join('\n\n');

    return `Search results for "${query}" (${data.results.length} results):\n\n${formattedResults}`;
  }
}
