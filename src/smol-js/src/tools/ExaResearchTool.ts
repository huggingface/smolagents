/**
 * ExaResearchTool - Deep research on a topic using Exa.ai
 *
 * Performs multi-step research by combining search and content retrieval
 * to produce comprehensive findings on a topic.
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';

export interface ExaResearchConfig {
  apiKey?: string;
}

export class ExaResearchTool extends Tool {
  readonly name = 'exa_research';
  readonly description = 'Perform deep research on a single topic using Exa.ai. Searches for relevant sources, retrieves their content, and finds similar pages for comprehensive coverage. Returns a structured research summary with sources. Use this for thorough research on any topic.';
  readonly inputs: ToolInputs = {
    topic: {
      type: 'string',
      description: 'The research topic or question to investigate',
      required: true,
    },
    numSources: {
      type: 'number',
      description: 'Number of primary sources to retrieve (default: 5, max: 10)',
      required: false,
      default: 5,
    },
    category: {
      type: 'string',
      description: 'Optional category: "research paper", "news", "blog", "company"',
      required: false,
    },
    includeDomains: {
      type: 'array',
      description: 'Only include results from these domains',
      required: false,
    },
    startPublishedDate: {
      type: 'string',
      description: 'Only include results published after this date (ISO 8601)',
      required: false,
    },
  };
  readonly outputType = 'string';

  private apiKey: string;

  constructor(config?: ExaResearchConfig) {
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
    const topic = args.topic as string;
    const numSources = Math.min((args.numSources as number) ?? 5, 10);

    // Step 1: Search for primary sources
    const searchBody: Record<string, unknown> = {
      query: topic,
      numResults: numSources,
      type: 'auto',
      text: { maxCharacters: 3000 },
    };

    if (args.category) searchBody.category = args.category;
    if (args.includeDomains) searchBody.includeDomains = args.includeDomains;
    if (args.startPublishedDate) searchBody.startPublishedDate = args.startPublishedDate;

    const searchResponse = await fetch('https://api.exa.ai/search', {
      method: 'POST',
      headers: {
        'x-api-key': this.apiKey,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(searchBody),
    });

    if (!searchResponse.ok) {
      const errorText = await searchResponse.text();
      throw new Error(`Exa research search failed (${searchResponse.status}): ${errorText}`);
    }

    const searchData = await searchResponse.json() as {
      results: Array<{
        title?: string;
        url: string;
        publishedDate?: string;
        author?: string;
        text?: string;
        score?: number;
      }>;
    };

    if (!searchData.results || searchData.results.length === 0) {
      return `No research sources found for topic: "${topic}"`;
    }

    // Step 2: Find similar pages to top result for broader coverage
    let similarResults: Array<{ title?: string; url: string; text?: string }> = [];
    if (searchData.results.length > 0) {
      try {
        const similarBody = {
          url: searchData.results[0].url,
          numResults: 3,
          text: { maxCharacters: 2000 },
        };

        const similarResponse = await fetch('https://api.exa.ai/findSimilar', {
          method: 'POST',
          headers: {
            'x-api-key': this.apiKey,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(similarBody),
        });

        if (similarResponse.ok) {
          const similarData = await similarResponse.json() as { results: typeof similarResults };
          similarResults = similarData.results ?? [];
        }
      } catch {
        // Non-critical, continue without similar results
      }
    }

    // Step 3: Compile research summary
    const allSources = [...searchData.results, ...similarResults];
    const seenUrls = new Set<string>();
    const uniqueSources = allSources.filter(s => {
      if (seenUrls.has(s.url)) return false;
      seenUrls.add(s.url);
      return true;
    });

    const sections: string[] = [];
    sections.push(`# Research: ${topic}\n`);
    sections.push(`Found ${uniqueSources.length} sources.\n`);

    sections.push('## Key Sources\n');
    for (let i = 0; i < uniqueSources.length; i++) {
      const source = uniqueSources[i];
      sections.push(`### ${i + 1}. ${source.title ?? 'Untitled'}`);
      sections.push(`URL: ${source.url}`);
      if ('publishedDate' in source && source.publishedDate) {
        sections.push(`Date: ${source.publishedDate}`);
      }
      if ('author' in source && source.author) {
        sections.push(`Author: ${source.author}`);
      }
      if (source.text) {
        sections.push(`\nContent:\n${source.text.slice(0, 2000)}`);
      }
      sections.push('');
    }

    // Compile source list
    sections.push('## Source URLs\n');
    uniqueSources.forEach((s, i) => {
      sections.push(`${i + 1}. ${s.url}`);
    });

    return sections.join('\n');
  }
}
