/**
 * Example 6: Deep Research Agent (MVP)
 *
 * A simplified deep research agent that performs multi-step online research:
 * 1. Searches the web for information
 * 2. Fetches and reads web pages
 * 3. Synthesizes findings into a research report
 *
 * This example uses REAL APIs - no mocking:
 * - DuckDuckGo Instant Answer API for search
 * - Direct fetch for web pages
 * - HTML to text extraction
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, Tool, LogLevel } from '../src/index.js';
import type { ToolInputs } from '../src/types.js';

/**
 * Web Search Tool - Uses DuckDuckGo Instant Answer API
 * Falls back to a simple scraping approach if needed
 */
class WebSearchTool extends Tool {
  readonly name = 'web_search';
  readonly description = `Search the web for information. Returns search results with titles, snippets, and URLs.
Use this to find information about any topic. The query should be a clear search phrase.`;
  readonly inputs: ToolInputs = {
    query: {
      type: 'string',
      description: 'The search query - be specific and descriptive',
      required: true,
    },
  };
  readonly outputType = 'array';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    const query = args.query as string;
    console.log(`[WebSearch] Searching for: "${query}"`);

    try {
      // Try DuckDuckGo Instant Answer API first
      const ddgUrl = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
      const response = await fetch(ddgUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SmolJS/1.0; Research Bot)',
        },
      });

      if (!response.ok) {
        throw new Error(`DuckDuckGo API error: ${response.status}`);
      }

      const data = await response.json() as {
        Abstract?: string;
        AbstractSource?: string;
        AbstractURL?: string;
        Heading?: string;
        RelatedTopics?: Array<{
          Text?: string;
          FirstURL?: string;
          Result?: string;
        }>;
        Results?: Array<{
          Text?: string;
          FirstURL?: string;
        }>;
      };

      const results: Array<{ title: string; snippet: string; url: string }> = [];

      // Add main abstract if available
      if (data.Abstract && data.AbstractURL) {
        results.push({
          title: data.Heading || 'Main Result',
          snippet: data.Abstract,
          url: data.AbstractURL,
        });
      }

      // Add related topics
      if (data.RelatedTopics) {
        for (const topic of data.RelatedTopics.slice(0, 5)) {
          if (topic.Text && topic.FirstURL) {
            results.push({
              title: topic.Text.split(' - ')[0] || 'Related',
              snippet: topic.Text,
              url: topic.FirstURL,
            });
          }
        }
      }

      // Add direct results
      if (data.Results) {
        for (const result of data.Results.slice(0, 3)) {
          if (result.Text && result.FirstURL) {
            results.push({
              title: result.Text.split(' - ')[0] || 'Result',
              snippet: result.Text,
              url: result.FirstURL,
            });
          }
        }
      }

      // If no results from DDG API, try a fallback search via Wikipedia API
      if (results.length === 0) {
        console.log('[WebSearch] No DDG results, trying Wikipedia...');
        const wikiResults = await this.searchWikipedia(query);
        results.push(...wikiResults);
      }

      console.log(`[WebSearch] Found ${results.length} results`);
      return results;
    } catch (error) {
      console.error('[WebSearch] Error:', (error as Error).message);
      // Fallback to Wikipedia search
      return this.searchWikipedia(query);
    }
  }

  private async searchWikipedia(query: string): Promise<Array<{ title: string; snippet: string; url: string }>> {
    try {
      const wikiUrl = `https://en.wikipedia.org/w/api.php?action=opensearch&search=${encodeURIComponent(query)}&limit=5&format=json&origin=*`;
      const response = await fetch(wikiUrl);
      const [, titles, snippets, urls] = await response.json() as [string, string[], string[], string[]];

      return titles.map((title, i) => ({
        title,
        snippet: snippets[i] || title,
        url: urls[i],
      }));
    } catch {
      return [];
    }
  }
}

/**
 * Web Page Fetcher Tool - Fetches and extracts text from web pages
 */
class WebFetchTool extends Tool {
  readonly name = 'fetch_webpage';
  readonly description = `Fetch a webpage and extract its text content.
Use this to read the full content of a URL found in search results.
Returns the page title and main text content.`;
  readonly inputs: ToolInputs = {
    url: {
      type: 'string',
      description: 'The URL to fetch',
      required: true,
    },
    maxLength: {
      type: 'number',
      description: 'Maximum characters to return (default: 8000)',
      required: false,
    },
  };
  readonly outputType = 'object';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    const url = args.url as string;
    const maxLength = (args.maxLength as number) || 8000;

    console.log(`[WebFetch] Fetching: ${url}`);

    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SmolJS/1.0; Research Bot)',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        },
        redirect: 'follow',
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const contentType = response.headers.get('content-type') || '';
      const html = await response.text();

      // Extract title
      const titleMatch = html.match(/<title[^>]*>([^<]+)<\/title>/i);
      const title = titleMatch ? titleMatch[1].trim() : 'Untitled';

      // Extract text content (simple HTML to text conversion)
      let text = this.htmlToText(html);

      // Truncate if too long
      if (text.length > maxLength) {
        text = text.slice(0, maxLength) + '\n\n[Content truncated...]';
      }

      console.log(`[WebFetch] Got ${text.length} chars from "${title}"`);

      return {
        url,
        title,
        content: text,
        contentType,
      };
    } catch (error) {
      console.error(`[WebFetch] Error fetching ${url}:`, (error as Error).message);
      return {
        url,
        title: 'Error',
        content: `Failed to fetch page: ${(error as Error).message}`,
        error: true,
      };
    }
  }

  private htmlToText(html: string): string {
    // Remove scripts and styles
    let text = html.replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '');
    text = text.replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '');
    text = text.replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, '');
    text = text.replace(/<header[^>]*>[\s\S]*?<\/header>/gi, '');
    text = text.replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, '');

    // Convert common block elements to newlines
    text = text.replace(/<\/?(p|div|br|h[1-6]|li|tr)[^>]*>/gi, '\n');

    // Remove all remaining HTML tags
    text = text.replace(/<[^>]+>/g, ' ');

    // Decode HTML entities
    text = text.replace(/&nbsp;/g, ' ');
    text = text.replace(/&amp;/g, '&');
    text = text.replace(/&lt;/g, '<');
    text = text.replace(/&gt;/g, '>');
    text = text.replace(/&quot;/g, '"');
    text = text.replace(/&#(\d+);/g, (_, code) => String.fromCharCode(parseInt(code)));

    // Clean up whitespace
    text = text.replace(/[ \t]+/g, ' ');
    text = text.replace(/\n\s*\n/g, '\n\n');
    text = text.trim();

    return text;
  }
}

/**
 * Get Current Date Tool - Provides current date for time-sensitive research
 */
class CurrentDateTool extends Tool {
  readonly name = 'get_current_date';
  readonly description = 'Get the current date and time. Useful for time-sensitive research.';
  readonly inputs: ToolInputs = {};
  readonly outputType = 'object';

  async execute(): Promise<unknown> {
    const now = new Date();
    return {
      date: now.toISOString().split('T')[0],
      time: now.toISOString().split('T')[1].split('.')[0],
      year: now.getFullYear(),
      month: now.getMonth() + 1,
      day: now.getDate(),
      timestamp: now.toISOString(),
    };
  }
}

async function main() {
  console.log('=== Example 6: Deep Research Agent ===\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
    maxTokens: 4096,
  });

  // Create research tools
  const webSearch = new WebSearchTool();
  const webFetch = new WebFetchTool();
  const currentDate = new CurrentDateTool();

  // Create the research agent
  const agent = new CodeAgent({
    model,
    tools: [webSearch, webFetch, currentDate],
    maxSteps: 15,
    codeExecutionDelay: 500, // Shorter delay for research
    verboseLevel: LogLevel.INFO,
    customInstructions: `You are a research assistant that finds and synthesizes information from the web.

When given a research question:
1. First, understand what information is needed
2. Search the web for relevant sources
3. Fetch and read the most promising pages
4. Extract key facts and cite your sources
5. Synthesize findings into a clear, well-organized answer

Always cite your sources with URLs. If information conflicts between sources, note the discrepancy.
Be thorough but focused - quality over quantity.`,
  });

  // Research question
  const question = `What are the latest developments in nuclear fusion energy research in 2024-2025?
Include information about major projects, recent breakthroughs, and timeline predictions for commercial fusion power.`;

  console.log(`Research Question: ${question}\n`);

  // Run the research
  const result = await agent.run(question);

  console.log('\n' + '='.repeat(60));
  console.log('RESEARCH COMPLETE');
  console.log('='.repeat(60));
  console.log('\nFinal Report:');
  console.log(typeof result.output === 'string' ? result.output : JSON.stringify(result.output, null, 2));
  console.log('\nStats:');
  console.log(`- Steps taken: ${result.steps.filter(s => s.type === 'action').length}`);
  console.log(`- Total duration: ${(result.duration / 1000).toFixed(2)}s`);
}

main().catch(console.error);
