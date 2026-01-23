/**
 * CurlTool - HTTP requests (GET/POST) using fetch
 */

import { Tool } from './Tool.js';
import type { ToolInputs } from '../types.js';

export class CurlTool extends Tool {
  readonly name = 'curl';
  readonly description = 'Make HTTP requests to any URL. Supports GET and POST methods with custom headers and body. Returns the response body as text.';
  readonly inputs: ToolInputs = {
    url: {
      type: 'string',
      description: 'The URL to request',
      required: true,
    },
    method: {
      type: 'string',
      description: 'HTTP method: GET or POST (default: GET)',
      required: false,
      default: 'GET',
      enum: ['GET', 'POST'],
    },
    headers: {
      type: 'object',
      description: 'Optional HTTP headers as key-value pairs (e.g., {"Content-Type": "application/json"})',
      required: false,
    },
    body: {
      type: 'string',
      description: 'Request body for POST requests (typically JSON string)',
      required: false,
    },
  };
  readonly outputType = 'string';

  private timeout: number;

  constructor(config?: { timeout?: number }) {
    super();
    this.timeout = config?.timeout ?? 30000;
  }

  async execute(args: Record<string, unknown>): Promise<string> {
    const url = args.url as string;
    const method = ((args.method as string) ?? 'GET').toUpperCase();
    const headers = (args.headers as Record<string, string>) ?? {};
    const body = args.body as string | undefined;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const fetchOptions: RequestInit = {
        method,
        headers,
        signal: controller.signal,
      };

      if (method === 'POST' && body) {
        fetchOptions.body = body;
        if (!headers['Content-Type'] && !headers['content-type']) {
          (fetchOptions.headers as Record<string, string>)['Content-Type'] = 'application/json';
        }
      }

      const response = await fetch(url, fetchOptions);
      const responseText = await response.text();

      const statusLine = `HTTP ${response.status} ${response.statusText}`;

      // Truncate very large responses
      const maxLength = 50000;
      const truncatedBody = responseText.length > maxLength
        ? responseText.slice(0, maxLength) + `\n\n[... truncated, response is ${responseText.length} characters total]`
        : responseText;

      return `${statusLine}\n\n${truncatedBody}`;
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw new Error(`Request timed out after ${this.timeout}ms: ${url}`);
      }
      throw new Error(`HTTP request failed: ${(error as Error).message}`);
    } finally {
      clearTimeout(timeoutId);
    }
  }
}
