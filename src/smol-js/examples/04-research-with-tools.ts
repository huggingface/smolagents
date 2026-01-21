/**
 * Example 4: Multi-step Research Task with Tools
 *
 * Demonstrates using custom tools to enable the agent to perform
 * research-like tasks such as web searches and API calls.
 */

import 'dotenv/config';
import { CodeAgent, OpenAIModel, Tool, LogLevel } from '../src/index.js';
import type { ToolInputs } from '../src/types.js';

// Custom tool for simulated web search
class WebSearchTool extends Tool {
  readonly name = 'web_search';
  readonly description = 'Search the web for information. Returns simulated search results for demo purposes.';
  readonly inputs: ToolInputs = {
    query: {
      type: 'string',
      description: 'The search query',
      required: true,
    },
  };
  readonly outputType = 'array';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    const query = args.query as string;

    // Simulated search results for demo
    const results = [
      {
        title: `Top result for: ${query}`,
        snippet: `This is a simulated search result about ${query}. In a real implementation, this would fetch actual web results.`,
        url: `https://example.com/search?q=${encodeURIComponent(query)}`,
      },
      {
        title: `Related: ${query} - Wikipedia`,
        snippet: `Wikipedia article about ${query} with comprehensive information and references.`,
        url: `https://en.wikipedia.org/wiki/${encodeURIComponent(query)}`,
      },
      {
        title: `${query} - Latest News`,
        snippet: `Breaking news and updates related to ${query}.`,
        url: `https://news.example.com/${encodeURIComponent(query)}`,
      },
    ];

    // Simulate network delay
    await new Promise((r) => setTimeout(r, 500));

    return results;
  }
}

// Custom tool for fetching weather (simulated)
class WeatherTool extends Tool {
  readonly name = 'get_weather';
  readonly description = 'Get current weather for a city. Returns simulated weather data.';
  readonly inputs: ToolInputs = {
    city: {
      type: 'string',
      description: 'The city name',
      required: true,
    },
  };
  readonly outputType = 'object';

  async execute(args: Record<string, unknown>): Promise<unknown> {
    const city = args.city as string;

    // Simulated weather data
    const weather = {
      city,
      temperature: Math.round(Math.random() * 30 + 5), // 5-35°C
      humidity: Math.round(Math.random() * 60 + 30), // 30-90%
      condition: ['sunny', 'cloudy', 'rainy', 'partly cloudy'][Math.floor(Math.random() * 4)],
      windSpeed: Math.round(Math.random() * 20 + 5), // 5-25 km/h
    };

    // Simulate network delay
    await new Promise((r) => setTimeout(r, 300));

    return weather;
  }
}

async function main() {
  console.log('=== Example 4: Multi-step Research Task with Tools ===\n');

  // Create the model
  const model = new OpenAIModel({
    modelId: 'anthropic/claude-sonnet-4.5',
  });

  // Create custom tools
  const webSearch = new WebSearchTool();
  const weather = new WeatherTool();

  // Create the agent with tools
  const agent = new CodeAgent({
    model,
    tools: [webSearch, weather],
    maxSteps: 10,
    codeExecutionDelay: 1000,
    verboseLevel: LogLevel.INFO,
  });

  // Run a research task
  const result = await agent.run(
    `Research the following cities and compile a travel report:
     1. Search the web for "best restaurants in Paris"
     2. Get the weather for Paris
     3. Search the web for "tourist attractions Tokyo"
     4. Get the weather for Tokyo

     Compile all information into a travel summary object with:
     - paris: { searchResults, weather }
     - tokyo: { searchResults, weather }

     Return this object using final_answer.`
  );

  console.log('\n=== Result ===');
  console.log('Output:', JSON.stringify(result.output, null, 2));
  console.log('Steps:', result.steps.length);
  console.log('Duration:', (result.duration / 1000).toFixed(2), 'seconds');
}

main().catch(console.error);
