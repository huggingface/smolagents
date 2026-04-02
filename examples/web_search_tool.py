"""
Web Search and Analysis Agent using Olostep and OpenAI

This example demonstrates how to use Olostep tools (search, scrape, answer)
with OpenAI models to research topics on the live web.

Olostep provides:
- Real-time web search with structured results
- JavaScript-rendered webpage scraping
- Grounded Q&A with source citations

Requirements:
    - OPENAI_API_KEY environment variable (get at https://platform.openai.com/api-keys)
    - OLOSTEP_API_KEY environment variable (get free key at https://www.olostep.com/dashboard/api-keys)

Example usage:
    python examples/web_search_tool.py
"""

import os
from smolagents import ToolCallingAgent, OpenAIModel, OlostepAnswerTool, OlostepScrapeWebpageTool, OlostepSearchTool


def main():
    """Run a web research agent using Olostep tools and OpenAI."""

    # Check for required API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Get an API key at: https://platform.openai.com/api-keys")
        return

    if not os.environ.get("OLOSTEP_API_KEY"):
        print("❌ Error: OLOSTEP_API_KEY environment variable not set")
        print("   Get a free API key at: https://www.olostep.com/dashboard/api-keys")
        return

    # Initialize the OpenAI model
    model = OpenAIModel(model_id="gpt-4o-mini")

    # Initialize Olostep tools for live web access
    search_tool = OlostepSearchTool(max_results=5, country="US")
    scrape_tool = OlostepScrapeWebpageTool(wait_before_scraping=0, country="US")
    answer_tool = OlostepAnswerTool()

    # Create a tool-calling agent with Olostep tools
    agent = ToolCallingAgent(
        tools=[search_tool, scrape_tool, answer_tool],
        model=model,
        max_steps=10,
        verbosity_level=2,
    )

    # Define research tasks
    research_queries = [
        # Task 1: Find latest information about a specific topic
        "What are the latest developments in open-source AI agents as of 2025? "
        "Use olostep_search to find recent articles, then olostep_scrape_webpage to get full details.",
        # Task 2: Comparative research
        "Compare the top 3 Python web scraping libraries in 2025. "
        "Use olostep_search to find each library's documentation.",
        # Task 3: Get grounded answers directly
        "What is the current status of Python 3.13 release? "
        "Use olostep_answer to get a grounded answer with sources.",
    ]

    # Run one of the research queries
    print("=" * 80)
    print("OLOSTEP + OPENAI WEB RESEARCH AGENT")
    print("=" * 80)
    print()

    selected_query = research_queries[0]  # Use the first query
    print(f"🔍 Research Query:\n{selected_query}")
    print()
    print("-" * 80)

    try:
        result = agent.run(selected_query)
        print()
        print("-" * 80)
        print("📊 Research Results:")
        print(result)
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
