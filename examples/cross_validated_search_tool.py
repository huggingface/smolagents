"""
Cross-Validated Search Tool for smolagents

This example demonstrates how to use cross-validated-search with smolagents
for hallucination-free web search. Unlike standard web search, cross-validated-search
verifies facts across multiple sources and assigns confidence scores.

Installation:
    pip install cross-validated-search smolagents

Usage:
    python cross_validated_search_tool.py
"""

from smolagents import CodeAgent, InferenceClientModel, tool
from typing import Dict, Any, List


@tool
def cross_validated_search(query: str, search_type: str = "text") -> str:
    """
    Search the web with cross-validation for hallucination-free results.
    
    This tool performs a cross-validated web search, verifying facts across
    multiple search engines and returning results with confidence scores.
    
    Args:
        query: The search query.
        search_type: Type of search - "text", "news", or "images". Default is "text".
    
    Returns:
        A string containing the answer, confidence level, and sources.
        Confidence levels:
        - "verified": 3+ sources agree, high confidence
        - "likely_true": 2 sources agree, medium confidence
        - "uncertain": single source or conflicts
        - "likely_false": major contradictions
    
    Example:
        >>> cross_validated_search("What is the latest version of Python?")
        "Answer: Python 3.13 is the latest stable release...
         Confidence: verified
         Sources: python.org, wikipedia.org, docs.python.org"
    """
    try:
        from cross_validated_search import CrossValidatedSearcher
    except ImportError:
        return "Error: cross-validated-search not installed. Run: pip install cross-validated-search"
    
    searcher = CrossValidatedSearcher()
    results = searcher.search(query, search_type=search_type)
    
    # Format the response
    response_parts = [
        f"Answer: {results.answer}",
        f"Confidence: {results.confidence}",
        f"Sources ({len(results.sources)} found):",
    ]
    
    # Add top sources
    for i, source in enumerate(results.sources[:5], 1):
        response_parts.append(f"  {i}. [{source.engine}] {source.title}")
        response_parts.append(f"     URL: {source.url}")
        if source.snippet:
            snippet = source.snippet[:100] + "..." if len(source.snippet) > 100 else source.snippet
            response_parts.append(f"     Snippet: {snippet}")
    
    # Add confidence explanation
    confidence_info = {
        "verified": "✅ Verified - 3+ sources agree",
        "likely_true": "🟢 Likely True - 2 sources agree",
        "uncertain": "🟡 Uncertain - Single source or conflicts",
        "likely_false": "🔴 Likely False - Major contradictions",
    }
    response_parts.append(f"\n{confidence_info.get(results.confidence, 'Unknown confidence')}")
    
    return "\n".join(response_parts)


@tool
def search_with_fact_check(claim: str) -> str:
    """
    Fact-check a claim using cross-validated web search.
    
    This tool is specifically designed for fact-checking. It searches for
    information about the claim and returns whether it's verified, likely true,
    uncertain, or likely false.
    
    Args:
        claim: The claim to fact-check.
    
    Returns:
        A fact-check result with confidence level and evidence.
    
    Example:
        >>> search_with_fact_check("Python 3.14 is released")
        "Claim: Python 3.14 is released
         Status: Likely False
         Evidence: Python 3.13 is the latest stable release..."
    """
    try:
        from cross_validated_search import CrossValidatedSearcher
    except ImportError:
        return "Error: cross-validated-search not installed. Run: pip install cross-validated-search"
    
    searcher = CrossValidatedSearcher()
    results = searcher.search(claim, search_type="text")
    
    # Determine claim status
    if results.confidence == "verified":
        status = "✅ VERIFIED"
        explanation = "Multiple sources confirm this claim."
    elif results.confidence == "likely_true":
        status = "🟢 LIKELY TRUE"
        explanation = "At least 2 sources agree on this claim."
    elif results.confidence == "uncertain":
        status = "🟡 UNCERTAIN"
        explanation = "Only one source found or sources disagree."
    else:
        status = "🔴 LIKELY FALSE"
        explanation = "Sources contradict this claim or no evidence found."
    
    # Format response
    response_parts = [
        f"Claim: {claim}",
        f"Status: {status}",
        f"Explanation: {explanation}",
        f"",
        "Evidence:",
    ]
    
    for source in results.sources[:3]:
        response_parts.append(f"  - [{source.engine}] {source.title}")
        response_parts.append(f"    {source.url}")
    
    return "\n".join(response_parts)


@tool
def search_news_with_verification(query: str) -> str:
    """
    Search for news with cross-verification.
    
    This tool searches news sources and cross-validates the information
    across multiple outlets.
    
    Args:
        query: The news query.
    
    Returns:
        News results with confidence scores.
    
    Example:
        >>> search_news_with_verification("AI breakthrough 2024")
        "News about 'AI breakthrough 2024':
         Confidence: verified
         Articles: ..."
    """
    try:
        from cross_validated_search import CrossValidatedSearcher
    except ImportError:
        return "Error: cross-validated-search not installed. Run: pip install cross-validated-search"
    
    searcher = CrossValidatedSearcher()
    results = searcher.search(query, search_type="news")
    
    response_parts = [
        f"News Search: {query}",
        f"Overall Confidence: {results.confidence}",
        f"",
        "Articles:",
    ]
    
    for i, source in enumerate(results.sources[:5], 1):
        response_parts.append(f"  {i}. {source.title}")
        response_parts.append(f"     Source: {source.engine}")
        response_parts.append(f"     URL: {source.url}")
        if hasattr(source, 'date') and source.date:
            response_parts.append(f"     Date: {source.date}")
    
    return "\n".join(response_parts)


def create_agent_with_cross_validated_search():
    """Create a smolagent with cross-validated search capabilities."""
    
    # Choose which LLM engine to use
    model = InferenceClientModel()
    
    # Create agent with cross-validated search tools
    agent = CodeAgent(
        tools=[
            cross_validated_search,
            search_with_fact_check,
            search_news_with_verification,
        ],
        model=model,
        stream_outputs=True,
    )
    
    return agent


def main():
    """Run example queries with the agent."""
    print("=" * 60)
    print("Cross-Validated Search with smolagents")
    print("=" * 60)
    
    agent = create_agent_with_cross_validated_search()
    
    # Example queries
    queries = [
        "What is the latest version of Python?",
        "Fact-check: Python 3.14 is released",
        "What are the latest AI breakthroughs?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        result = agent.run(query)
        print(result)
    
    print("\n" + "=" * 60)
    print("For more information:")
    print("https://github.com/wd041216-bit/cross-validated-search")
    print("=" * 60)


if __name__ == "__main__":
    # Uncomment to run the agent
    # main()
    
    # Quick example
    print("Cross-Validated Search Tool for smolagents")
    print("=" * 40)
    print("\nInstallation:")
    print("    pip install cross-validated-search smolagents")
    print("\nUsage:")
    print("    from cross_validated_search_tool import create_agent_with_cross_validated_search")
    print("    agent = create_agent_with_cross_validated_search()")
    print("    result = agent.run('What is the latest version of Python?')")
    print("\nTools available:")
    print("    - cross_validated_search: Web search with confidence scoring")
    print("    - search_with_fact_check: Fact-checking claims")
    print("    - search_news_with_verification: News search with verification")
    print("\nConfidence Levels:")
    print("    ✅ Verified - 3+ sources agree")
    print("    🟢 Likely True - 2 sources agree")
    print("    🟡 Uncertain - Single source or conflicts")
    print("    🔴 Likely False - Major contradictions")