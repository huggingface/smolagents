#!/usr/bin/env python
# coding=utf-8

"""
Example: Web Search with Domain Filtering

This example demonstrates how to use domain filtering with smolagents web search tools
to control which websites can appear in search results.
"""

from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel


def example_blocklist():
    """Example 1: Using a blocklist to exclude unwanted domains."""
    print("=" * 80)
    print("Example 1: Blocklist - Excluding specific domains")
    print("=" * 80)

    # Block known low-quality or undesirable domains
    search_tool = DuckDuckGoSearchTool(
        max_results=5,
        blocked_domains=[
            "pinterest.com",  # Often not useful for factual information
            "*.quora.com",  # Block Quora and all subdomains
            "facebook.com",
            "twitter.com",
        ],
    )

    # Create an agent with the filtered search tool
    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nSearching with blocklist applied...")
    print("Blocked domains: pinterest.com, *.quora.com, facebook.com, twitter.com")
    print("\nNote: Results will exclude these domains\n")


def example_allowlist():
    """Example 2: Using an allowlist to only include trusted sources."""
    print("=" * 80)
    print("Example 2: Allowlist - Only trusted educational and government sources")
    print("=" * 80)

    # Only allow educational, government, and specific trusted domains
    search_tool = DuckDuckGoSearchTool(
        max_results=5,
        allowed_domains=[
            "*.edu",  # All educational institutions
            "*.gov",  # Government websites
            "*.ac.uk",  # UK academic institutions
            "wikipedia.org",  # Wikipedia
            "arxiv.org",  # Academic papers
            "nature.com",  # Scientific journal
            "science.org",  # Scientific journal
        ],
    )

    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nSearching with allowlist applied...")
    print("Only allowed: *.edu, *.gov, *.ac.uk, wikipedia.org, arxiv.org, etc.")
    print("\nNote: Only results from these trusted sources will appear\n")


def example_security_filtering():
    """Example 3: Security-focused filtering to block malicious and tracking domains."""
    print("=" * 80)
    print("Example 3: Security Filtering - Block ads, trackers, and malicious sites")
    print("=" * 80)

    # Block various categories of undesirable domains
    search_tool = DuckDuckGoSearchTool(
        max_results=5,
        blocked_domains=[
            # Ad networks
            "*.ads.com",
            "*.ads.net",
            "*.doubleclick.*",
            "*.googlesyndication.com",
            # Trackers
            "*.tracking.*",
            "*.analytics.*",
            "*.pixel.*",
            # Content farms and low-quality sites
            "*.blogspot.com",
            "*.wordpress.com",  # Can contain lots of low-quality content
            # Social media (often not primary sources)
            "*.facebook.com",
            "*.twitter.com",
            "*.instagram.com",
            "*.tiktok.com",
        ],
    )

    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nSearching with security filtering applied...")
    print("Blocking: ad networks, trackers, analytics, social media, content farms")
    print("\nNote: More secure and focused search results\n")


def example_combined_filtering():
    """Example 4: Combining allowlist and blocklist for fine-grained control."""
    print("=" * 80)
    print("Example 4: Combined Filtering - Allowlist with exceptions")
    print("=" * 80)

    # Allow educational domains but block specific ones
    search_tool = DuckDuckGoSearchTool(
        max_results=5,
        allowed_domains=[
            "*.edu",
            "*.ac.uk",
        ],
        blocked_domains=[
            "spam-university.edu",  # Block a specific bad actor
            "sketchy.ac.uk",  # Block another specific domain
        ],
    )

    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nSearching with combined filtering...")
    print("Allowed: *.edu, *.ac.uk")
    print("Blocked within allowed: spam-university.edu, sketchy.ac.uk")
    print("\nNote: Allowlist provides base filter, blocklist refines it\n")


def example_research_assistant():
    """Example 5: Practical use case - Research assistant for academic work."""
    print("=" * 80)
    print("Example 5: Research Assistant - Academic-focused search")
    print("=" * 80)

    search_tool = DuckDuckGoSearchTool(
        max_results=10,
        allowed_domains=[
            # Academic
            "*.edu",
            "*.ac.uk",
            "*.ac.jp",
            # Research repositories
            "arxiv.org",
            "scholar.google.com",
            "pubmed.ncbi.nlm.nih.gov",
            # Academic publishers
            "*.springer.com",
            "*.ieee.org",
            "nature.com",
            "science.org",
            "*.sciencedirect.com",
            # Reference
            "wikipedia.org",
            "britannica.com",
        ],
    )

    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nResearch-focused search configured!")
    print("Sources limited to: academic institutions, research repositories, publishers")
    print("\nExample query: 'What are the latest developments in quantum computing?'\n")

    # You can now use the agent for research queries
    # result = agent.run("What are the latest developments in quantum computing?")


def example_corporate_compliance():
    """Example 6: Corporate use case - Compliance with company policies."""
    print("=" * 80)
    print("Example 6: Corporate Compliance - Company-approved sources only")
    print("=" * 80)

    search_tool = DuckDuckGoSearchTool(
        max_results=5,
        allowed_domains=[
            # Company internal
            "intranet.company.com",
            "*.company.com",
            # Approved external sources
            "*.gov",
            "*.edu",
            # Industry-specific trusted sources
            "techcrunch.com",
            "wired.com",
            "arstechnica.com",
            # Documentation
            "docs.python.org",
            "developer.mozilla.org",
            "stackoverflow.com",
        ],
        blocked_domains=[
            # Even within allowed domains, block specific pages
            "blog.company.com",  # Block employee blogs
        ],
    )

    model = InferenceClientModel()
    CodeAgent(tools=[search_tool], model=model)

    print("\nCorporate-compliant search configured!")
    print("Only company-approved sources will be accessed")
    print("\nNote: Helps ensure compliance with corporate information policies\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("SMOLAGENTS DOMAIN FILTERING EXAMPLES")
    print("=" * 80 + "\n")

    examples = [
        ("Blocklist Filtering", example_blocklist),
        ("Allowlist Filtering", example_allowlist),
        ("Security Filtering", example_security_filtering),
        ("Combined Filtering", example_combined_filtering),
        ("Research Assistant", example_research_assistant),
        ("Corporate Compliance", example_corporate_compliance),
    ]

    print("This script demonstrates various domain filtering strategies:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning examples...\n")

    for _, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"Note: Example requires configuration: {e}\n")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Domain filtering in smolagents provides:

✓ BLOCKLIST: Exclude specific domains (e.g., ads, trackers, low-quality sites)
✓ ALLOWLIST: Restrict to trusted sources only (e.g., .edu, .gov)
✓ WILDCARDS: Use patterns like *.edu, *.ads.*, tracker.*
✓ SUBDOMAIN SUPPORT: Blocking example.com also blocks www.example.com
✓ COMBINED FILTERS: Use both allowlist and blocklist together
✓ SECURITY: Prevent access to malicious or inappropriate domains
✓ COMPLIANCE: Enforce organizational policies on information sources

For more information, see the smolagents documentation.
    """)


if __name__ == "__main__":
    main()
