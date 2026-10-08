from .tools import Tool

class ArxivResearchTool(Tool):
    name = "arxiv_research_tool"
    description = "Searches arXiv for research papers and returns summaries and PDF links. Input: query (str). Output: formatted string with top results."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query (e.g., 'LLM quantization', 'Attention Is All You Need')."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the Arxiv Research tool.
        """
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        """
        Searches arXiv for research papers and returns summaries.
        Useful for finding academic sources, latest techniques, or technical details.
        
        Args:
            query: The search query (e.g., 'LLM quantization', 'Attention Is All You Need').
        """
        try:
            # Lazy Import
            import arxiv
        except ImportError:
            return "Error: Please install 'arxiv' package to use this tool."

        try:
            # 1. Construct Client (Robustness)
            client = arxiv.Client()
            
            # 2. Search Config
            search = arxiv.Search(
                query=query,
                max_results=3, # Keep context window clean (Top 3 only)
                sort_by=arxiv.SortCriterion.Relevance
            )

            results = []
            # 3. Fetch and Format
            for r in client.results(search):
                published_date = r.published.strftime("%Y-%m-%d")
                # Clean up newlines in summary to save tokens
                summary = r.summary.replace("\n", " ").strip()
                
                entry = (
                    f"\U0001F4C4 TITLE: {r.title}\n"
                    f"\U0001F4C5 DATE: {published_date}\n"
                    f"\U0001F517 PDF: {r.pdf_url}\n"
                    f"\U0001F4DD SUMMARY: {summary[:500]}..." # Truncate summary
                )
                results.append(entry)

            if not results:
                return "No papers found for this query."

            return "\n\n".join(results)

        except Exception as e:
            return f"Error searching arXiv: {str(e)}"
