from typing import Literal

from smolagents import Tool
from linkup import LinkupClient


class LinkupSearchTool(Tool):
    name = "linkup_web_search"
    description = "Performs a search for your query using Linkup sdk then returns a string of the top search results."
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "depth": {"type": "string",
                  "description": "The search depth to perform. Use 'standard' for straightforward queries with likely direct answers (e.g., facts, definitions, simple explanations). Use 'deep' for: 1) complex queries requiring comprehensive analysis or information synthesis, 2) queries containing uncommon terms, specialized jargon, or abbreviations that may need additional context, or 3) questions likely requiring up-to-date or specialized web search results to answer effectively.",
                  "nullable": True, },
    }
    output_type = "string"

    def __init__(self, answer_output_type: Literal["searchResults", "sourcedAnswer", "structured"] = "sourcedAnswer",
                 **kwargs):
        super().__init__(self)
        import os

        self.api_key = os.getenv("LINKUP_API_KEY")
        if self.api_key is None:
            raise ValueError("Missing Linkup API key. Make sure you have 'LINKUP_API_KEY' in your env variables.")

        self.client = LinkupClient(api_key=self.api_key)
        self.answer_output_type = answer_output_type

    def forward(self, query: str, depth: str = "standard") -> str:
        response = self.client.search(
            query=query,
            depth=depth,
            output_type=self.answer_output_type
        )

        answer_text = getattr(response, "answer", "No answer provided.")
        sources_obj = getattr(response, "sources", [])

        citations_list = []
        for i, source in enumerate(sources_obj, start=0):
            citation_marker = f"[{i}]"
            citations_list.append(
                {
                    "marker": citation_marker,
                    "name": source.name,
                    "url": source.url,
                    "snippet": source.snippet,
                }
            )

        if citations_list:
            citations_text = "\n".join(
                f"{c['marker']} {c['name']} ({c['url']})"
                for c in citations_list
            )
            answer_with_citations = (
                    f"{answer_text}\n\n" "References:\n" + citations_text
            )
        else:
            answer_with_citations = answer_text

        return f'## Search Results\n{answer_with_citations}'
