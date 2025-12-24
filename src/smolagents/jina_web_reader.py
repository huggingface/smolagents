import requests
from .tools import Tool

class JinaWebReaderTool(Tool):
    name = "jina_web_reader"
    description = "Reads a webpage and converts it to clean Markdown using Jina AI's Reader. Handles JavaScript-heavy sites."
    inputs = {
        "url": {"type": "string", "description": "The URL to visit (e.g., 'https://example.com')."}
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, url: str) -> str:
        """
        Reads a webpage and converts it to clean Markdown using Jina AI's Reader.
        This tool handles JavaScript-heavy sites (React, Next.js) better than standard scrapers.

        Args:
            url: The URL to visit (e.g., 'https://example.com').
        """
        # Prefixing with r.jina.ai triggers their free LLM-friendly extraction service
        jina_url = f"https://r.jina.ai/{url}"
        
        try:
            # We use a simple GET request. Jina handles the complex rendering.
            response = requests.get(jina_url, timeout=20)
            response.raise_for_status()
            
            content = response.text
            
            # Safety Check: If content is empty or an error message
            if not content or "Error" in content[:50]:
                 return f"Could not read content from {url}. Status: {response.status_code}"

            return content

        except requests.exceptions.Timeout:
            return f"Error: The request to {url} timed out."
        except requests.exceptions.RequestException as e:
            return f"Error fetching the page: {str(e)}"
