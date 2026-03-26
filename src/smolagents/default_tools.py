#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any

from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    BASE_PYTHON_TOOLS,
    MAX_EXECUTION_TIME_SECONDS,
    evaluate_python_code,
)
from .tools import PipelineTool, Tool


@dataclass
class PreTool:
    name: str
    inputs: dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {
            "type": "string",
            "description": "The python code to run in interpreter",
        }
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, timeout_seconds=MAX_EXECUTION_TIME_SECONDS, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES))
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(authorized_imports))
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {self.authorized_imports}."
                ),
            }
        }
        self.base_python_tools = BASE_PYTHON_TOOLS
        self.python_evaluator = evaluate_python_code
        self.timeout_seconds = timeout_seconds
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        state = {}
        output = str(
            self.python_evaluator(
                code,
                state=state,
                static_tools=self.base_python_tools,
                authorized_imports=self.authorized_imports,
                timeout_seconds=self.timeout_seconds,
            )[0]  # The second element is boolean is_final_answer
        )
        return f"Stdout:\n{str(state['_print_outputs'])}\nOutput: {output}"


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {"answer": {"type": "any", "description": "The final answer to the problem"}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {"question": {"type": "string", "description": "The question to ask the user"}}
    output_type = "string"

    def forward(self, question):
        user_input = input(f"{question} => Type your answer here:")
        return user_input


class DuckDuckGoSearchTool(Tool):
    """Web search tool that performs searches using the DuckDuckGo search engine.

    Args:
        max_results (`int`, default `10`): Maximum number of search results to return.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.
        **kwargs: Additional keyword arguments for the `DDGS` client.

    Examples:
        ```python
        >>> from smolagents import DuckDuckGoSearchTool
        >>> web_search_tool = DuckDuckGoSearchTool(max_results=5, rate_limit=2.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results."""
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, rate_limit: float | None = 1.0, **kwargs):
        super().__init__()
        self.max_results = max_results
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0
        try:
            from ddgs import DDGS
        except ImportError as e:
            raise ImportError(
                "You must install package `ddgs` to run this tool: for instance run `pip install ddgs`."
            ) from e
        self.ddgs = DDGS(**kwargs)

    def forward(self, query: str) -> str:
        self._enforce_rate_limit()
        results = self.ddgs.text(query, max_results=self.max_results)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        postprocessed_results = [f"[{result['title']}]({result['href']})\n{result['body']}" for result in results]
        return "## Search Results\n\n" + "\n\n".join(postprocessed_results)

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()


class GoogleSearchTool(Tool):
    name = "web_search"
    description = """Performs a google web search for your query then returns a string of the top search results."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi"):
        super().__init__()
        import os

        self.provider = provider
        if provider == "serpapi":
            self.organic_key = "organic_results"
            api_key_env_name = "SERPAPI_API_KEY"
        else:
            self.organic_key = "organic"
            api_key_env_name = "SERPER_API_KEY"
        self.api_key = os.getenv(api_key_env_name)
        if self.api_key is None:
            raise ValueError(f"Missing API key. Make sure you have '{api_key_env_name}' in your env variables.")

    def forward(self, query: str, filter_year: int | None = None) -> str:
        import requests

        if self.provider == "serpapi":
            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "google_domain": "google.com",
            }
            base_url = "https://serpapi.com/search.json"
        else:
            params = {
                "q": query,
                "api_key": self.api_key,
            }
            base_url = "https://google.serper.dev/search"
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise ValueError(response.json())

        if self.organic_key not in results.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")
        if len(results[self.organic_key]) == 0:
            year_filter_message = f" with filter year={filter_year}" if filter_year is not None else ""
            return f"No results found for '{query}'{year_filter_message}. Try with a more general query, or remove the year filter."

        web_snippets = []
        if self.organic_key in results:
            for idx, page in enumerate(results[self.organic_key]):
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                web_snippets.append(redacted_version)

        return "## Search Results\n" + "\n\n".join(web_snippets)


class ApiWebSearchTool(Tool):
    """Web search tool that performs API-based searches.
    By default, it uses the Brave Search API.

    This tool implements a rate limiting mechanism to ensure compliance with API usage policies.
    By default, it limits requests to 1 query per second.

    Args:
        endpoint (`str`): API endpoint URL. Defaults to Brave Search API.
        api_key (`str`): API key for authentication.
        api_key_name (`str`): Environment variable name containing the API key. Defaults to "BRAVE_API_KEY".
        headers (`dict`, *optional*): Headers for API requests.
        params (`dict`, *optional*): Parameters for API requests.
        rate_limit (`float`, default `1.0`): Maximum queries per second. Set to `None` to disable rate limiting.

    Examples:
        ```python
        >>> from smolagents import ApiWebSearchTool
        >>> web_search_tool = ApiWebSearchTool(rate_limit=50.0)
        >>> results = web_search_tool("Hugging Face")
        >>> print(results)
        ```
    """

    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, URLs, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(
        self,
        endpoint: str = "",
        api_key: str = "",
        api_key_name: str = "",
        headers: dict = None,
        params: dict = None,
        rate_limit: float | None = 1.0,
    ):
        import os

        super().__init__()
        self.endpoint = endpoint or "https://api.search.brave.com/res/v1/web/search"
        self.api_key_name = api_key_name or "BRAVE_API_KEY"
        self.api_key = api_key or os.getenv(self.api_key_name)
        self.headers = headers or {"X-Subscription-Token": self.api_key}
        self.params = params or {"count": 10}
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        import time

        # No rate limit enforced
        if not self.rate_limit:
            return

        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def forward(self, query: str) -> str:
        import requests

        self._enforce_rate_limit()
        params = {**self.params, "q": query}
        response = requests.get(self.endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        data = response.json()
        results = self.extract_results(data)
        return self.format_markdown(results)

    def extract_results(self, data: dict) -> list:
        results = []
        for result in data.get("web", {}).get("results", []):
            results.append(
                {"title": result["title"], "url": result["url"], "description": result.get("description", "")}
            )
        return results

    def format_markdown(self, results: list) -> str:
        if not results:
            return "No results found."
        return "## Search Results\n\n" + "\n\n".join(
            [
                f"{idx}. [{result['title']}]({result['url']})\n{result['description']}"
                for idx, result in enumerate(results, start=1)
            ]
        )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions."
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}
    output_type = "string"

    def __init__(self, max_results: int = 10, engine: str = "duckduckgo"):
        super().__init__()
        self.max_results = max_results
        self.engine = engine

    def forward(self, query: str) -> str:
        results = self.search(query)
        if len(results) == 0:
            raise Exception("No results found! Try a less restrictive/shorter query.")
        return self.parse_results(results)

    def search(self, query: str) -> list:
        if self.engine == "duckduckgo":
            return self.search_duckduckgo(query)
        elif self.engine == "bing":
            return self.search_bing(query)
        else:
            raise ValueError(f"Unsupported engine: {self.engine}")

    def parse_results(self, results: list) -> str:
        return "## Search Results\n\n" + "\n\n".join(
            [f"[{result['title']}]({result['link']})\n{result['description']}" for result in results]
        )

    def search_duckduckgo(self, query: str) -> list:
        import requests

        response = requests.get(
            "https://lite.duckduckgo.com/lite/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        parser = self._create_duckduckgo_parser()
        parser.feed(response.text)
        return parser.results

    def _create_duckduckgo_parser(self):
        from html.parser import HTMLParser

        class SimpleResultParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.capture_title = False
                self.capture_description = False
                self.capture_link = False

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == "a" and attrs.get("class") == "result-link":
                    self.capture_title = True
                elif tag == "td" and attrs.get("class") == "result-snippet":
                    self.capture_description = True
                elif tag == "span" and attrs.get("class") == "link-text":
                    self.capture_link = True

            def handle_endtag(self, tag):
                if tag == "a" and self.capture_title:
                    self.capture_title = False
                elif tag == "td" and self.capture_description:
                    self.capture_description = False
                elif tag == "span" and self.capture_link:
                    self.capture_link = False
                elif tag == "tr":
                    # Store current result if all parts are present
                    if {"title", "description", "link"} <= self.current.keys():
                        self.current["description"] = " ".join(self.current["description"])
                        self.results.append(self.current)
                        self.current = {}

            def handle_data(self, data):
                if self.capture_title:
                    self.current["title"] = data.strip()
                elif self.capture_description:
                    self.current.setdefault("description", [])
                    self.current["description"].append(data.strip())
                elif self.capture_link:
                    self.current["link"] = "https://" + data.strip()

        return SimpleResultParser()

    def search_bing(self, query: str) -> list:
        import xml.etree.ElementTree as ET

        import requests

        response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        results = [
            {
                "title": item.findtext("title"),
                "link": item.findtext("link"),
                "description": item.findtext("description"),
            }
            for item in items[: self.max_results]
        ]
        return results


class VisitWebpageTool(Tool):
    name = "visit_webpage"
    description = (
        "Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The url of the webpage to visit.",
        }
    }
    output_type = "string"

    def __init__(self, max_output_length: int = 40000):
        super().__init__()
        self.max_output_length = max_output_length

    def _truncate_content(self, content: str, max_length: int) -> str:
        if len(content) <= max_length:
            return content
        return (
            content[:max_length] + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        )

    def forward(self, url: str) -> str:
        try:
            import re

            import requests
            from markdownify import markdownify
            from requests.exceptions import RequestException
        except ImportError as e:
            raise ImportError(
                "You must install packages `markdownify` and `requests` to run this tool: for instance run `pip install markdownify requests`."
            ) from e
        try:
            # Send a GET request to the URL with a 20-second timeout
            response = requests.get(url, timeout=20)
            response.raise_for_status()  # Raise an exception for bad status codes

            # Convert the HTML content to Markdown
            markdown_content = markdownify(response.text).strip()

            # Remove multiple line breaks
            markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

            return self._truncate_content(markdown_content, self.max_output_length)

        except requests.exceptions.Timeout:
            return "The request timed out. Please try again later or check the URL."
        except RequestException as e:
            return f"Error fetching the webpage: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


class WikipediaSearchTool(Tool):
    """
    Search Wikipedia and return the summary or full text of the requested article, along with the page URL.

    Attributes:
        user_agent (`str`): Custom user-agent string to identify the project. This is required as per Wikipedia API policies.
            See: https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
        language (`str`, default `"en"`): Language in which to retrieve Wikipedia article.
            See: http://meta.wikimedia.org/wiki/List_of_Wikipedias
        content_type (`Literal["summary", "text"]`, default `"text"`): Type of content to fetch. Can be "summary" for a short summary or "text" for the full article.
        extract_format (`Literal["HTML", "WIKI"]`, default `"WIKI"`): Extraction format of the output. Can be `"WIKI"` or `"HTML"`.

    Example:
        ```python
        >>> from smolagents import CodeAgent, InferenceClientModel, WikipediaSearchTool
        >>> agent = CodeAgent(
        >>>     tools=[
        >>>            WikipediaSearchTool(
        >>>                user_agent="MyResearchBot (myemail@example.com)",
        >>>                language="en",
        >>>                content_type="summary",  # or "text"
        >>>                extract_format="WIKI",
        >>>            )
        >>>        ],
        >>>     model=InferenceClientModel(),
        >>> )
        >>> agent.run("Python_(programming_language)")
        ```
    """

    name = "wikipedia_search"
    description = "Searches Wikipedia and returns a summary or full text of the given topic, along with the page URL."
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic to search on Wikipedia.",
        }
    }
    output_type = "string"

    def __init__(
        self,
        user_agent: str = "Smolagents (myemail@example.com)",
        language: str = "en",
        content_type: str = "text",
        extract_format: str = "WIKI",
    ):
        super().__init__()
        try:
            import wikipediaapi
        except ImportError as e:
            raise ImportError(
                "You must install `wikipedia-api` to run this tool: for instance run `pip install wikipedia-api`"
            ) from e
        if not user_agent:
            raise ValueError("User-agent is required. Provide a meaningful identifier for your project.")

        self.user_agent = user_agent
        self.language = language
        self.content_type = content_type

        # Map string format to wikipediaapi.ExtractFormat
        extract_format_map = {
            "WIKI": wikipediaapi.ExtractFormat.WIKI,
            "HTML": wikipediaapi.ExtractFormat.HTML,
        }

        if extract_format not in extract_format_map:
            raise ValueError("Invalid extract_format. Choose between 'WIKI' or 'HTML'.")

        self.extract_format = extract_format_map[extract_format]

        self.wiki = wikipediaapi.Wikipedia(
            user_agent=self.user_agent, language=self.language, extract_format=self.extract_format
        )

    def forward(self, query: str) -> str:
        try:
            page = self.wiki.page(query)

            if not page.exists():
                return f"No Wikipedia page found for '{query}'. Try a different query."

            title = page.title
            url = page.fullurl

            if self.content_type == "summary":
                text = page.summary
            elif self.content_type == "text":
                text = page.text
            else:
                return "⚠️ Invalid `content_type`. Use either 'summary' or 'text'."

            return f"✅ **Wikipedia Page:** {title}\n\n**Content:** {text}\n\n🔗 **Read more:** {url}"

        except Exception as e:
            return f"Error fetching Wikipedia summary: {str(e)}"


class CrwScrapeTool(Tool):
    """Web scraping tool that uses a CRW server (Firecrawl-compatible API) to scrape
    a single webpage and return its content as clean markdown.

    CRW is an open-source web scraper for AI agents that provides high-quality
    content extraction with optional JavaScript rendering.

    Args:
        api_url (`str`): Base URL of the CRW server. Defaults to ``http://localhost:3000``.
        api_key (`str`, *optional*): Bearer token for authentication. Falls back to the
            ``CRW_API_KEY`` environment variable.
        only_main_content (`bool`, default `True`): Strip navigation, footer, and sidebar.
        formats (`list[str]`, *optional*): Output formats (``markdown``, ``html``,
            ``plainText``, ``links``). Defaults to ``["markdown"]``.

    Examples:
        ```python
        >>> from smolagents import CrwScrapeTool
        >>> scrape_tool = CrwScrapeTool(api_url="http://localhost:3000")
        >>> result = scrape_tool("https://example.com")
        >>> print(result)
        ```
    """

    name = "crw_scrape"
    description = (
        "Scrapes a single webpage using a CRW server and returns its content as clean markdown. "
        "CRW handles JavaScript-rendered pages, strips boilerplate (nav, footer, ads), and "
        "converts the result to clean markdown. Use this instead of visit_webpage when you need "
        "higher-quality content extraction."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The URL of the webpage to scrape.",
        },
        "css_selector": {
            "type": "string",
            "description": "Optional CSS selector to extract only matching elements.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        api_url: str = "http://localhost:3000",
        api_key: str | None = None,
        only_main_content: bool = True,
        formats: list[str] | None = None,
        max_output_length: int = 40000,
    ):
        super().__init__()
        import os

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.getenv("CRW_API_KEY")
        self.only_main_content = only_main_content
        self.formats = formats or ["markdown"]
        self.max_output_length = max_output_length

    def forward(self, url: str, css_selector: str | None = None) -> str:
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict = {
            "url": url,
            "formats": self.formats,
            "onlyMainContent": self.only_main_content,
        }
        if css_selector:
            payload["cssSelector"] = css_selector

        try:
            response = requests.post(
                f"{self.api_url}/v1/scrape",
                json=payload,
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            return "The CRW scrape request timed out. Try again or check the CRW server."
        except requests.exceptions.RequestException as e:
            return f"Error calling CRW scrape API: {e}"

        data = response.json()
        if not data.get("success"):
            return f"CRW scrape failed: {data.get('error', 'Unknown error')}"

        result_data = data.get("data", {})
        content = result_data.get("markdown") or result_data.get("html") or result_data.get("plainText") or ""

        metadata = result_data.get("metadata", {})
        title = metadata.get("title", "")
        header = f"## {title}\nSource: {url}\n\n" if title else f"Source: {url}\n\n"

        full_content = header + content
        if len(full_content) > self.max_output_length:
            full_content = (
                full_content[: self.max_output_length]
                + f"\n..._Content truncated to {self.max_output_length} characters_...\n"
            )
        return full_content


class CrwCrawlTool(Tool):
    """Web crawling tool that uses a CRW server (Firecrawl-compatible API) to crawl
    a website starting from a URL and return content from multiple pages.

    This tool starts an asynchronous crawl job, polls for completion, and returns
    the aggregated markdown content from all crawled pages.

    Args:
        api_url (`str`): Base URL of the CRW server. Defaults to ``http://localhost:3000``.
        api_key (`str`, *optional*): Bearer token for authentication. Falls back to the
            ``CRW_API_KEY`` environment variable.
        max_depth (`int`, default `2`): Maximum link-follow depth.
        max_pages (`int`, default `10`): Maximum number of pages to crawl.
        poll_interval (`float`, default `2.0`): Seconds between status checks.
        poll_timeout (`float`, default `120.0`): Maximum seconds to wait for crawl completion.

    Examples:
        ```python
        >>> from smolagents import CrwCrawlTool
        >>> crawl_tool = CrwCrawlTool(api_url="http://localhost:3000", max_pages=5)
        >>> result = crawl_tool("https://example.com")
        >>> print(result)
        ```
    """

    name = "crw_crawl"
    description = (
        "Crawls a website starting from the given URL using a CRW server and returns markdown "
        "content from multiple pages. Use this when you need to gather information from several "
        "pages of a website, not just a single page."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "The starting URL to crawl from.",
        },
        "max_pages": {
            "type": "integer",
            "description": "Maximum number of pages to crawl. Defaults to 10.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        api_url: str = "http://localhost:3000",
        api_key: str | None = None,
        max_depth: int = 2,
        max_pages: int = 10,
        poll_interval: float = 2.0,
        poll_timeout: float = 120.0,
        max_output_length: int = 40000,
    ):
        super().__init__()
        import os

        self.api_url = api_url.rstrip("/")
        self.api_key = api_key or os.getenv("CRW_API_KEY")
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.max_output_length = max_output_length

    def forward(self, url: str, max_pages: int | None = None) -> str:
        import time

        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "url": url,
            "maxDepth": self.max_depth,
            "maxPages": max_pages or self.max_pages,
            "formats": ["markdown"],
            "onlyMainContent": True,
        }

        # Start the crawl job
        try:
            response = requests.post(
                f"{self.api_url}/v1/crawl",
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Error starting CRW crawl: {e}"

        data = response.json()
        if not data.get("success"):
            return f"CRW crawl failed to start: {data.get('error', 'Unknown error')}"

        crawl_id = data["id"]

        # Poll for completion
        start_time = time.time()
        while time.time() - start_time < self.poll_timeout:
            time.sleep(self.poll_interval)
            try:
                status_response = requests.get(
                    f"{self.api_url}/v1/crawl/{crawl_id}",
                    headers=headers,
                    timeout=15,
                )
                status_response.raise_for_status()
            except requests.exceptions.RequestException as e:
                return f"Error checking crawl status: {e}"

            status_data = status_response.json()
            status = status_data.get("status")

            if status == "completed":
                return self._format_crawl_results(status_data, url)
            elif status == "failed":
                return f"CRW crawl failed for {url}."

        return f"CRW crawl timed out after {self.poll_timeout}s. Crawl ID: {crawl_id}"

    def _format_crawl_results(self, status_data: dict, start_url: str) -> str:
        pages = status_data.get("data", [])
        total = status_data.get("total", len(pages))

        parts = [f"## Crawl Results for {start_url}\nPages crawled: {total}\n"]
        for page in pages:
            metadata = page.get("metadata", {})
            title = metadata.get("title", "Untitled")
            source = metadata.get("sourceURL", "")
            markdown = page.get("markdown", "")
            parts.append(f"### {title}\nURL: {source}\n\n{markdown}\n")

        full_content = "\n---\n".join(parts)
        if len(full_content) > self.max_output_length:
            full_content = (
                full_content[: self.max_output_length]
                + f"\n..._Content truncated to {self.max_output_length} characters_...\n"
            )
        return full_content


class SpeechToTextTool(PipelineTool):
    default_checkpoint = "openai/whisper-large-v3-turbo"
    description = "This is a tool that transcribes an audio into text. It returns the transcribed text."
    name = "transcriber"
    inputs = {
        "audio": {
            "type": "audio",
            "description": "The audio to transcribe. Can be a local path, an url, or a tensor.",
        }
    }
    output_type = "string"

    def __new__(cls, *args, **kwargs):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        cls.pre_processor_class = WhisperProcessor
        cls.model_class = WhisperForConditionalGeneration
        return super().__new__(cls)

    def encode(self, audio):
        from .agent_types import AgentAudio

        audio = AgentAudio(audio).to_raw()
        return self.pre_processor(audio, return_tensors="pt")

    def forward(self, inputs):
        return self.model.generate(inputs["input_features"])

    def decode(self, outputs):
        return self.pre_processor.batch_decode(outputs, skip_special_tokens=True)[0]


TOOL_MAPPING = {
    tool_class.name: tool_class
    for tool_class in [
        PythonInterpreterTool,
        DuckDuckGoSearchTool,
        VisitWebpageTool,
    ]
}

__all__ = [
    "ApiWebSearchTool",
    "CrwCrawlTool",
    "CrwScrapeTool",
    "PythonInterpreterTool",
    "FinalAnswerTool",
    "UserInputTool",
    "WebSearchTool",
    "DuckDuckGoSearchTool",
    "GoogleSearchTool",
    "VisitWebpageTool",
    "WikipediaSearchTool",
    "SpeechToTextTool",
]
