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
        elif self.engine == "exa":
            return self.search_exa(query)
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

    def search_exa(self, query: str) -> list:
        """Search using the Exa API. Requires an EXA_API_KEY environment variable."""
        import os

        import requests

        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError("Missing API key. Make sure you have 'EXA_API_KEY' in your env variables.")

        response = requests.post(
            "https://api.exa.ai/search",
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "x-exa-integration": "smolagents",
            },
            json={
                "query": query,
                "numResults": self.max_results,
                "contents": {"highlights": True},
            },
            timeout=getattr(self, "timeout", 30),
        )
        response.raise_for_status()
        data = response.json()
        return [
            {
                "title": result.get("title", ""),
                "link": result["url"],
                "description": " ".join(result.get("highlights") or []),
            }
            for result in data.get("results", [])
        ]


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


class BilibiliHotTool(Tool):
    """Fetch the top trending videos from Bilibili (a major Chinese video platform).

    The tool wraps Bilibili's public "popular" endpoint, which requires no
    authentication and returns a curated list of currently trending videos.
    Results are formatted as markdown and include the title, UP master
    (creator), and engagement metrics (view / danmaku / like counts).

    Args:
        rate_limit (`float`, default `1.0`): Maximum queries per second.
            Set to `None` to disable rate limiting.

    Examples:
        ```python
        >>> from smolagents import BilibiliHotTool
        >>> tool = BilibiliHotTool()
        >>> print(tool(limit=5))
        ```
    """

    name = "bilibili_hot"
    description = (
        "Fetches the top trending videos from Bilibili's public 'popular' list. "
        "Use this when you need to know what Chinese-language videos are popular right now "
        "on Bilibili (a major Chinese video platform). Returns a markdown list of titles, "
        "creators (UP masters), and engagement metrics such as view, danmaku, and like counts."
    )
    inputs = {
        "limit": {
            "type": "integer",
            "description": (
                "Number of top videos to return, between 1 and 50. Default 10. "
                "Values outside this range are clamped to the nearest bound."
            ),
            "nullable": True,
        }
    }
    output_type = "string"

    DEFAULT_LIMIT = 10
    MIN_LIMIT = 1
    MAX_LIMIT = 50
    ENDPOINT = "https://api.bilibili.com/x/web-interface/popular"
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, rate_limit: float | None = 1.0):
        super().__init__()
        self.rate_limit = rate_limit
        self._min_interval = 1.0 / rate_limit if rate_limit else 0.0
        self._last_request_time = 0.0

    def _enforce_rate_limit(self) -> None:
        import time

        if not self.rate_limit:
            return
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    @staticmethod
    def _format_count(n) -> str:
        if not isinstance(n, (int, float)):
            return "0"
        n = int(n)
        if n >= 100_000_000:
            return f"{n / 100_000_000:.1f}亿"
        if n >= 10_000:
            return f"{n / 10_000:.1f}万"
        return str(n)

    @classmethod
    def _clamp_limit(cls, limit: int | None) -> int:
        if limit is None:
            return cls.DEFAULT_LIMIT
        return max(cls.MIN_LIMIT, min(int(limit), cls.MAX_LIMIT))

    def forward(self, limit: int | None = None) -> str:
        try:
            import requests
        except ImportError as e:
            raise ImportError(
                "You must install `requests` to run this tool: for instance run `pip install requests`."
            ) from e

        clamped = self._clamp_limit(limit)
        params = {"ps": clamped, "pn": 1}
        headers = {"User-Agent": self.USER_AGENT}

        try:
            self._enforce_rate_limit()
            response = requests.get(self.ENDPOINT, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.Timeout:
            return "Error: request to Bilibili timed out. Please try again later."
        except requests.exceptions.HTTPError:
            return f"Error: Bilibili returned HTTP {response.status_code}."
        except requests.exceptions.RequestException as e:
            return f"Error fetching Bilibili popular videos: {str(e)}"
        except ValueError:
            return "Error: Bilibili returned a non-JSON response."

        if not isinstance(payload, dict) or payload.get("code") != 0:
            message = (payload or {}).get("message") if isinstance(payload, dict) else None
            code = (payload or {}).get("code") if isinstance(payload, dict) else None
            return f"Error: Bilibili API returned code={code}: {message or 'unknown error'}."

        data = payload.get("data") or {}
        items = data.get("list") or []
        if not items:
            return "No trending videos found on Bilibili right now."

        lines = [f"## Bilibili 热门视频 Top {len(items)}", ""]
        for idx, item in enumerate(items, start=1):
            title = item.get("title") or "(untitled)"
            bvid = item.get("bvid") or ""
            url = (
                item.get("short_link_v2")
                or (f"https://www.bilibili.com/video/{bvid}" if bvid else "")
            )
            owner = item.get("owner") or {}
            up_name = owner.get("name") or "unknown"
            tname = item.get("tname") or ""
            stat = item.get("stat") or {}
            view = self._format_count(stat.get("view"))
            danmaku = self._format_count(stat.get("danmaku"))
            like = self._format_count(stat.get("like"))
            category = f" [{tname}]" if tname else ""
            lines.append(
                f"{idx}. [{title}]({url}){category}\n"
                f"   UP主: {up_name} | 播放: {view} | 弹幕: {danmaku} | 点赞: {like}"
            )
        return "\n\n".join(lines)


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
    "BilibiliHotTool",
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
