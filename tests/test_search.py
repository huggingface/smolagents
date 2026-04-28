# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

from unittest.mock import MagicMock, patch

import pytest

from smolagents import DuckDuckGoSearchTool, TavilySearchTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


def _make_tavily_response(
    query="q",
    answer="",
    results=None,
    *,
    images=None,
    response_time=0.5,
    **extra,
):
    payload = {
        "query": query,
        "answer": answer,
        "images": [] if images is None else images,
        "results": [] if results is None else results,
        "response_time": response_time,
    }
    payload.update(extra)
    return payload


def _make_tavily_client(response):
    """Build a fake Tavily client whose `.search()` returns ``response``."""
    client = MagicMock()
    client.search.return_value = response
    return client


class TestTavilySearchTool:
    """Tavily tests mirror the mocked-client style used for other search integrations."""

    def _build_tool(self, mock_client_cls, response, env=None, **tool_kwargs):
        fake_instance = _make_tavily_client(response)
        mock_client_cls.return_value = fake_instance
        env = env or {"TAVILY_API_KEY": "tvly-test"}
        with patch.dict("os.environ", env, clear=False):
            tool = TavilySearchTool(**tool_kwargs)
        return tool, fake_instance

    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            TavilySearchTool()

    @patch("tavily.TavilyClient")
    def test_forward_formats_results(self, mock_client_cls):
        response = _make_tavily_response(
            query="What is Hugging Face?",
            answer="Hugging Face builds open-source AI tooling.",
            results=[
                {
                    "title": "Hugging Face",
                    "url": "https://huggingface.co",
                    "content": "Open-source AI models, datasets, and apps.",
                }
            ],
            response_time=1.23,
        )
        tool, _ = self._build_tool(
            mock_client_cls,
            response,
            max_results=3,
            search_depth="advanced",
            chunks_per_source=2,
            topic="news",
            time_range="week",
            start_date="2025-01-01",
            end_date="2025-12-31",
            include_answer="advanced",
            include_raw_content="markdown",
            include_images=True,
            include_image_descriptions=True,
            include_favicon=True,
            include_domains=["huggingface.co"],
            exclude_domains=["example.com"],
            country="united states",
            auto_parameters=True,
            exact_match=True,
            include_usage=True,
            safe_search=True,
        )

        result = tool("What is Hugging Face?")

        mock_client_cls.assert_called_once_with(api_key="tvly-test", client_source="smolagents")
        mock_client_cls.return_value.search.assert_called_once_with(
            "What is Hugging Face?",
            max_results=3,
            search_depth="advanced",
            chunks_per_source=2,
            topic="news",
            time_range="week",
            start_date="2025-01-01",
            end_date="2025-12-31",
            include_answer="advanced",
            include_raw_content="markdown",
            include_images=True,
            include_image_descriptions=True,
            include_favicon=True,
            include_domains=["huggingface.co"],
            exclude_domains=["example.com"],
            country="united states",
            auto_parameters=True,
            exact_match=True,
            include_usage=True,
            safe_search=True,
        )
        assert result["query"] == "What is Hugging Face?"
        assert result["answer"] == "Hugging Face builds open-source AI tooling."
        assert result["results"][0]["url"] == "https://huggingface.co"
        assert result["results"][0]["content"] == "Open-source AI models, datasets, and apps."

    @patch("tavily.TavilyClient")
    def test_client_source_from_argument(self, mock_client_cls):
        tool, _ = self._build_tool(
            mock_client_cls,
            {"query": "q", "results": []},
            client_source="my_integration",
        )
        tool("q")

        mock_client_cls.assert_called_once_with(api_key="tvly-test", client_source="my_integration")

    @patch("tavily.TavilyClient")
    def test_client_source_from_env(self, mock_client_cls):
        tool, _ = self._build_tool(
            mock_client_cls,
            {"query": "q", "results": []},
            env={"TAVILY_API_KEY": "tvly-test", "TAVILY_CLIENT_SOURCE": "from_env"},
        )
        tool("q")

        mock_client_cls.assert_called_once_with(api_key="tvly-test", client_source="from_env")

    @patch("tavily.TavilyClient")
    def test_forward_handles_empty_results(self, mock_client_cls):
        response = _make_tavily_response(
            query="No results expected",
            answer="",
            results=[],
            response_time=0.5,
        )
        tool, _ = self._build_tool(mock_client_cls, response)

        result = tool("No results expected")

        assert result["results"] == []


class TestDuckDuckGoSearchTool(ToolTesterMixin):
    def setup_method(self):
        self.tool = DuckDuckGoSearchTool()
        self.tool.setup()

    @require_run_all
    def test_exact_match_arg(self):
        result = self.tool("Agents")
        assert isinstance(result, str)

    @require_run_all
    def test_agent_type_output(self):
        super().test_agent_type_output()
