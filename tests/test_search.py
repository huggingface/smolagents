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

from smolagents import DuckDuckGoSearchTool, PerplexitySearchTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


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


def _mock_post(payload):
    """Build a MagicMock that mimics `requests.post(...).json()` returning the given payload."""
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    post = MagicMock(return_value=response)
    return post, response


class TestPerplexitySearchTool:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("PPLX_API_KEY", raising=False)
        with pytest.raises(ValueError, match="PERPLEXITY_API_KEY"):
            PerplexitySearchTool()

    def test_pplx_api_key_fallback(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.setenv("PPLX_API_KEY", "fallback-key")
        tool = PerplexitySearchTool()
        assert tool.api_key == "fallback-key"

    def test_invalid_recency_filter_raises(self):
        with pytest.raises(ValueError, match="Invalid `search_recency_filter`"):
            PerplexitySearchTool(api_key="test-key", search_recency_filter="bogus")

    def test_mixed_domain_filter_raises(self):
        with pytest.raises(ValueError, match="cannot mix allowlist and denylist"):
            PerplexitySearchTool(api_key="test-key", search_domain_filter=["nytimes.com", "-pinterest.com"])

    def test_no_results_returns_empty_message(self):
        tool = PerplexitySearchTool(api_key="test-key")
        post, _ = _mock_post({"results": []})
        with patch("requests.post", post):
            out = tool("anything")
        assert "No results found" in out

    def test_formats_results(self):
        tool = PerplexitySearchTool(api_key="test-key")
        payload = {
            "results": [
                {
                    "title": "First",
                    "url": "https://a.example/",
                    "snippet": "alpha snippet",
                    "date": "2026-04-01",
                },
                {"title": "Second", "url": "https://b.example/", "snippet": "beta snippet"},
            ]
        }
        post, _ = _mock_post(payload)
        with patch("requests.post", post):
            out = tool("hello")
        assert "## Search Results" in out
        assert "1. [First](https://a.example/)" in out
        assert "Published: 2026-04-01" in out
        assert "alpha snippet" in out
        assert "2. [Second](https://b.example/)" in out
        assert "beta snippet" in out

    def test_request_authorization_header(self):
        tool = PerplexitySearchTool(api_key="secret")
        post, _ = _mock_post({"results": []})
        with patch("requests.post", post):
            tool("hello")
        kwargs = post.call_args.kwargs
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["headers"]["Content-Type"] == "application/json"

    def test_filter_passthrough(self):
        tool = PerplexitySearchTool(
            api_key="test-key",
            max_results=3,
            search_domain_filter=["nytimes.com"],
            search_recency_filter="week",
            search_after_date_filter="1/1/2026",
            search_before_date_filter="12/31/2026",
            max_tokens_per_page=512,
        )
        post, _ = _mock_post({"results": []})
        with patch("requests.post", post):
            tool("hello")
        body = post.call_args.kwargs["json"]
        assert body == {
            "query": "hello",
            "max_results": 3,
            "search_domain_filter": ["nytimes.com"],
            "search_recency_filter": "week",
            "search_after_date_filter": "1/1/2026",
            "search_before_date_filter": "12/31/2026",
            "max_tokens_per_page": 512,
        }

    def test_omits_unset_filters(self):
        tool = PerplexitySearchTool(api_key="test-key")
        post, _ = _mock_post({"results": []})
        with patch("requests.post", post):
            tool("hello")
        body = post.call_args.kwargs["json"]
        assert set(body) == {"query", "max_results"}
        assert body["query"] == "hello"
        assert body["max_results"] == 5

    def test_endpoint_default_and_override(self):
        default_tool = PerplexitySearchTool(api_key="test-key")
        assert default_tool.endpoint == "https://api.perplexity.ai/search"

        custom_tool = PerplexitySearchTool(api_key="test-key", endpoint="https://example.com/search")
        post, _ = _mock_post({"results": []})
        with patch("requests.post", post):
            custom_tool("hello")
        assert post.call_args.args[0] == "https://example.com/search"

    def test_raise_for_status_propagates_errors(self):
        tool = PerplexitySearchTool(api_key="test-key")
        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("boom")
        with patch("requests.post", return_value=response):
            with pytest.raises(RuntimeError, match="boom"):
                tool("hello")
