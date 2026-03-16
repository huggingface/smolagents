import os
from unittest.mock import MagicMock, patch

import pytest

from smolagents.default_tools import TOOL_MAPPING, PerplexitySearchTool


class TestPerplexitySearchToolInit:
    def test_init_with_explicit_api_key(self):
        tool = PerplexitySearchTool(api_key="test-key-123")
        assert tool.api_key == "test-key-123"

    def test_init_from_env_var(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key-456")
        tool = PerplexitySearchTool()
        assert tool.api_key == "env-key-456"

    def test_init_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Missing API key"):
            PerplexitySearchTool()

    def test_explicit_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_KEY", "env-key")
        tool = PerplexitySearchTool(api_key="explicit-key")
        assert tool.api_key == "explicit-key"


class TestPerplexitySearchToolMapping:
    def test_perplexity_search_in_mapping(self):
        assert "perplexity_search" in TOOL_MAPPING
        assert TOOL_MAPPING["perplexity_search"] is PerplexitySearchTool


class TestPerplexitySearchToolForward:
    @patch("httpx.post")
    def test_forward_returns_formatted_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "id": "abc-123",
            "results": [
                {
                    "title": "First Result",
                    "url": "https://example.com/1",
                    "snippet": "This is the first result.",
                    "date": "2024-12-15",
                },
                {
                    "title": "Second Result",
                    "url": "https://example.com/2",
                    "snippet": "This is the second result.",
                    "date": "2024-12-16",
                },
            ],
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = PerplexitySearchTool(api_key="test-key")
        result = tool.forward("test query", max_results=2)

        mock_post.assert_called_once_with(
            "https://api.perplexity.ai/search",
            json={"query": "test query", "max_results": 2},
            headers={
                "Authorization": "Bearer test-key",
                "Content-Type": "application/json",
                "X-Source": "smolagents",
            },
            timeout=30,
        )
        assert "[1] First Result" in result
        assert "URL: https://example.com/1" in result
        assert "This is the first result." in result
        assert "[2] Second Result" in result
        assert "URL: https://example.com/2" in result
        assert "This is the second result." in result

    @patch("httpx.post")
    def test_forward_empty_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "abc-123", "results": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = PerplexitySearchTool(api_key="test-key")
        result = tool.forward("empty query")

        assert result == "No results found."

    @patch("httpx.post")
    def test_forward_default_max_results(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "abc-123", "results": []}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = PerplexitySearchTool(api_key="test-key")
        tool.forward("test query")

        _, kwargs = mock_post.call_args
        assert kwargs["json"]["max_results"] == 5

    @patch("httpx.post")
    def test_forward_raises_on_http_error(self, mock_post):
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock()
        )
        mock_post.return_value = mock_response

        tool = PerplexitySearchTool(api_key="test-key")
        with pytest.raises(httpx.HTTPStatusError):
            tool.forward("test query")
