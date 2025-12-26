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
"""Integration tests for domain filtering in search tools."""

import unittest
from unittest.mock import MagicMock, patch

from smolagents.default_tools import (
    ApiWebSearchTool,
    DuckDuckGoSearchTool,
    WebSearchTool,
)


class TestSearchToolsWithDomainFiltering(unittest.TestCase):
    """Test domain filtering integration with search tools."""

    def test_duckduckgo_with_blocklist(self):
        """Test DuckDuckGoSearchTool correctly initializes with blocked domains."""
        tool = DuckDuckGoSearchTool(max_results=5, blocked_domains=["example.com", "*.ads.*"])
        assert tool.domain_filter is not None
        assert tool.domain_filter.blocked_domains == ["example.com", "*.ads.*"]
        assert tool.domain_filter.allowed_domains == []

    def test_duckduckgo_with_allowlist(self):
        """Test DuckDuckGoSearchTool correctly initializes with allowed domains."""
        tool = DuckDuckGoSearchTool(max_results=5, allowed_domains=["*.edu", "wikipedia.org"])
        assert tool.domain_filter is not None
        assert tool.domain_filter.allowed_domains == ["*.edu", "wikipedia.org"]
        assert tool.domain_filter.blocked_domains == []

    def test_duckduckgo_with_combined_filters(self):
        """Test DuckDuckGoSearchTool with both blocklist and allowlist."""
        tool = DuckDuckGoSearchTool(allowed_domains=["*.edu"], blocked_domains=["spam.edu"])
        assert tool.domain_filter.allowed_domains == ["*.edu"]
        assert tool.domain_filter.blocked_domains == ["spam.edu"]

    def test_websearch_with_blocklist(self):
        """Test WebSearchTool correctly initializes with blocked domains."""
        tool = WebSearchTool(blocked_domains=["blocked.com"])
        assert tool.domain_filter is not None
        assert tool.domain_filter.blocked_domains == ["blocked.com"]

    def test_websearch_with_allowlist(self):
        """Test WebSearchTool correctly initializes with allowed domains."""
        tool = WebSearchTool(allowed_domains=["*.gov"])
        assert tool.domain_filter is not None
        assert tool.domain_filter.allowed_domains == ["*.gov"]

    def test_api_websearch_with_filters(self):
        """Test ApiWebSearchTool correctly initializes with domain filters."""
        tool = ApiWebSearchTool(api_key="test_key", blocked_domains=["spam.com"], allowed_domains=["*.edu"])
        assert tool.domain_filter is not None
        assert tool.domain_filter.blocked_domains == ["spam.com"]
        assert tool.domain_filter.allowed_domains == ["*.edu"]

    @patch("ddgs.DDGS")
    def test_duckduckgo_filters_results(self, mock_ddgs_class):
        """Test that DuckDuckGoSearchTool actually filters results."""
        # Mock DDGS to return fake results
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Good Site", "href": "https://allowed.com", "body": "Good content"},
            {"title": "Bad Site", "href": "https://blocked.com", "body": "Bad content"},
            {"title": "Another Good", "href": "https://other.com", "body": "More content"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        # Create tool with blocklist
        tool = DuckDuckGoSearchTool(blocked_domains=["blocked.com"])

        # Mock the rate limit enforcement
        tool._enforce_rate_limit = MagicMock()

        # Run search
        result = tool.forward("test query")

        # Verify blocked.com is not in results
        assert "blocked.com" not in result
        assert "allowed.com" in result
        assert "other.com" in result

    @patch("ddgs.DDGS")
    def test_duckduckgo_allowlist_filters_results(self, mock_ddgs_class):
        """Test that DuckDuckGoSearchTool allowlist filters results correctly."""
        # Mock DDGS to return fake results
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "EDU Site", "href": "https://mit.edu", "body": "Educational"},
            {"title": "COM Site", "href": "https://example.com", "body": "Commercial"},
            {"title": "Another EDU", "href": "https://stanford.edu", "body": "Academic"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        # Create tool with allowlist
        tool = DuckDuckGoSearchTool(allowed_domains=["*.edu"])
        tool._enforce_rate_limit = MagicMock()

        # Run search
        result = tool.forward("test query")

        # Verify only .edu domains are in results
        assert "mit.edu" in result
        assert "stanford.edu" in result
        assert "example.com" not in result

    def test_websearch_filters_by_link_key(self):
        """Test that WebSearchTool uses correct URL key for filtering."""
        tool = WebSearchTool(blocked_domains=["blocked.com"])

        # Create mock results
        mock_results = [
            {"title": "Good", "link": "https://allowed.com"},
            {"title": "Bad", "link": "https://blocked.com"},
        ]

        # Filter results
        filtered = tool.domain_filter.filter_results(mock_results, url_key="link")

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    def test_api_websearch_filters_by_url_key(self):
        """Test that ApiWebSearchTool uses correct URL key for filtering."""
        tool = ApiWebSearchTool(api_key="test_key", blocked_domains=["spam.com"])

        # Test with URL key (used by Brave Search)
        mock_results = [
            {"title": "Good", "url": "https://good.com"},
            {"title": "Spam", "url": "https://spam.com"},
        ]

        filtered = tool.domain_filter.filter_results(mock_results, url_key="url")

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    @patch("ddgs.DDGS")
    def test_no_results_after_filtering_raises_exception(self, mock_ddgs_class):
        """Test that exception is raised when all results are filtered out."""
        # Mock DDGS to return results that will all be filtered
        mock_ddgs = MagicMock()
        mock_ddgs.text.return_value = [
            {"title": "Blocked 1", "href": "https://blocked.com", "body": "Content 1"},
            {"title": "Blocked 2", "href": "https://www.blocked.com", "body": "Content 2"},
        ]
        mock_ddgs_class.return_value = mock_ddgs

        # Create tool that blocks all results
        tool = DuckDuckGoSearchTool(blocked_domains=["blocked.com"])
        tool._enforce_rate_limit = MagicMock()

        # Should raise exception when no results remain
        with self.assertRaises(Exception) as context:
            tool.forward("test query")

        assert "No results found" in str(context.exception)
        assert "domain filters" in str(context.exception)


if __name__ == "__main__":
    unittest.main()
