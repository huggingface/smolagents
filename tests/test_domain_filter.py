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
"""Tests for domain filtering functionality."""

import unittest

from smolagents.domain_filter import DomainFilter


class TestDomainFilter(unittest.TestCase):
    """Test cases for DomainFilter class."""

    def test_no_filters(self):
        """Test that all URLs are allowed when no allowlist is specified."""
        filter = DomainFilter()
        assert filter.is_allowed("https://example.com")
        assert filter.is_allowed("https://test.org")
        assert filter.is_allowed("https://anything.net")

    def test_allowlist_exact_match(self):
        """Test exact domain allowlist."""
        filter = DomainFilter(allowed_domains=["trusted.com", "safe.org"])
        assert filter.is_allowed("https://trusted.com")
        assert filter.is_allowed("https://safe.org")
        assert not filter.is_allowed("https://untrusted.com")
        assert not filter.is_allowed("https://example.net")

    def test_allowlist_subdomain(self):
        """Test that allowing a domain also allows its subdomains."""
        filter = DomainFilter(allowed_domains=["example.com"])
        assert filter.is_allowed("https://example.com")
        assert filter.is_allowed("https://www.example.com")
        assert filter.is_allowed("https://api.example.com")
        assert filter.is_allowed("https://sub.domain.example.com")
        assert not filter.is_allowed("https://other.org")
        assert not filter.is_allowed("https://notexample.com")

    def test_allowlist_wildcard_subdomain(self):
        """Test wildcard patterns for subdomains."""
        filter = DomainFilter(allowed_domains=["*.edu", "*.gov"])
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://stanford.edu")
        assert filter.is_allowed("https://whitehouse.gov")
        assert filter.is_allowed("https://nasa.gov")
        assert not filter.is_allowed("https://example.com")
        assert not filter.is_allowed("https://company.net")

    def test_allowlist_wildcard_tld(self):
        """Test wildcard patterns for TLDs."""
        filter = DomainFilter(allowed_domains=["trusted.*"])
        assert filter.is_allowed("https://trusted.com")
        assert filter.is_allowed("https://trusted.net")
        assert filter.is_allowed("https://trusted.org")
        assert not filter.is_allowed("https://example.com")

    def test_allowlist_multiple_wildcards(self):
        """Test patterns with multiple wildcards."""
        filter = DomainFilter(allowed_domains=["*.edu", "*.gov", "*.ac.uk"])
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://nasa.gov")
        assert filter.is_allowed("https://ox.ac.uk")
        assert not filter.is_allowed("https://example.com")

    def test_case_insensitive(self):
        """Test that domain matching is case-insensitive."""
        filter = DomainFilter(allowed_domains=["Example.COM"])
        assert filter.is_allowed("https://example.com")
        assert filter.is_allowed("https://EXAMPLE.COM")
        assert filter.is_allowed("https://Example.Com")
        assert filter.is_allowed("https://www.example.com")

    def test_url_with_path(self):
        """Test URLs with paths and query parameters."""
        filter = DomainFilter(allowed_domains=["allowed.com"])
        assert filter.is_allowed("https://allowed.com/path/to/page")
        assert filter.is_allowed("https://allowed.com/page?query=test")
        assert filter.is_allowed("https://allowed.com/path#anchor")
        assert not filter.is_allowed("https://notallowed.com/path/to/page")

    def test_url_with_port(self):
        """Test URLs with port numbers."""
        filter = DomainFilter(allowed_domains=["example.com"])
        assert filter.is_allowed("https://example.com:8080")
        assert filter.is_allowed("http://example.com:3000/api")
        assert not filter.is_allowed("https://other.com:8080")

    def test_url_without_protocol(self):
        """Test URLs without explicit protocol."""
        filter = DomainFilter(allowed_domains=["allowed.com"])
        assert filter.is_allowed("allowed.com")
        assert filter.is_allowed("www.allowed.com")
        assert not filter.is_allowed("notallowed.com")

    def test_invalid_urls(self):
        """Test handling of invalid URLs."""
        filter = DomainFilter(allowed_domains=["example.com"])
        assert not filter.is_allowed("")
        assert not filter.is_allowed("not-a-url")
        assert not filter.is_allowed("://invalid")

    def test_filter_results_with_url_key(self):
        """Test filtering a list of results with default 'url' key."""
        filter = DomainFilter(allowed_domains=["allowed.com", "other.org"])
        results = [
            {"title": "Good 1", "url": "https://allowed.com/1"},
            {"title": "Bad", "url": "https://blocked.com/bad"},
            {"title": "Good 2", "url": "https://other.org/2"},
        ]
        filtered = filter.filter_results(results)
        assert len(filtered) == 2
        assert filtered[0]["title"] == "Good 1"
        assert filtered[1]["title"] == "Good 2"

    def test_filter_results_with_custom_key(self):
        """Test filtering with custom URL key name."""
        filter = DomainFilter(allowed_domains=["allowed.com"])
        results = [
            {"title": "Good", "link": "https://allowed.com"},
            {"title": "Bad", "link": "https://blocked.com"},
        ]
        filtered = filter.filter_results(results, url_key="link")
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    def test_filter_results_with_href_key(self):
        """Test filtering with 'href' key (used by DuckDuckGo)."""
        filter = DomainFilter(allowed_domains=["*.edu"])
        results = [
            {"title": "MIT", "href": "https://mit.edu"},
            {"title": "Example", "href": "https://example.com"},
            {"title": "Stanford", "href": "https://stanford.edu"},
        ]
        filtered = filter.filter_results(results, url_key="href")
        assert len(filtered) == 2
        assert filtered[0]["title"] == "MIT"
        assert filtered[1]["title"] == "Stanford"

    def test_filter_results_empty_list(self):
        """Test filtering an empty results list."""
        filter = DomainFilter(allowed_domains=["example.com"])
        filtered = filter.filter_results([])
        assert len(filtered) == 0

    def test_filter_results_all_filtered(self):
        """Test when all results are filtered out."""
        filter = DomainFilter(allowed_domains=["example.com"])
        results = [
            {"title": "Page 1", "url": "https://other.com/1"},
            {"title": "Page 2", "url": "https://another.com/2"},
        ]
        filtered = filter.filter_results(results)
        assert len(filtered) == 0

    def test_filter_results_missing_url(self):
        """Test results with missing URL keys."""
        filter = DomainFilter(allowed_domains=["allowed.com"])
        results = [
            {"title": "Good", "url": "https://allowed.com"},
            {"title": "No URL"},  # Missing URL key
            {"title": "Bad", "url": "https://blocked.com"},
        ]
        filtered = filter.filter_results(results)
        assert len(filtered) == 1
        assert filtered[0]["title"] == "Good"

    def test_complex_wildcard_patterns(self):
        """Test complex wildcard patterns."""
        filter = DomainFilter(allowed_domains=["*.edu", "*.gov", "*.ac.uk", "wikipedia.org"])
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://nasa.gov")
        assert filter.is_allowed("https://ox.ac.uk")
        assert filter.is_allowed("https://wikipedia.org")
        assert filter.is_allowed("https://en.wikipedia.org")
        assert not filter.is_allowed("https://example.com")
        assert not filter.is_allowed("https://company.net")

    def test_edge_case_single_char_domain(self):
        """Test edge case with single character domain parts."""
        filter = DomainFilter(allowed_domains=["a.b.c"])
        assert filter.is_allowed("https://a.b.c")
        assert filter.is_allowed("https://x.a.b.c")
        assert not filter.is_allowed("https://a.b.d")

    def test_repr(self):
        """Test string representation of DomainFilter."""
        filter = DomainFilter(allowed_domains=["*.edu"])
        repr_str = repr(filter)
        assert "*.edu" in repr_str
        assert "DomainFilter" in repr_str

    def test_practical_research_scenario(self):
        """Test a practical academic/research filtering scenario."""
        # Only allow trusted educational and government sources
        filter = DomainFilter(
            allowed_domains=[
                "*.edu",
                "*.gov",
                "wikipedia.org",
                "arxiv.org",
                "*.ac.uk",  # UK academic institutions
            ]
        )

        # These should be allowed
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://stanford.edu")
        assert filter.is_allowed("https://nasa.gov")
        assert filter.is_allowed("https://wikipedia.org")
        assert filter.is_allowed("https://en.wikipedia.org")
        assert filter.is_allowed("https://arxiv.org")
        assert filter.is_allowed("https://ox.ac.uk")
        assert filter.is_allowed("https://cam.ac.uk")

        # These should be blocked
        assert not filter.is_allowed("https://random-blog.com")
        assert not filter.is_allowed("https://commercial-site.net")

    def test_practical_corporate_scenario(self):
        """Test a practical corporate/trusted sources scenario."""
        # Allow approved sources for company use
        filter = DomainFilter(
            allowed_domains=[
                "*.company.com",
                "*.edu",
                "*.gov",
                "github.com",
                "stackoverflow.com",
                "python.org",
            ]
        )

        # These should be allowed
        assert filter.is_allowed("https://docs.company.com")
        assert filter.is_allowed("https://mit.edu/research")
        assert filter.is_allowed("https://github.com/repo")
        assert filter.is_allowed("https://stackoverflow.com/questions")

        # These should be blocked
        assert not filter.is_allowed("https://random-blog.com")
        assert not filter.is_allowed("https://social-media.net")


if __name__ == "__main__":
    unittest.main()
