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
        """Test that all URLs are allowed when no filters are specified."""
        filter = DomainFilter()
        assert filter.is_allowed("https://example.com")
        assert filter.is_allowed("https://test.org")
        assert filter.is_allowed("https://anything.net")

    def test_blocklist_exact_match(self):
        """Test exact domain blocking."""
        filter = DomainFilter(blocked_domains=["example.com", "blocked.net"])
        assert not filter.is_allowed("https://example.com")
        assert not filter.is_allowed("https://blocked.net")
        assert filter.is_allowed("https://allowed.com")

    def test_blocklist_subdomain(self):
        """Test that blocking a domain also blocks its subdomains."""
        filter = DomainFilter(blocked_domains=["example.com"])
        assert not filter.is_allowed("https://example.com")
        assert not filter.is_allowed("https://www.example.com")
        assert not filter.is_allowed("https://api.example.com")
        assert not filter.is_allowed("https://sub.domain.example.com")
        assert filter.is_allowed("https://example.org")
        assert filter.is_allowed("https://notexample.com")

    def test_blocklist_wildcard_subdomain(self):
        """Test wildcard patterns for subdomains."""
        filter = DomainFilter(blocked_domains=["*.ads.com", "tracker.*"])
        assert not filter.is_allowed("https://banner.ads.com")
        assert not filter.is_allowed("https://popup.ads.com")
        # Wildcard matches one or more chars, so base domain without subdomain is allowed
        assert filter.is_allowed("https://ads.com")
        # tracker.* matches tracker.X where X is a TLD (single component)
        assert not filter.is_allowed("https://tracker.com")
        assert not filter.is_allowed("https://tracker.net")
        assert not filter.is_allowed("https://tracker.org")
        # And since we block subdomains of blocked domains, sub.tracker.com is also blocked
        assert not filter.is_allowed("https://sub.tracker.com")
        assert filter.is_allowed("https://example.com")

    def test_blocklist_wildcard_tld(self):
        """Test wildcard patterns for TLDs."""
        filter = DomainFilter(blocked_domains=["spam.*"])
        assert not filter.is_allowed("https://spam.com")
        assert not filter.is_allowed("https://spam.net")
        assert not filter.is_allowed("https://spam.org")
        assert filter.is_allowed("https://example.com")

    def test_blocklist_multiple_wildcards(self):
        """Test patterns with multiple wildcards."""
        filter = DomainFilter(blocked_domains=["*.ads.*", "*.tracker.*"])
        assert not filter.is_allowed("https://banner.ads.com")
        assert not filter.is_allowed("https://popup.ads.net")
        assert not filter.is_allowed("https://script.tracker.org")
        assert filter.is_allowed("https://legitimate.site.com")

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
        assert not filter.is_allowed("https://other.org")

    def test_allowlist_wildcard(self):
        """Test wildcard patterns in allowlist."""
        filter = DomainFilter(allowed_domains=["*.edu", "*.gov", "wikipedia.org"])
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://stanford.edu")
        assert filter.is_allowed("https://whitehouse.gov")
        assert filter.is_allowed("https://wikipedia.org")
        assert filter.is_allowed("https://en.wikipedia.org")
        assert not filter.is_allowed("https://example.com")

    def test_allowlist_takes_precedence(self):
        """Test that allowlist takes precedence over blocklist."""
        filter = DomainFilter(allowed_domains=["*.edu"], blocked_domains=["bad.edu"])
        # Only .edu domains are allowed
        assert filter.is_allowed("https://mit.edu")
        assert filter.is_allowed("https://stanford.edu")
        # bad.edu is in blocklist, so it should be blocked
        assert not filter.is_allowed("https://bad.edu")
        # Non-.edu domains are not allowed (allowlist restriction)
        assert not filter.is_allowed("https://example.com")

    def test_case_insensitive(self):
        """Test that domain matching is case-insensitive."""
        filter = DomainFilter(blocked_domains=["Example.COM"])
        assert not filter.is_allowed("https://example.com")
        assert not filter.is_allowed("https://EXAMPLE.COM")
        assert not filter.is_allowed("https://Example.Com")
        assert not filter.is_allowed("https://WWW.EXAMPLE.COM")

    def test_url_with_path(self):
        """Test URLs with paths and query parameters."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
        assert not filter.is_allowed("https://blocked.com/path/to/page")
        assert not filter.is_allowed("https://blocked.com/page?query=test")
        assert not filter.is_allowed("https://blocked.com/path#anchor")
        assert filter.is_allowed("https://allowed.com/path/to/page")

    def test_url_with_port(self):
        """Test URLs with port numbers."""
        filter = DomainFilter(blocked_domains=["localhost"])
        assert not filter.is_allowed("https://localhost:8080")
        assert not filter.is_allowed("http://localhost:3000/api")
        assert filter.is_allowed("https://example.com:8080")

    def test_url_without_protocol(self):
        """Test URLs without explicit protocol."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
        assert not filter.is_allowed("blocked.com")
        assert not filter.is_allowed("www.blocked.com")
        assert filter.is_allowed("allowed.com")

    def test_invalid_urls(self):
        """Test handling of invalid URLs."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
        assert not filter.is_allowed("")
        assert not filter.is_allowed("not-a-url")
        assert not filter.is_allowed("://invalid")

    def test_filter_results_with_url_key(self):
        """Test filtering a list of results with default 'url' key."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
        results = [
            {"title": "Good 1", "url": "https://allowed.com/1"},
            {"title": "Bad", "url": "https://blocked.com/bad"},
            {"title": "Good 2", "url": "https://allowed.org/2"},
        ]
        filtered = filter.filter_results(results)
        assert len(filtered) == 2
        assert filtered[0]["title"] == "Good 1"
        assert filtered[1]["title"] == "Good 2"

    def test_filter_results_with_custom_key(self):
        """Test filtering with custom URL key name."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
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
        filter = DomainFilter(blocked_domains=["blocked.com"])
        filtered = filter.filter_results([])
        assert len(filtered) == 0

    def test_filter_results_all_blocked(self):
        """Test when all results are blocked."""
        filter = DomainFilter(blocked_domains=["example.com"])
        results = [
            {"title": "Page 1", "url": "https://example.com/1"},
            {"title": "Page 2", "url": "https://www.example.com/2"},
        ]
        filtered = filter.filter_results(results)
        assert len(filtered) == 0

    def test_filter_results_missing_url(self):
        """Test results with missing URL keys."""
        filter = DomainFilter(blocked_domains=["blocked.com"])
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
        filter = DomainFilter(
            blocked_domains=[
                "*.ad.com",
                "*.ad.net",
                "*.tracking.com",
                "*.analytics.*",  # Block subdomains of analytics.X
            ]
        )
        assert not filter.is_allowed("https://banner.ad.com")
        assert not filter.is_allowed("https://popup.ad.net")
        assert not filter.is_allowed("https://pixel.tracking.com")
        assert not filter.is_allowed("https://cdn.analytics.com")
        assert not filter.is_allowed("https://api.analytics.net")
        assert filter.is_allowed("https://legitimate.website.com")

    def test_edge_case_single_char_domain(self):
        """Test edge case with single character domain parts."""
        filter = DomainFilter(blocked_domains=["a.b.c"])
        assert not filter.is_allowed("https://a.b.c")
        assert not filter.is_allowed("https://x.a.b.c")
        assert filter.is_allowed("https://a.b.d")

    def test_repr(self):
        """Test string representation of DomainFilter."""
        filter = DomainFilter(blocked_domains=["blocked.com"], allowed_domains=["*.edu"])
        repr_str = repr(filter)
        assert "blocked.com" in repr_str
        assert "*.edu" in repr_str
        assert "DomainFilter" in repr_str

    def test_practical_security_scenario(self):
        """Test a practical security filtering scenario."""
        # Block known malicious and ad domains
        filter = DomainFilter(
            blocked_domains=[
                "malware.com",
                "phishing.net",
                "*.ads.com",
                "*.ads.net",
                "*.adserver.com",
                "tracking.*.com",
            ]
        )

        # These should be blocked
        assert not filter.is_allowed("https://malware.com")
        assert not filter.is_allowed("https://phishing.net/steal-creds")
        assert not filter.is_allowed("https://banner.ads.com")
        assert not filter.is_allowed("https://popup.ads.net")
        assert not filter.is_allowed("https://popup.adserver.com")

        # These should be allowed
        assert filter.is_allowed("https://wikipedia.org")
        assert filter.is_allowed("https://github.com")
        assert filter.is_allowed("https://stackoverflow.com")

    def test_practical_allowlist_scenario(self):
        """Test a practical trusted sources scenario."""
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


if __name__ == "__main__":
    unittest.main()
