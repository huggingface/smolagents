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
"""Domain filtering utilities for web search tools."""

import re
from typing import Any
from urllib.parse import urlparse


__all__ = ["DomainFilter"]


class DomainFilter:
    """
    Filters URLs based on domain allowlists with support for wildcard patterns.

    This class provides flexible domain filtering for web search results by restricting
    results to a defined set of trusted domains. It handles wildcard patterns and subdomain
    matching, ensuring that agents only access approved sources.

    Args:
        allowed_domains (`list[str]`, *optional*):
            List of domain patterns to allow. When specified, ONLY these domains (and their
            subdomains) will be permitted in search results. Supports wildcards (e.g., "*.edu",
            "*.gov", "wikipedia.org"). If not specified, no filtering is applied.
            Defaults to None (no filtering).

    Examples:
        ```python
        >>> # Allow only specific domains
        >>> filter = DomainFilter(allowed_domains=["wikipedia.org", "*.edu", "*.gov"])
        >>> filter.is_allowed("https://en.wikipedia.org/wiki/Python")
        True
        >>> filter.is_allowed("https://mit.edu/research")
        True
        >>> filter.is_allowed("https://example.com")
        False

        >>> # Allow educational and government sources
        >>> filter = DomainFilter(allowed_domains=["*.edu", "*.gov", "*.ac.uk"])
        >>> filter.is_allowed("https://harvard.edu")
        True
        >>> filter.is_allowed("https://nasa.gov")
        True
        ```
    """

    def __init__(
        self,
        allowed_domains: list[str] | None = None,
    ):
        self.allowed_domains = allowed_domains or []

        # Compile patterns for efficient matching
        self._allowed_patterns = [self._compile_domain_pattern(d) for d in self.allowed_domains]

    def _compile_domain_pattern(self, pattern: str) -> re.Pattern:
        """
        Convert a domain pattern with wildcards into a compiled regex pattern.

        Args:
            pattern: Domain pattern (e.g., "example.com", "*.ads.*", "subdomain.example.com")

        Returns:
            Compiled regex pattern for matching domains
        """
        # Normalize pattern: remove protocol, path, and convert to lowercase
        pattern = pattern.lower().strip()
        pattern = re.sub(r"^https?://", "", pattern)
        pattern = pattern.split("/")[0]  # Remove path if present

        # Escape special regex characters except *
        pattern = re.escape(pattern)

        # Replace escaped \* with appropriate regex based on position
        # If * is at the start, it matches subdomain parts: *.example.com matches sub.example.com
        # If * is at the end or middle, it matches any characters except dots: example.* matches example.com, example.net
        # For middle wildcards like *.ad.*, we need to be more flexible

        # Split by \* to handle each segment
        parts = pattern.split(r"\*")
        regex_parts = []

        for i, part in enumerate(parts):
            if i > 0:
                # Add wildcard matcher before this part
                # If we're matching at start (i==1 and first part is empty), match subdomain
                # Otherwise match any non-dot sequence
                if i == 1 and parts[0] == "":
                    # Leading wildcard: *.example.com
                    regex_parts.append(r"[^.]+")
                else:
                    # Middle or trailing wildcard
                    regex_parts.append(r"[^.]+")
            regex_parts.append(part)

        pattern = "".join(regex_parts)

        # Match pattern at domain boundaries (handles subdomains)
        # The pattern should match:
        # 1. Exact domain at the end: example.com$
        # 2. As a subdomain component: .example.com$
        pattern = rf"(^|\.){pattern}$"

        return re.compile(pattern, re.IGNORECASE)

    def _extract_domain(self, url: str) -> str | None:
        """
        Extract the domain from a URL.

        Args:
            url: URL string to extract domain from

        Returns:
            Domain string or None if URL is invalid
        """
        try:
            if not url or not isinstance(url, str):
                return None

            # Handle URLs without protocol
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]

            # Validate domain has at least one dot (except localhost)
            if not domain or (domain != "localhost" and "." not in domain):
                return None

            return domain
        except Exception:
            return None

    def _matches_pattern(self, domain: str, patterns: list[re.Pattern]) -> bool:
        """
        Check if a domain matches any of the given patterns.

        Args:
            domain: Domain to check
            patterns: List of compiled regex patterns

        Returns:
            True if domain matches any pattern, False otherwise
        """
        if not patterns:
            return False

        for pattern in patterns:
            if pattern.search(domain):
                return True
        return False

    def is_allowed(self, url: str) -> bool:
        """
        Check if a URL is allowed based on the configured allowlist.

        When an allowlist is specified, ONLY domains matching the allowlist patterns
        are permitted. This provides a clear boundary of trusted sources.

        Args:
            url: URL to check

        Returns:
            True if the URL is allowed, False if it should be filtered out
        """
        # If no allowlist is specified, all URLs are allowed
        if not self.allowed_domains:
            return True

        domain = self._extract_domain(url)
        if not domain:
            # Invalid URL, reject by default
            return False

        # Check if domain matches allowlist
        return self._matches_pattern(domain, self._allowed_patterns)

    def filter_results(self, results: list[dict[str, Any]], url_key: str = "url") -> list[dict[str, Any]]:
        """
        Filter a list of search results based on domain allowlist.

        Args:
            results: List of result dictionaries containing URLs
            url_key: Key name in result dict that contains the URL (default: "url")

        Returns:
            Filtered list of results with non-allowed domains removed
        """
        filtered = []
        for result in results:
            url = result.get(url_key)
            if url and self.is_allowed(url):
                filtered.append(result)
        return filtered

    def __repr__(self) -> str:
        return f"DomainFilter(allowed_domains={self.allowed_domains})"
