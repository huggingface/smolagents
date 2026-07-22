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
"""Defense-in-depth SSRF guard for the local Python executor.

Rationale (CVE-2026-2654 / GHSA-jxgv-6j54-wwc7): when a developer allow-lists a
network-capable module (``requests``, ``urllib``, ``httpx``, ``http.client``,
``socket`` …) via ``additional_authorized_imports``, the local executor performs
no egress filtering. Model-generated (or prompt-injected) code can then reach
internal-only endpoints — cloud metadata (169.254.169.254), RFC1918 hosts,
loopback services — i.e. Server-Side Request Forgery (CWE-918).

This module blocks outbound connections to private / reserved / loopback /
link-local destinations at the single DNS choke point every stdlib and
third-party HTTP client funnels through: ``socket.getaddrinfo``. Filtering on the
**resolved IP** (not the hostname) defeats DNS-rebinding. Numeric-IP targets are
covered too, since ``getaddrinfo`` is invoked to build the sockaddr even when the
host is already an IP literal.

This is defense-in-depth, not a security boundary: ``LocalPythonExecutor`` is
explicitly not a sandbox. For untrusted code, use a remote executor
(Docker / E2B / Wasm).
"""

from __future__ import annotations

import contextlib
import ipaddress
import socket
from collections.abc import Iterator


class SSRFBlockedError(OSError):
    """Raised when the SSRF guard blocks a connection to a non-public address."""


def is_blocked_address(host: str) -> bool:
    """Return True if ``host`` (an IP literal) resolves to a non-public range.

    Blocks loopback, private (RFC1918 / ULA), link-local (incl. the
    169.254.169.254 cloud-metadata address), reserved, multicast and the
    unspecified address, for both IPv4 and IPv6. A host that is not a valid IP
    literal is treated as non-blocked here (the caller resolves it first).
    """
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # ``is_global`` is the authoritative "publicly routable" flag; the explicit
    # checks below make the intent auditable and catch a few ranges that some
    # Python versions classify inconsistently.
    return (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


@contextlib.contextmanager
def ssrf_guard(allowlist_hosts: set[str] | None = None) -> Iterator[None]:
    """Patch ``socket.getaddrinfo`` to reject non-public destinations.

    Args:
        allowlist_hosts: hostnames (exact match, case-insensitive) explicitly
            permitted to resolve to otherwise-blocked addresses (e.g. an internal
            service the developer intentionally exposes to the agent).

    The patch is process-global for the duration of the ``with`` block and is
    restored on exit. It targets the local (single-run) executor's execution
    window; concurrent unrelated network calls in the same process are also
    subject to it while active.
    """
    allow = {h.lower() for h in (allowlist_hosts or set())}
    real_getaddrinfo = socket.getaddrinfo

    def guarded_getaddrinfo(host, *args, **kwargs):
        results = real_getaddrinfo(host, *args, **kwargs)
        if host is not None and str(host).lower() in allow:
            return results
        for family, type_, proto, canonname, sockaddr in results:
            ip = sockaddr[0]
            if is_blocked_address(ip):
                raise SSRFBlockedError(
                    f"SSRF guard: blocked connection to non-public address "
                    f"{ip} (host={host!r}). Add the host to the executor's "
                    f"network_allowlist to permit it explicitly."
                )
        return results

    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo


# Network-capable stdlib / common third-party modules whose presence in the
# authorized imports means outbound requests are reachable from agent code.
NETWORK_CAPABLE_MODULES = frozenset(
    {
        "socket",
        "ssl",
        "http",
        "urllib",
        "urllib3",
        "ftplib",
        "telnetlib",
        "smtplib",
        "poplib",
        "imaplib",
        "requests",
        "httpx",
        "aiohttp",
        "websocket",
        "websockets",
        "asyncio",
    }
)


def authorized_imports_reach_network(authorized_imports: list[str]) -> bool:
    """True if any authorized import can perform network egress (incl. wildcard)."""
    for imp in authorized_imports:
        if imp == "*":
            return True
        if imp.split(".")[0] in NETWORK_CAPABLE_MODULES:
            return True
    return False
