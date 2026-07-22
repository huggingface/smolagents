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
"""Tests for the SSRF guard (CVE-2026-2654 / GHSA-jxgv-6j54-wwc7)."""

import socket

import pytest

from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.network_guard import (
    SSRFBlockedError,
    authorized_imports_reach_network,
    is_blocked_address,
    ssrf_guard,
)


@pytest.mark.parametrize(
    "ip, blocked",
    [
        ("169.254.169.254", True),  # cloud metadata
        ("127.0.0.1", True),  # loopback
        ("10.0.0.5", True),  # RFC1918
        ("192.168.1.1", True),  # RFC1918
        ("172.16.0.1", True),  # RFC1918
        ("::1", True),  # IPv6 loopback
        ("fe80::1", True),  # IPv6 link-local
        ("fc00::1", True),  # IPv6 ULA
        ("8.8.8.8", False),  # public
        ("1.1.1.1", False),  # public
    ],
)
def test_is_blocked_address(ip, blocked):
    assert is_blocked_address(ip) is blocked


def test_is_blocked_address_non_ip_is_not_blocked():
    # Hostnames are resolved by the caller; the classifier only judges IP literals.
    assert is_blocked_address("example.com") is False


def test_ssrf_guard_blocks_metadata_and_private():
    with ssrf_guard():
        for host in ("169.254.169.254", "127.0.0.1", "10.0.0.1"):
            with pytest.raises(SSRFBlockedError):
                socket.getaddrinfo(host, 80)


def test_ssrf_guard_allows_public():
    with ssrf_guard():
        # Public literal resolves without raising (no connection is made).
        assert socket.getaddrinfo("8.8.8.8", 53)


def test_ssrf_guard_allowlist_overrides():
    with ssrf_guard(allowlist_hosts={"127.0.0.1"}):
        assert socket.getaddrinfo("127.0.0.1", 80)


def test_ssrf_guard_restores_getaddrinfo():
    before = socket.getaddrinfo
    with ssrf_guard():
        pass
    assert socket.getaddrinfo is before


@pytest.mark.parametrize(
    "imports, reaches",
    [
        (["math", "requests"], True),
        (["urllib.request"], True),
        (["math", "pandas"], False),
        (["*"], True),
    ],
)
def test_authorized_imports_reach_network(imports, reaches):
    assert authorized_imports_reach_network(imports) is reaches


# --- Integration through LocalPythonExecutor -------------------------------


def _run(executor: LocalPythonExecutor, code: str):
    executor.send_tools({"final_answer": (lambda x: x)})
    return executor(code).output


_CONNECT = (
    "import socket\n"
    "try:\n"
    "    s = socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
    "    s.close()\n"
    "    outcome = 'REACHED'\n"
    "except Exception as e:\n"
    "    outcome = 'BLOCKED ' + str(e)[:120]\n"
    "final_answer(outcome)\n"
)


def _serve() -> tuple[socket.socket, int]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv, srv.getsockname()[1]


def test_executor_blocks_ssrf_to_reachable_internal_target():
    """The guard blocks a genuinely reachable loopback service (fail-closed)."""
    srv, port = _serve()
    try:
        ex = LocalPythonExecutor(additional_authorized_imports=["socket"])
        out = _run(ex, _CONNECT.format(port=port))
        assert out.startswith("BLOCKED")
        assert "127.0.0.1" in out
    finally:
        srv.close()


def test_executor_reaches_target_when_guard_disabled():
    """block_ssrf=False restores the historical (unguarded) behaviour."""
    srv, port = _serve()
    try:
        ex = LocalPythonExecutor(additional_authorized_imports=["socket"], block_ssrf=False)
        assert _run(ex, _CONNECT.format(port=port)) == "REACHED"
    finally:
        srv.close()


def test_executor_allowlist_permits_internal_host():
    srv, port = _serve()
    try:
        ex = LocalPythonExecutor(additional_authorized_imports=["socket"], network_allowlist=["127.0.0.1"])
        assert _run(ex, _CONNECT.format(port=port)) == "REACHED"
    finally:
        srv.close()


def test_executor_guard_not_armed_without_network_module():
    """No network module authorized → getaddrinfo is never patched (zero overhead)."""
    before = socket.getaddrinfo
    ex = LocalPythonExecutor(additional_authorized_imports=["math"])
    assert _run(ex, "import math\nfinal_answer(math.factorial(6))") == 720
    assert socket.getaddrinfo is before
