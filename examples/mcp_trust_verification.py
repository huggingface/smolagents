"""Verify that an MCP server is trustworthy BEFORE loading its tools.

Connecting an agent to an MCP server is a trust decision:

- **Stdio servers** (spawned via `StdioServerParameters`) run code on *your*
  machine: trust them like you would trust any software you install.
- **Remote servers** (Streamable HTTP / SSE, passed as a ``{"url": ...}`` dict)
  do not run code locally, but they receive your prompts and they control the
  tool definitions that are handed to your model — a compromised or malicious
  server can poison tool descriptions, exfiltrate data, or steer the agent.

smolagents already refuses to load MCP tools unless you explicitly acknowledge
the trust decision with ``trust_remote_code=True`` (see
``ToolCollection.from_mcp``). This example adds the missing first step: a
fail-closed *verification gate* you run BEFORE ``from_mcp``, deciding *whether*
that acknowledgment is warranted. The gate implements the trust-verification
pattern requested in https://github.com/huggingface/smolagents/issues/2305:

1. stdio servers must match an explicit allowlist of ``(command, args)`` pairs
   (pin versions!);
2. remote servers must use ``https://`` (plain ``http://`` is only accepted for
   loopback addresses, e.g. a local dev server) and their host must be in your
   allowlist;
3. optionally, an external behavioral trust scorer (any API returning a score in
   ``[0.0, 1.0]``, e.g. the one proposed in issue #2305) can be plugged in and
   gated by a threshold.

Any failed check raises :class:`ServerTrustVerificationError` and the tools are
never loaded. The script runs fully offline against a local in-process MCP
server, so you can execute it as-is: no network, no Node.js, no LLM call.

Requirements:
    # Until huggingface/smolagents#2585 lands, mcp 2.x breaks mcpadapt, so pin mcp<2:
    pip install "smolagents[mcp]" "mcp<2"
"""

from __future__ import annotations

import math
import sys
import urllib.parse
from typing import Callable, Iterable

from mcp import StdioServerParameters

from smolagents import ToolCollection


# ---------------------------------------------------------------------------
# 1. The verification gate
# ---------------------------------------------------------------------------


class ServerTrustVerificationError(RuntimeError):
    """Raised when an MCP server fails a trust check, before any connection."""


# A behavioral trust scorer maps a server URL to a score in [0.0, 1.0].
# Plug in any service you trust (see issue #2305 for a discussion of scoring
# APIs). ``None`` disables the check.
TrustScorer = Callable[[str], float]

# Hosts for which plain http:// is accepted (local development only).
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def verify_server_trust(
    server_parameters: StdioServerParameters | dict | Iterable[StdioServerParameters | dict],
    *,
    trusted_stdio_servers: Iterable[tuple[str, tuple[str, ...]]] = (),
    allowed_remote_hosts: Iterable[str] = (),
    require_https_for_remote: bool = True,
    trust_scorer: TrustScorer | None = None,
    score_threshold: float = 0.7,
) -> None:
    """Fail-closed trust gate: raise :class:`ServerTrustVerificationError` if the
    MCP server does not pass every check. Call this BEFORE
    ``ToolCollection.from_mcp(..., trust_remote_code=True)``.

    Args:
        server_parameters: ``mcp.StdioServerParameters`` for stdio servers, or a
            ``{"url": ..., "transport": ...}`` dict (or list of either) for
            Streamable HTTP / SSE servers — same shapes accepted by
            ``ToolCollection.from_mcp``.
        trusted_stdio_servers: allowlist of ``(command, args_tuple)`` pairs.
            Stdio servers execute code locally, so the default is DENY ALL:
            only servers whose exact ``command`` and ``args`` are listed here
            pass. Prefer pinned package invocations, e.g.
            ``("uvx", ("--quiet", "pubmedmcp@0.1.3"))``.
            Note this is a coarse command/args allowlist: it does not capture
            ``cwd``, ``env``, or ``PATH``-based executable resolution, so treat
            it as one layer of defense and prefer absolute executable paths,
            pinned packages, and (for untrusted servers) running them in a
            sandbox or container.
        allowed_remote_hosts: allowlist of hostnames (no scheme, no port,
            case-insensitive) that remote servers may use. Default is DENY ALL.
            Loopback hosts (localhost/127.0.0.1/::1) are treated as your local
            trust domain: they are exempt from this allowlist and from the
            ``trust_scorer``, and may use ``http://`` (any non-http(s) URL is
            still refused). This is an intentional escape hatch for local
            development servers.
        require_https_for_remote: refuse ``http://`` URLs unless the host is a
            loopback address. ``https://`` guarantees the tool definitions and
            your prompts are not readable in transit.
        trust_scorer: optional callable ``url -> score in [0.0, 1.0]``, e.g. a
            wrapper around an external behavioral trust API.
        score_threshold: minimum score required when ``trust_scorer`` is set.

    Raises:
        ServerTrustVerificationError: with an actionable message listing every
            failed check. Nothing is connected or executed on failure.
        ValueError: if ``trust_scorer`` is set but ``score_threshold`` is not a
            finite number in ``[0.0, 1.0]``.
    """
    if trust_scorer is not None and (
        not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not math.isfinite(score_threshold)
        or not 0.0 <= score_threshold <= 1.0
    ):
        raise ValueError(f"score_threshold must be a finite number in [0.0, 1.0], got {score_threshold!r}.")
    if isinstance(server_parameters, (StdioServerParameters, dict)):
        server_parameters = [server_parameters]
    elif isinstance(server_parameters, str):
        raise ServerTrustVerificationError(
            f"Expected MCP server parameters, got a bare URL string '{server_parameters}': "
            "pass {'url': ..., 'transport': ...} instead."
        )
    failures: list[str] = []
    for parameters in server_parameters:
        if isinstance(parameters, StdioServerParameters):
            _check_stdio_server(parameters, set(trusted_stdio_servers), failures)
        elif isinstance(parameters, dict):
            _check_remote_server(
                parameters,
                set(allowed_remote_hosts),
                require_https_for_remote,
                trust_scorer,
                score_threshold,
                failures,
            )
        else:
            failures.append(
                f"Unsupported server_parameters type {type(parameters).__name__}: "
                "expected mcp.StdioServerParameters or a {'url': ..., 'transport': ...} dict."
            )
    if failures:
        raise ServerTrustVerificationError("Refusing to load tools from MCP server(s):\n- " + "\n- ".join(failures))


def _check_stdio_server(
    parameters: StdioServerParameters, trusted_stdio_servers: set[tuple[str, tuple[str, ...]]], failures: list[str]
) -> None:
    command = parameters.command
    args = tuple(parameters.args or ())
    if (command, args) not in trusted_stdio_servers:
        failures.append(
            f"Stdio server '{command}' is not in your trusted_stdio_servers allowlist. "
            "Stdio servers execute code on YOUR machine: only allow servers you have "
            "inspected, with pinned versions (e.g. ('uvx', ('--quiet', 'pkg@1.2.3')))."
        )


def _check_remote_server(
    parameters: dict,
    allowed_remote_hosts: set[str],
    require_https_for_remote: bool,
    trust_scorer: TrustScorer | None,
    score_threshold: float,
    failures: list[str],
) -> None:
    url = parameters.get("url")
    if not isinstance(url, str):
        failures.append(f"Remote MCP server dict has no 'url' string: {parameters!r}")
        return
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"}:
        failures.append(f"Refusing non-http(s) URL '{url}': MCP servers must be reached over http(s).")
        return
    if scheme == "http" and require_https_for_remote and hostname not in _LOOPBACK_HOSTS:
        failures.append(
            f"Refusing cleartext http:// URL '{url}': tool definitions and prompts would travel "
            "unencrypted. Use https://, or only http:// for loopback hosts (localhost/127.0.0.1)."
        )
    if hostname not in _LOOPBACK_HOSTS and hostname not in {host.lower() for host in allowed_remote_hosts}:
        failures.append(
            f"Remote host '{hostname}' is not in your allowed_remote_hosts allowlist. "
            "Only add servers whose operator and behavior you have verified."
        )
    if trust_scorer is not None and hostname not in _LOOPBACK_HOSTS:
        score = trust_scorer(url)
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            failures.append(
                f"Trust scorer returned an invalid score {score!r} for '{url}': "
                "expected a finite number in [0.0, 1.0]. Refusing to trust a server "
                "whose score cannot be validated."
            )
        elif score < score_threshold:
            failures.append(
                f"Behavioral trust score {score:.2f} for '{url}' is below threshold {score_threshold:.2f}."
            )


# ---------------------------------------------------------------------------
# 2. A local MCP server used by the demo (stdio, runs in this Python process)
# ---------------------------------------------------------------------------

_ECHO_SERVER_CODE = """
import logging

logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Echo Server")

@mcp.tool()
def echo_tool(text: str) -> str:
    \"\"\"Echo the input text\"\"\"
    return f"Echo: {text}"

mcp.run()
"""

# In real life, allowlist what you actually use, with pinned versions, e.g.:
# trusted_stdio_servers = {("uvx", ("--quiet", "pubmedmcp@0.1.3"))}
TRUSTED_STDIO_SERVERS = {(sys.executable, ("-c", _ECHO_SERVER_CODE))}

ALLOWED_REMOTE_HOSTS = {"mcp-server.example.com"}


def main() -> None:
    print("== MCP server trust verification demo ==")
    print("(fully offline: the trusted server runs inside this process)\n")

    # [1] Trusted local stdio server: gate passes -> tools load -> tool runs.
    echo_server = StdioServerParameters(command=sys.executable, args=["-c", _ECHO_SERVER_CODE])
    verify_server_trust(echo_server, trusted_stdio_servers=TRUSTED_STDIO_SERVERS)
    with ToolCollection.from_mcp(echo_server, trust_remote_code=True, structured_output=False) as tool_collection:
        result = tool_collection.tools[0].forward(text="hello from a verified MCP server")
        print(f"[1] PASS trusted stdio server loaded and ran: {result!r}")

    # [2] Unknown remote server: not allowlisted -> refused BEFORE connecting.
    unknown_remote = {"url": "https://evil-mcp.example.com/mcp", "transport": "streamable-http"}
    try:
        verify_server_trust(unknown_remote, allowed_remote_hosts=ALLOWED_REMOTE_HOSTS)
    except ServerTrustVerificationError as error:
        print(f"[2] BLOCKED unallowlisted remote server before any connection:\n    {error}")
    else:
        raise AssertionError("Unallowlisted remote server passed verification!")

    # [3] Cleartext remote URL: http:// to a non-loopback host -> refused.
    cleartext_remote = {"url": "http://mcp-server.example.com/mcp", "transport": "streamable-http"}
    try:
        verify_server_trust(cleartext_remote, allowed_remote_hosts=ALLOWED_REMOTE_HOSTS)
    except ServerTrustVerificationError as error:
        print(f"[3] BLOCKED cleartext http:// remote server:\n    {error}")
    else:
        raise AssertionError("Cleartext remote URL passed verification!")

    # [4] Behavioral trust scorer plug-in (issue #2305): a scoring API reports a
    # low score for an otherwise allowlisted host -> refused.
    def stub_trust_scorer(url: str) -> float:  # stand-in for an external scoring API
        return 0.4  # e.g. unknown operator, no SLA

    low_score_remote = {"url": "https://mcp-server.example.com/mcp", "transport": "streamable-http"}
    try:
        verify_server_trust(
            low_score_remote,
            allowed_remote_hosts=ALLOWED_REMOTE_HOSTS,
            trust_scorer=stub_trust_scorer,
            score_threshold=0.7,
        )
    except ServerTrustVerificationError as error:
        print(f"[4] BLOCKED allowlisted host with low behavioral trust score:\n    {error}")
    else:
        raise AssertionError("Low-score server passed verification!")

    # [5] A scorer that returns an invalid score (NaN) must also fail closed:
    # an unvalidatable score is not a pass.
    def nan_trust_scorer(url: str) -> float:  # stand-in for a broken scoring API
        return float("nan")

    try:
        verify_server_trust(
            low_score_remote,
            allowed_remote_hosts=ALLOWED_REMOTE_HOSTS,
            trust_scorer=nan_trust_scorer,
            score_threshold=0.7,
        )
    except ServerTrustVerificationError as error:
        print(f"[5] BLOCKED invalid (NaN) behavioral trust score:\n    {error}")
    else:
        raise AssertionError("Invalid scorer output passed verification!")

    # [6] Loopback http:// is fine for local development servers.
    local_dev_server = {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}
    verify_server_trust(local_dev_server, allowed_remote_hosts=ALLOWED_REMOTE_HOSTS)
    print("[6] PASS loopback http:// server accepted for local development (verification only)")

    print("\nAll trust checks behaved as expected: verified servers load, untrusted servers are refused.")


if __name__ == "__main__":
    main()
