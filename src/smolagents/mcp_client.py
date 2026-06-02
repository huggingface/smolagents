#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any

from smolagents.mcp_firewall import (
    MCPAuditLogger,
    MCPCallSentinel,
    MCPPayloadValidator,
    MCPRateLimiter,
    MCPResponseSanitizer,
    MCPRugPullDetectedError,
    MCPSecurityHook,
    MCPServerUntrustedError,
    MCPToolAllowlist,
    MCPToolBlockedError,
    MCPToolFingerprinter,
    TrustVerifier,
    _dispatch_hooks,
    _extract_server_id,
    wrap_tool_with_guardian,
)
from smolagents.tools import Tool


__all__ = ["MCPClient"]

if TYPE_CHECKING:
    from mcpadapt.core import StdioServerParameters


class MCPClient:
    """Manages the connection to an MCP server and make its tools available to SmolAgents.

    Note: tools can only be accessed after the connection has been started with the
        `connect()` method, done during the init. If you don't use the context manager
        we strongly encourage to use "try ... finally" to ensure the connection is cleaned up.

    Args:
        server_parameters (StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]]):
            Configuration parameters to connect to the MCP server. Can be a list if you want to connect multiple MCPs at once.

            - An instance of `mcp.StdioServerParameters` for connecting a Stdio MCP server via standard input/output using a subprocess.

            - A `dict` with at least:
              - "url": URL of the server.
              - "transport": Transport protocol to use, one of:
                - "streamable-http": Streamable HTTP transport (default).
                - "sse": Legacy HTTP+SSE transport (deprecated).
        adapter_kwargs (dict[str, Any], optional):
            Additional keyword arguments to be passed directly to `MCPAdapt`.
        structured_output (bool, optional, defaults to False):
            Whether to enable structured output features for MCP tools. If True, enables:
            - Support for outputSchema in MCP tools
            - Structured content handling (structuredContent from MCP responses)
            - JSON parsing fallback for structured data
            If False, uses the original simple text-only behavior for backwards compatibility.
        trust_verifier (TrustVerifier, optional):
            A pre-flight trust verifier that is evaluated *before* any TCP connection
            is established. Use ``StaticTrustVerifier`` for URL/command-based rules,
            or subclass ``TrustVerifier`` to call an external reputation API.
            When provided, raises ``MCPServerUntrustedError`` if the server is rejected.
            Defaults to None (no verification — preserves backward compatibility).
        payload_validator (MCPPayloadValidator, optional):
            A post-connection validator that inspects each tool's name, description,
            and input schema for prompt-injection patterns and resource-exhaustion
            attacks. Raises ``MCPPayloadValidationError`` if any tool fails validation.
            Defaults to None (no validation — preserves backward compatibility).
        fingerprinter (MCPToolFingerprinter, optional):
            Rug-pull detector.  On first connect, records SHA-256 fingerprints of
            every tool's definition in a local lockfile (``.mcp-lock.json``).
            On every subsequent connect, re-fingerprints and raises
            ``MCPRugPullDetectedError`` if any definition has changed.
            Defaults to None (no fingerprinting).
        sentinel (MCPCallSentinel, optional):
            Runtime firewall.  Wraps each tool's ``forward()`` to scan arguments
            for credential exfiltration patterns (pre-call) and responses for
            injection patterns and size violations (post-call).
            Raises ``MCPCallInterceptedError`` when a violation is detected.
            Defaults to None (no call interception).
        audit_logger (MCPAuditLogger, optional):
            Structured audit log.  Records every tool call (timestamp, server_id,
            hashed args, hashed response, duration, blocked status) to a JSONL
            file.  Defaults to None (no audit logging).

    Example:
        ```python
        # fully managed context manager + stdio
        with MCPClient(...) as tools:
            # tools are now available

        # context manager + Streamable HTTP transport:
        with MCPClient({"url": "http://localhost:8000/mcp", "transport": "streamable-http"}) as tools:
            # tools are now available

        # Enable structured output for advanced MCP tools:
        with MCPClient(server_parameters, structured_output=True) as tools:
            # tools with structured output support are now available

        # With security layers enabled:
        from smolagents import StaticTrustVerifier, MCPPayloadValidator
        with MCPClient(
            server_parameters,
            trust_verifier=StaticTrustVerifier(require_https=True),
            payload_validator=MCPPayloadValidator(),
        ) as tools:
            # tools have been trust-verified and payload-validated

        # manually manage the connection via the mcp_client object:
        try:
            mcp_client = MCPClient(...)
            tools = mcp_client.get_tools()

            # use your tools here.
        finally:
            mcp_client.disconnect()
        ```
    """

    def __init__(
        self,
        server_parameters: "StdioServerParameters" | dict[str, Any] | list["StdioServerParameters" | dict[str, Any]],
        adapter_kwargs: dict[str, Any] | None = None,
        structured_output: bool | None = None,
        trust_verifier: TrustVerifier | None = None,
        payload_validator: MCPPayloadValidator | None = None,
        fingerprinter: MCPToolFingerprinter | None = None,
        sentinel: MCPCallSentinel | None = None,
        audit_logger: MCPAuditLogger | None = None,
        allowlist: MCPToolAllowlist | None = None,
        sanitizer: MCPResponseSanitizer | None = None,
        rate_limiter: MCPRateLimiter | None = None,
        hooks: list[MCPSecurityHook] | None = None,
    ):
        self._hooks: list[MCPSecurityHook] = list(hooks) if hooks else []

        # --- Layer 1: Pre-flight trust verification (before any TCP connection) ---
        self._trust_score: float | None = None
        if trust_verifier is not None:
            result = trust_verifier.verify(server_parameters)
            if not result.trusted:
                _dispatch_hooks(self._hooks, "server_blocked", result.server_id, {
                    "trust_score": result.trust_score,
                    "reasons": result.reasons,
                })
                raise MCPServerUntrustedError(
                    server_id=result.server_id,
                    trust_score=result.trust_score,
                    reasons=result.reasons,
                )
            self._trust_score = result.trust_score

        self._payload_validator = payload_validator
        self._fingerprinter = fingerprinter
        self._sentinel = sentinel
        self._audit_logger = audit_logger
        self._allowlist = allowlist
        self._sanitizer = sanitizer
        self._rate_limiter = rate_limiter
        self._server_id = _extract_server_id(server_parameters)

        # Handle future warning for structured_output default value change
        if structured_output is None:
            warnings.warn(
                "Parameter 'structured_output' was not specified. "
                "Currently it defaults to False, but in version 1.25, the default will change to True. "
                "To suppress this warning, explicitly set structured_output=True (new behavior) or structured_output=False (legacy behavior). "
                "See documentation at https://huggingface.co/docs/smolagents/tutorials/tools#structured-output-and-output-schema-support for more details.",
                FutureWarning,
                stacklevel=2,
            )
            structured_output = False

        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install 'mcp' extra to use MCPClient: `pip install 'smolagents[mcp]'`")
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                transport = "streamable-http"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        adapter_kwargs = adapter_kwargs or {}
        self._adapter = MCPAdapt(
            server_parameters, SmolAgentsAdapter(structured_output=structured_output), **adapter_kwargs
        )
        self._tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        """Connect to the MCP server and initialize the tools."""
        self._tools: list[Tool] = self._adapter.__enter__()
        # --- Layer 2: Post-connection payload validation ---
        if self._payload_validator is not None and self._tools is not None:
            self._payload_validator.validate_tool_list(self._tools, self._server_id)
        # --- Layer 3: Rug-pull fingerprint verification ---
        if self._fingerprinter is not None and self._tools is not None:
            try:
                self._fingerprinter.fingerprint_and_verify(self._tools, self._server_id)
            except MCPRugPullDetectedError as exc:
                _dispatch_hooks(self._hooks, "rug_pull_detected", self._server_id, {
                    "tool_name": getattr(exc, "tool_name", "?"),
                    "reason": str(exc),
                })
                raise
        # --- Layer 5: Allowlist enforcement ---
        if self._allowlist is not None and self._tools is not None:
            try:
                self._allowlist.validate_tool_list(self._tools, self._server_id)
            except MCPToolBlockedError as exc:
                _dispatch_hooks(self._hooks, "tool_blocked", self._server_id, {
                    "tool_name": exc.tool_name,
                    "reason": exc.reason,
                })
                raise
        # --- Layers 4/6/7/8: Wrap tools with runtime guardian ---
        if (
            self._sentinel is not None or self._audit_logger is not None
            or self._allowlist is not None or self._sanitizer is not None
            or self._rate_limiter is not None or self._hooks
        ) and self._tools is not None:
            for tool in self._tools:
                wrap_tool_with_guardian(
                    tool,
                    server_id=self._server_id,
                    sentinel=self._sentinel,
                    audit_logger=self._audit_logger,
                    trust_score=self._trust_score,
                    fingerprint_verified=self._fingerprinter is not None,
                    allowlist=self._allowlist,
                    sanitizer=self._sanitizer,
                    rate_limiter=self._rate_limiter,
                    hooks=self._hooks or None,
                )

    def disconnect(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        exc_traceback: TracebackType | None = None,
    ):
        """Disconnect from the MCP server"""
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)

    def get_tools(self) -> list[Tool]:
        """The SmolAgents tools available from the MCP server.

        Note: for now, this always returns the tools available at the creation of the session,
        but it will in a future release return also new tools available from the MCP server if
        any at call time.

        Raises:
            ValueError: If the MCP server tools is None (usually assuming the server is not started).

        Returns:
            list[Tool]: The SmolAgents tools available from the MCP server.
        """
        if self._tools is None:
            raise ValueError(
                "Couldn't retrieve tools from MCP server, run `mcp_client.connect()` first before accessing `tools`"
            )
        return self._tools

    def __enter__(self) -> list[Tool]:
        """Connect to the MCP server and return the tools directly.

        Note that because of the `.connect` in the init, the mcp_client
        is already connected at this point.
        """
        return self._tools

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        """Disconnect from the MCP server."""
        self.disconnect(exc_type, exc_value, exc_traceback)
