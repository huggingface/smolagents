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

from smolagents.mcp_firewall import MCPPayloadValidator, MCPServerUntrustedError, TrustVerifier, _extract_server_id
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
    ):
        # --- Layer 1: Pre-flight trust verification (before any TCP connection) ---
        if trust_verifier is not None:
            result = trust_verifier.verify(server_parameters)
            if not result.trusted:
                raise MCPServerUntrustedError(
                    server_id=result.server_id,
                    trust_score=result.trust_score,
                    reasons=result.reasons,
                )

        self._payload_validator = payload_validator
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
