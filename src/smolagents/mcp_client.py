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
                - "streamable-http": (recommended) Streamable HTTP transport.
                - "sse": Legacy HTTP+SSE transport (deprecated).
              If "transport" is omitted, the legacy "sse" transport is assumed (a deprecation warning will be issued).

            <Deprecated version="1.17.0">
            The HTTP+SSE transport is deprecated and future behavior will default to the Streamable HTTP transport.
            Please pass explicitly the "transport" key.
            </Deprecated>

    Example:
        ```python
        # fully managed context manager + stdio
        with MCPClient(...) as tools:
            # tools are now available

        # context manager + Streamable HTTP transport:
        with MCPClient({"url": "http://localhost:8000/mcp", "transport": "streamable-http"}) as tools:
            # tools are now available

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
    ):
        try:
            from mcpadapt.core import MCPAdapt
            from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install 'mcp' extra to use MCPClient: `pip install 'smolagents[mcp]'`")
        if isinstance(server_parameters, dict):
            transport = server_parameters.get("transport")
            if transport is None:
                warnings.warn(
                    "Passing a dict as server_parameters without specifying the 'transport' key is deprecated. "
                    "For now, it defaults to the legacy 'sse' (HTTP+SSE) transport, but this default will change "
                    "to 'streamable-http' in version 1.20. Please add the 'transport' key explicitly. ",
                    FutureWarning,
                )
                transport = "sse"
                server_parameters["transport"] = transport
            if transport not in {"sse", "streamable-http"}:
                raise ValueError(
                    f"Unsupported transport: {transport}. Supported transports are 'streamable-http' and 'sse'."
                )
        self._adapter = MCPAdapt(server_parameters, SmolAgentsAdapter())
        self._tools: list[Tool] | None = None
        self.connect()

    def connect(self):
        """Connect to the MCP server and initialize the tools."""
        self._tools: list[Tool] = self._adapter.__enter__()

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
