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
try:
    from mcp import (
        StdioServerParameters,
    )
    from mcpadapt.core import MCPAdapt
    from mcpadapt.smolagents_adapter import SmolAgentsAdapter
except ImportError:
    raise ImportError("MCPClient needs optional dependencies to be installed, run `pip install smolagents[mcp]`")

from typing import Any, Optional, Type, TracebackType

from smolagents.tools import Tool


class MCPClient:
    """Manages the connection to an MCP server and make its tools available to SmolAgents.

    Note: tools can only be accessed after the connection has been started with the
        `connect()` method, done during the init. If you don't use the context manager
        we strongly encourage to use "try ... finally" to ensure the connection is cleaned up.

    Attributes:
        tools: The SmolAgents tools available from the MCP server.

    Usage:
        # fully managed context manager + stdio
        with MCPClient(...) as tools:
            # tools are now available

        # context manager + sse
        with MCPClient({"url": "http://localhost:8000/sse"}) as tools:
            # tools are now available

        # manually manage the connection via the mcp_client object:
        try:
            mcp_client = MCPClient(...)
            tools = mcp_client.tools

            # use your tools here.
        finally:
            mcp_client.stop()
    """

    def __init__(
        self,
        serverparams: StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]],
    ):
        """Initialize the MCP Client

        Args:
            serverparams (StdioServerParameters | dict[str, Any] | list[StdioServerParameters | dict[str, Any]]):
                MCP server parameters (stdio or sse). Can be a list if you want to connect multiple MCPs at once.

        """
        super().__init__()
        self._adapter = MCPAdapt(serverparams, SmolAgentsAdapter())
        self._tools = None

        self.connect()

    def connect(self):
        """Connect to the MCP server and initialize the tools."""
        self._tools = self._adapter.__enter__()

    def disconnect(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ):
        """Disconnect from the MCP server"""
        self._adapter.__exit__(exc_type, exc_value, exc_traceback)

    @property
    def tools(self) -> list[Tool]:
        """The SmolAgents tools available from the MCP server.

        Raises:
            ValueError: If the MCP server tools is None (usually assuming the server is not started).

        Returns:
            The SmolAgents tools available from the MCP server.
        """
        if self._tools is None:
            raise ValueError(
                "Couldn't retrieve tools from MCP server, run `mcp_client.connect()` first before accessing `tools`"
            )
        return self._tools

    def __enter__(self):
        """Connect to the MCP server and return the tools directly."""
        self.connect()
        return self.tools

    def __exit__(self, exc_type, exc_value, traceback):
        """Disconnect from the MCP server."""
        self.disconnect(exc_type, exc_value, traceback)
