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

import asyncio
import base64
import warnings
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable

from smolagents.tools import Tool


__all__ = ["MCPClient"]

if TYPE_CHECKING:
    from mcpadapt.core import StdioServerParameters


class _MCPAccessTool(Tool):
    """Internal lightweight Tool wrapper for MCP access primitives.

    Wraps a synchronous callable into a SmolAgents Tool without any
    signature validation, used for resources and prompts access tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        inputs: dict[str, dict[str, str | type | bool]],
        output_type: str,
        func: Callable[..., Any],
    ):
        self.name = name
        self.description = description
        self.inputs = inputs
        self.output_type = output_type
        self._func = func
        self.is_initialized = True
        self.skip_forward_signature_validation = True

    def forward(self, *args, **kwargs) -> Any:
        if args:
            raise ValueError(
                f"tool {self.name} does not support positional arguments, please use keyword arguments"
            )
        return self._func(**kwargs)


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
    ):
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

    def get_resource_access_tools(self) -> list[Tool]:
        """Return SmolAgents tools to access the MCP server's resources.

        MCP servers can expose resources: context-efficient, read-only data
        (documents, configuration, schemas, ...) addressable by URI. This
        method returns two tools that let the agent discover and read them
        like any other tool:

        - ``list_resources``: lists the available resources with their URI,
          name, MIME type and description. Call it first to discover what
          data is available.
        - ``read_resource(uri)``: reads the content of the resource
          identified by ``uri`` (as returned by ``list_resources``). Text
          content is returned as-is; binary content is returned base64-encoded.

        Example:
            ```python
            mcp_client = MCPClient(server_parameters)
            resource_tools = mcp_client.get_resource_access_tools()
            tools = mcp_client.get_tools() + resource_tools
            ```

        Returns:
            list[Tool]: A list containing the ``list_resources`` and
            ``read_resource`` tools.

        Raises:
            RuntimeError: If the MCP server sessions are not initialized
                (usually meaning the client is not connected).
        """
        if not self._adapter.sessions:
            raise RuntimeError(
                "Couldn't retrieve resources from MCP server, run `mcp_client.connect()` first before accessing resources"
            )

        # mcp is guaranteed to be installed here (MCPClient init requires mcpadapt)
        from mcp.types import BlobResourceContents, TextResourceContents
        from pydantic.networks import AnyUrl

        def _sync_list_resources() -> list[dict[str, Any]]:
            resources: list[dict[str, Any]] = []
            for session in self._adapter.sessions:
                result = asyncio.run_coroutine_threadsafe(session.list_resources(), self._adapter.loop).result()
                for resource in result.resources:
                    resources.append(
                        {
                            "uri": str(resource.uri),
                            "name": resource.name or "",
                            "mimeType": resource.mimeType or "",
                            "description": resource.description or "",
                        }
                    )
            return resources

        def _sync_read_resource(uri: str) -> dict[str, Any]:
            for session in self._adapter.sessions:
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        session.read_resource(AnyUrl(uri)), self._adapter.loop
                    ).result()
                except Exception:
                    continue  # resource not available on this server, try the next one
                contents: list[dict[str, Any]] = []
                for content in result.contents:
                    entry: dict[str, Any] = {
                        "uri": str(content.uri),
                        "mimeType": content.mimeType or "",
                    }
                    if isinstance(content, TextResourceContents):
                        entry["content"] = content.text
                    elif isinstance(content, BlobResourceContents):
                        raw = base64.b64decode(content.blob)
                        try:
                            entry["content"] = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            # binary content: keep the base64 payload
                            entry["content"] = content.blob
                            entry["encoding"] = "base64"
                    contents.append(entry)
                return {"uri": uri, "contents": contents}
            return {"uri": uri, "error": f"No resource found with uri '{uri}'"}

        return [
            _MCPAccessTool(
                name="list_resources",
                description=(
                    "List all resources available from the MCP server(s). "
                    "Returns a list of resources with their URI, name, MIME type and description. "
                    "Use this tool first to discover what data is available, "
                    "then call `read_resource` with the URI of the resource you want to fetch."
                ),
                inputs={},
                output_type="object",
                func=_sync_list_resources,
            ),
            _MCPAccessTool(
                name="read_resource",
                description=(
                    "Read the content of a resource exposed by the MCP server(s). "
                    "Takes the `uri` of the resource to read, as returned by `list_resources`. "
                    "Returns the resource content: text content is returned as-is, "
                    "binary content is returned base64-encoded."
                ),
                inputs={
                    "uri": {
                        "type": "string",
                        "description": "The URI of the resource to read, as returned by `list_resources`.",
                    }
                },
                output_type="object",
                func=_sync_read_resource,
            ),
        ]

    def get_prompt_access_tools(self) -> list[Tool]:
        """Return SmolAgents tools to access the MCP server's prompts.

        MCP servers can expose prompts: reusable prompt templates (with
        optional arguments) that instruct the agent how to perform a task.
        This method returns two tools that let the agent discover and fetch
        them like any other tool:

        - ``list_prompts``: lists the available prompts with their name,
          description and argument schema. Call it first to discover what
          templates are available.
        - ``get_prompt(name, arguments)``: fetches the rendered content of
          the prompt identified by ``name`` (as returned by
          ``list_prompts``), filling the template with the provided
          arguments. Text messages are returned as-is; image/audio content
          is returned base64-encoded.

        Example:
            ```python
            mcp_client = MCPClient(server_parameters)
            prompt_tools = mcp_client.get_prompt_access_tools()
            tools = mcp_client.get_tools() + prompt_tools
            ```

        Returns:
            list[Tool]: A list containing the ``list_prompts`` and
            ``get_prompt`` tools.

        Raises:
            RuntimeError: If the MCP server sessions are not initialized
                (usually meaning the client is not connected).
        """
        if not self._adapter.sessions:
            raise RuntimeError(
                "Couldn't retrieve prompts from MCP server, run `mcp_client.connect()` first before accessing prompts"
            )

        # mcp is guaranteed to be installed here (MCPClient init requires mcpadapt)
        from mcp.types import AudioContent, EmbeddedResource, ImageContent, TextContent

        def _sync_list_prompts() -> list[dict[str, Any]]:
            prompts: list[dict[str, Any]] = []
            for session in self._adapter.sessions:
                result = asyncio.run_coroutine_threadsafe(session.list_prompts(), self._adapter.loop).result()
                for prompt in result.prompts:
                    prompts.append(
                        {
                            "name": prompt.name,
                            "description": prompt.description or "",
                            "arguments": [
                                {
                                    "name": argument.name,
                                    "description": argument.description or "",
                                    "required": bool(argument.required),
                                }
                                for argument in (prompt.arguments or [])
                            ],
                        }
                    )
            return prompts

        def _sync_get_prompt(name: str, arguments: dict[str, str] | None = None) -> dict[str, Any]:
            for session in self._adapter.sessions:
                try:
                    result = asyncio.run_coroutine_threadsafe(
                        session.get_prompt(name, arguments or {}), self._adapter.loop
                    ).result()
                except Exception:
                    continue  # prompt not available on this server, try the next one
                messages: list[dict[str, Any]] = []
                for message in result.messages:
                    entry: dict[str, Any] = {"role": message.role}
                    content = message.content
                    if isinstance(content, TextContent):
                        entry["type"] = "text"
                        entry["content"] = content.text
                    elif isinstance(content, ImageContent):
                        entry["type"] = "image"
                        entry["mimeType"] = content.mimeType
                        entry["data"] = content.data
                    elif isinstance(content, AudioContent):
                        entry["type"] = "audio"
                        entry["mimeType"] = content.mimeType
                        entry["data"] = content.data
                    elif isinstance(content, EmbeddedResource):
                        entry["type"] = "embedded_resource"
                        entry["resource"] = {
                            "uri": str(content.resource.uri),
                            "mimeType": content.resource.mimeType or "",
                            "text": getattr(content.resource, "text", None),
                        }
                    else:
                        entry["type"] = "unknown"
                        entry["content"] = str(content)
                    messages.append(entry)
                return {"name": name, "messages": messages}
            return {"name": name, "error": f"No prompt found with name '{name}'"}

        return [
            _MCPAccessTool(
                name="list_prompts",
                description=(
                    "List all prompts available from the MCP server(s). "
                    "Returns a list of prompts with their name, description and argument schema. "
                    "Use this tool first to discover what templates are available, "
                    "then call `get_prompt` with the name of the prompt you want to fetch."
                ),
                inputs={},
                output_type="object",
                func=_sync_list_prompts,
            ),
            _MCPAccessTool(
                name="get_prompt",
                description=(
                    "Fetch the rendered content of a prompt exposed by the MCP server(s). "
                    "Takes the `name` of the prompt to fetch (as returned by `list_prompts`) "
                    "and optional `arguments` to fill the prompt template. "
                    "Returns the prompt messages with their roles and content; "
                    "text content is returned as-is, image/audio content is returned base64-encoded."
                ),
                inputs={
                    "name": {
                        "type": "string",
                        "description": "The name of the prompt to fetch, as returned by `list_prompts`.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Optional key-value arguments to fill the prompt template. Omit if the prompt has no arguments.",
                        "nullable": True,
                    },
                },
                output_type="object",
                func=_sync_get_prompt,
            ),
        ]

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ):
        """Disconnect from the MCP server."""
        self.disconnect(exc_type, exc_value, exc_traceback)
