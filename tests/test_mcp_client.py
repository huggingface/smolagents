import json
from textwrap import dedent

import pytest
from mcp import StdioServerParameters

from smolagents.mcp_client import MCPClient


@pytest.fixture
def echo_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Echo Server")

        @mcp.tool()
        def echo_tool(text: str) -> str:
            """Echo the input text"""
            return f"Echo: {text}"

        mcp.run()
        '''
    )


@pytest.fixture
def structured_output_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP
        from typing import Any

        mcp = FastMCP("Structured Output Server")

        @mcp.tool()
        def user_info_tool(name: str) -> dict[str, Any]:
            """Get user information as structured data"""
            user_data = {
                "name": name,
                "age": 25,
                "email": f"{name.lower()}@example.com",
                "active": True
            }
            return user_data

        mcp.run()
        '''
    )


@pytest.fixture
def multi_tool_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Multi Tool Server")

        @mcp.tool()
        def search_docs(query: str) -> str:
            """Search documentation"""
            return f"Results for: {query}"

        @mcp.tool()
        def run_shell(cmd: str) -> str:
            """Run a shell command"""
            return f"Ran: {cmd}"

        @mcp.tool()
        def delete_file(path: str) -> str:
            """Delete a file"""
            return f"Deleted: {path}"

        mcp.run()
        '''
    )


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_mcp_client_with_syntax(echo_server_script: str):
    """Test the MCPClient with the context manager syntax."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"


def test_mcp_client_with_structured_output(structured_output_server_script: str):
    """Test the MCPClient with structured_output=True parameter."""
    server_parameters = StdioServerParameters(command="python", args=["-c", structured_output_server_script])
    with MCPClient(server_parameters, structured_output=True) as tools:
        assert len(tools) == 1
        assert tools[0].name == "user_info_tool"
        assert tools[0].output_type == "object"  # Should be object due to outputSchema

        # Check the output schema {'additionalProperties': True, 'title': 'user_info_toolDictOutput', 'type': 'object'}
        assert tools[0].output_schema is not None
        schema = tools[0].output_schema
        assert isinstance(schema, dict)
        assert schema.get("type") == "object"

        # Test that structured output is properly parsed
        result = tools[0].forward(**{"name": "Alice"})
        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 25
        assert result["email"] == "alice@example.com"
        assert result["active"] is True


def test_mcp_client_without_structured_output(structured_output_server_script: str):
    """Test the MCPClient with structured_output=False (default) for comparison."""
    server_parameters = StdioServerParameters(command="python", args=["-c", structured_output_server_script])
    with MCPClient(server_parameters, structured_output=False) as tools:
        assert len(tools) == 1
        assert tools[0].name == "user_info_tool"
        assert tools[0].output_type == "object"

        # Test that output is returned as raw text
        result = tools[0].forward(**{"name": "Alice"})
        assert isinstance(result, str)
        # Should be JSON string, not parsed object
        parsed_result = json.loads(result)
        assert parsed_result["name"] == "Alice"


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_mcp_client_try_finally_syntax(echo_server_script: str):
    """Test the MCPClient with the try ... finally syntax."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    mcp_client = MCPClient(server_parameters)
    try:
        tools = mcp_client.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"
    finally:
        mcp_client.disconnect()


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_multiple_servers(echo_server_script: str):
    """Test the MCPClient with multiple servers."""
    server_parameters = [
        StdioServerParameters(command="python", args=["-c", echo_server_script]),
        StdioServerParameters(command="python", args=["-c", echo_server_script]),
    ]
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 2
        assert tools[0].name == "echo_tool"
        assert tools[1].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"
        assert tools[1].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"


# ── tool_filter tests ──────────────────────────────────────────────


def test_tool_filter_allowlist(multi_tool_server_script: str):
    """tool_filter keeps only tools matching a name prefix."""
    server_parameters = StdioServerParameters(command="python", args=["-c", multi_tool_server_script])
    with MCPClient(
        server_parameters,
        structured_output=False,
        tool_filter=lambda t: t.name.startswith("search_"),
    ) as tools:
        assert len(tools) == 1
        assert tools[0].name == "search_docs"


def test_tool_filter_denylist(multi_tool_server_script: str):
    """tool_filter drops tools in a denylist."""
    server_parameters = StdioServerParameters(command="python", args=["-c", multi_tool_server_script])
    _DENYLIST = {"run_shell", "delete_file"}
    with MCPClient(
        server_parameters,
        structured_output=False,
        tool_filter=lambda t: t.name not in _DENYLIST,
    ) as tools:
        assert len(tools) == 1
        assert tools[0].name == "search_docs"


def test_tool_filter_drops_all(multi_tool_server_script: str):
    """When the filter rejects every tool, get_tools returns an empty list."""
    server_parameters = StdioServerParameters(command="python", args=["-c", multi_tool_server_script])
    with MCPClient(
        server_parameters,
        structured_output=False,
        tool_filter=lambda t: False,
    ) as tools:
        assert tools == []


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_tool_filter_default_none(echo_server_script: str):
    """Default (no filter) preserves backward compatibility — all tools kept."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
