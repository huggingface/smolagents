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
def resource_server_script():
    return dedent(
        '''
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("Resource Server")

        @mcp.resource("config://app")
        def app_config() -> str:
            """Application configuration"""
            return '{"name": "demo", "version": "1.0"}'

        @mcp.resource("config://schema")
        def db_schema() -> str:
            """Database schema"""
            return "CREATE TABLE users (id INT, name TEXT)"

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


# Ignore FutureWarning about structured_output default value change: this test intentionally uses default behavior
@pytest.mark.filterwarnings("ignore:.*structured_output:FutureWarning")
def test_mcp_client_resource_access(resource_server_script: str):
    """Test listing and reading MCP resources through resource access tools."""
    server_parameters = StdioServerParameters(command="python", args=["-c", resource_server_script])
    mcp_client = MCPClient(server_parameters)
    try:
        resource_tools = mcp_client.get_resource_access_tools()
        assert len(resource_tools) == 2
        assert {tool.name for tool in resource_tools} == {"list_resources", "read_resource"}

        # list_resources returns metadata for all exposed resources
        list_tool = next(tool for tool in resource_tools if tool.name == "list_resources")
        listed = list_tool.forward()
        assert isinstance(listed, list)
        assert len(listed) == 2
        uris = {resource["uri"] for resource in listed}
        assert "config://app" in uris
        assert "config://schema" in uris
        # metadata includes a human-readable name and description
        app_resource = next(resource for resource in listed if resource["uri"] == "config://app")
        assert app_resource["name"]
        assert app_resource["description"]

        # read_resource returns the text content for a known URI
        read_tool = next(tool for tool in resource_tools if tool.name == "read_resource")
        content = read_tool.forward(uri="config://app")
        assert content["uri"] == "config://app"
        assert content["contents"][0]["content"] == '{"name": "demo", "version": "1.0"}'

        schema_content = read_tool.forward(uri="config://schema")
        assert "CREATE TABLE users" in schema_content["contents"][0]["content"]

        # read_resource reports an error for an unknown URI instead of raising
        missing = read_tool.forward(uri="config://missing")
        assert "error" in missing
    finally:
        mcp_client.disconnect()
