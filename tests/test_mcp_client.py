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


def test_mcp_client_with_syntax(echo_server_script: str):
    """Test the MCPClient with the context manager syntax."""
    server_parameters = StdioServerParameters(command="python", args=["-c", echo_server_script])
    with MCPClient(server_parameters) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward(**{"text": "Hello, world!"}) == "Echo: Hello, world!"


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
