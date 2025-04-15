# TODO(@grll): add tests for the MCPClient when designed accepted.
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
    serverparams = StdioServerParameters(command="uv", args=["run", "python", "-c", echo_server_script])
    with MCPClient(serverparams) as tools:
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward({"text": "Hello, world!"}) == "Echo: Hello, world!"


def test_mcp_client_try_finally_syntax(echo_server_script: str):
    """Test the MCPClient with the try ... finally syntax."""
    serverparams = StdioServerParameters(command="uv", args=["run", "python", "-c", echo_server_script])
    mcp_client = MCPClient(serverparams)
    try:
        tools = mcp_client.tools
        assert len(tools) == 1
        assert tools[0].name == "echo_tool"
        assert tools[0].forward({"text": "Hello, world!"}) == "Echo: Hello, world!"
    finally:
        mcp_client.disconnect()
