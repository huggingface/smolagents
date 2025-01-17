"""An example of loading a ToolCollection directly from an MCP server.

Requirements: to run this example, you need to have uv installed and in your path in
order to run the MCP server with uvx see `mcp_server_params` below.

Note this is just a demo MCP server that was implemented for the purpose of this example.
It only provide a single tool to search amongst pubmed papers abstracts.

Usage:
>>> uv run examples/tool_calling_agent_mcp.py
"""

import os

from mcp import StdioServerParameters
from smolagents import CodeAgent, HfApiModel, ToolCollection

mcp_server_params = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(mcp_server_params) as tool_collection:
    # print(tool_collection.tools[0](request={"term": "efficient treatment hangover"}))
    agent = CodeAgent(tools=tool_collection.tools, model=HfApiModel())
    agent.run("Find studies about hangover?")
