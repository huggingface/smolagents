"""Attach a cryptographic Proof-of-Time timestamp to an agent's output via MCP.

This example shows how any smolagents agent can gain access to a Proof-of-Time
(PoT) tool over MCP, so it can attach a tamper-evident timestamp receipt to a
piece of content it produces (a summary, a computation, a generated file) --
useful when you need to later prove *when* an agent's output was created and
that it wasn't altered afterward.

The MCP server (`@helm-protocol/ttt-mcp`, npm, MIT license,
https://github.com/Helm-Protocol/openttt-mcp -- reference implementation of
IETF draft-helmprotocol-tttps, https://datatracker.ietf.org/doc/draft-helmprotocol-tttps/)
is spawned automatically via `npx` -- no server setup required, and it works
through `ToolCollection.from_mcp()` with zero adapter code.

Note: the timestamp proves *when* content existed and that it hasn't been
tampered with since -- it does not verify the correctness of the content
itself.

Requirements:
    pip install smolagents mcp
    Node.js + npx available on PATH (to run the MCP server)
"""
import os

from mcp import StdioServerParameters
from smolagents import CodeAgent, InferenceClientModel, ToolCollection

server_parameters = StdioServerParameters(
    command="npx",
    args=["-y", "@helm-protocol/ttt-mcp"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=InferenceClientModel())
    result = agent.run(
        "Summarize in one sentence why cryptographic timestamps matter for AI "
        "agent outputs, then use the available Proof-of-Time tool to generate "
        "a verifiable timestamp for a SHA-256 digest of your own summary."
    )
    print(result)
