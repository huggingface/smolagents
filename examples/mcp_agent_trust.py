"""Example: Using TWZRD Agent Intel MCP server with smolagents.

TWZRD Agent Intel (https://intel.twzrd.xyz) scores autonomous AI agents on Solana
based on real transaction history. Use it to verify agent trust before routing
value through an x402 workflow or any agent-to-agent interaction.

This example connects a CodeAgent to the TWZRD MCP server via streamable-http
and asks it to evaluate a known agent wallet.

Install:
    pip install smolagents mcp
"""

from smolagents import CodeAgent, InferenceClientModel, MCPClient


# TWZRD Agent Intel MCP server — free tools, no API key required
mcp_server_parameters = {
    "url": "https://intel.twzrd.xyz/mcp",
    "transport": "streamable-http",
}

mcp_client = MCPClient(server_parameters=mcp_server_parameters)

model = InferenceClientModel()

agent = CodeAgent(
    model=model,
    tools=mcp_client.get_tools(),
)

# score_agent returns a trust score (0–100) derived from on-chain transaction history.
# resolve_agent maps an arbitrary identifier (Twitter handle, ENS, .sol domain) to a wallet.
# preflight_check returns human-readable context on a wallet before committing to a payment.
result = agent.run(
    "Use the score_agent tool to get the trust score for wallet "
    "D1QkbFJKiPsymJ65RKHhF6DFB8sPMfpBaFBzuHKfJGWi, "
    "then explain what the score means."
)

print(result)

mcp_client.disconnect()
