"""Example: Using x402 payment handling with smolagents.

This example demonstrates three ways to add payment capabilities to smolagents:
1. Native Tool subclass (simulation mode — no wallet needed)
2. Native Tool subclass (live mode — requires wallet)
3. MCP integration via agentpay-mcp (recommended for production)
"""

from smolagents import CodeAgent, InferenceClientModel

# ==============================================================================
# Example 1: Simulation mode (default — safe for testing)
# ==============================================================================
#
# The tool logs all payment intents and returns simulated success.
# No wallet, no real money, no risk. Perfect for development.

from x402_payment_tool import PaymentMode, SpendingPolicy, X402PaymentTool

payment_tool = X402PaymentTool(
    spending_policy=SpendingPolicy(
        mode=PaymentMode.SIMULATION,
        max_per_transaction=5.00,
        rolling_cap=50.00,
        rolling_window_seconds=3600,
    )
)

agent = CodeAgent(
    tools=[payment_tool],
    model=InferenceClientModel(),
)

# The agent can now handle APIs that return HTTP 402
# In simulation mode, it will "pay" without moving real funds
result = agent.run(
    "Check the simulated cost of accessing the premium weather API. "
    "The API returned a 402 with: merchant=weather.example.com, amount=0.01, "
    "asset=USDC, network=base, recipient=0xWeatherAPI"
)
print(result)

# Check what happened
print("\nAudit log:")
for entry in payment_tool.get_audit_log():
    print(f"  {entry['merchant']}: ${entry['amount']:.4f} ({entry['status']})")
print(f"Rolling spend: ${payment_tool.get_rolling_spend():.4f}")


# ==============================================================================
# Example 2: Informational mode (cost transparency, no payments)
# ==============================================================================
#
# Shows the user what an API would cost without paying or simulating.

informational_tool = X402PaymentTool(
    spending_policy=SpendingPolicy(
        mode=PaymentMode.INFORMATIONAL,
    )
)

# This will report the cost but never attempt payment
# Useful for cost estimation and budget planning


# ==============================================================================
# Example 3: Live mode with human approval (production)
# ==============================================================================
#
# Real payments with explicit guardrails:
# - Per-transaction limit
# - Rolling spend cap
# - Merchant allowlist
# - Human approval for amounts above threshold

def ask_human(request: dict) -> bool:
    """Simple human-in-the-loop approval."""
    print(f"\n⚠️  Payment approval required:")
    print(f"  Merchant: {request.get('merchant')}")
    print(f"  Amount: ${request.get('amount', 0):.4f} {request.get('asset', 'USDC')}")
    print(f"  Network: {request.get('network', 'base')}")
    response = input("  Approve? (y/n): ").strip().lower()
    return response == "y"


# Uncomment to use live mode (requires real wallet):
#
# live_tool = X402PaymentTool(
#     spending_policy=SpendingPolicy(
#         mode=PaymentMode.LIVE,
#         max_per_transaction=1.00,
#         rolling_cap=10.00,
#         merchant_allowlist=["api.trusted-provider.com", "data.example.com"],
#         require_human_approval_above=0.50,
#     ),
#     wallet_private_key="0x...",  # Your wallet private key
#     chain_id=8453,  # Base mainnet
#     human_approval_callback=ask_human,
# )


# ==============================================================================
# Example 4: MCP integration (recommended for production)
# ==============================================================================
#
# Uses agentpay-mcp server via smolagents' existing MCPClient.
# This is the simplest path — no code changes to smolagents core.

# from x402_payment_tool import create_agentpay_mcp_client
#
# mcp_client = create_agentpay_mcp_client(
#     spending_limit=50.00,
# )
#
# agent = CodeAgent(
#     tools=[*mcp_client.get_tools()],
#     model=InferenceClientModel(),
# )
