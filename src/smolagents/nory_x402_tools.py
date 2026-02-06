"""Nory x402 Payment Tools for smolagents.

Tools for AI agents to make payments using the x402 HTTP protocol.
Supports Solana and 7 EVM chains with sub-400ms settlement.
"""

import json
from typing import Literal

from smolagents.tools import Tool

__all__ = [
    "NoryGetPaymentRequirementsTool",
    "NoryVerifyPaymentTool",
    "NorySettlePaymentTool",
    "NoryLookupTransactionTool",
    "NoryHealthCheckTool",
    "get_nory_x402_tools",
]

NORY_API_BASE = "https://noryx402.com"

NoryNetwork = Literal[
    "solana-mainnet",
    "solana-devnet",
    "base-mainnet",
    "polygon-mainnet",
    "arbitrum-mainnet",
    "optimism-mainnet",
    "avalanche-mainnet",
    "sei-mainnet",
    "iotex-mainnet",
]


class NoryGetPaymentRequirementsTool(Tool):
    """Tool to get x402 payment requirements for a resource."""

    name = "nory_get_payment_requirements"
    description = """Get x402 payment requirements for accessing a paid resource.

Use this when you encounter an HTTP 402 Payment Required response and need to know
how much to pay and where to send payment. Returns payment requirements including
amount, supported networks, and wallet address.

Nory supports Solana and 7 EVM chains (Base, Polygon, Arbitrum, Optimism, Avalanche,
Sei, IoTeX) with sub-400ms settlement times."""

    inputs = {
        "resource": {
            "type": "string",
            "description": "The resource path requiring payment (e.g., /api/premium/data).",
        },
        "amount": {
            "type": "string",
            "description": "Amount in human-readable format (e.g., '0.10' for $0.10 USDC).",
        },
        "network": {
            "type": "string",
            "description": "Preferred blockchain network (optional). Options: solana-mainnet, solana-devnet, base-mainnet, polygon-mainnet, arbitrum-mainnet, optimism-mainnet, avalanche-mainnet, sei-mainnet, iotex-mainnet.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        """Initialize the tool.

        Args:
            api_key: Nory API key (optional for public endpoints).
        """
        super().__init__(**kwargs)
        self.api_key = api_key

    def forward(self, resource: str, amount: str, network: str | None = None) -> str:
        """Get payment requirements for a resource.

        Args:
            resource: The resource path requiring payment.
            amount: Amount in human-readable format.
            network: Preferred blockchain network (optional).

        Returns:
            JSON string with payment requirements.
        """
        import requests

        params = {"resource": resource, "amount": amount}
        if network:
            params["network"] = network

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            f"{NORY_API_BASE}/api/x402/requirements",
            params=params,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


class NoryVerifyPaymentTool(Tool):
    """Tool to verify a signed payment transaction."""

    name = "nory_verify_payment"
    description = """Verify a signed payment transaction before settlement.

Use this to validate that a payment transaction is correct before submitting
it to the blockchain. Returns verification result including validity and payer info."""

    inputs = {
        "payload": {
            "type": "string",
            "description": "Base64-encoded payment payload containing signed transaction.",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        """Initialize the tool.

        Args:
            api_key: Nory API key (optional for public endpoints).
        """
        super().__init__(**kwargs)
        self.api_key = api_key

    def forward(self, payload: str) -> str:
        """Verify a payment transaction.

        Args:
            payload: Base64-encoded payment payload.

        Returns:
            JSON string with verification result.
        """
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{NORY_API_BASE}/api/x402/verify",
            json={"payload": payload},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


class NorySettlePaymentTool(Tool):
    """Tool to settle a payment on-chain."""

    name = "nory_settle_payment"
    description = """Settle a payment on-chain with ~400ms settlement time.

Use this to submit a verified payment transaction to the blockchain.
Settlement typically completes in under 400ms. Returns settlement result
including transaction ID."""

    inputs = {
        "payload": {
            "type": "string",
            "description": "Base64-encoded payment payload.",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        """Initialize the tool.

        Args:
            api_key: Nory API key (optional for public endpoints).
        """
        super().__init__(**kwargs)
        self.api_key = api_key

    def forward(self, payload: str) -> str:
        """Settle a payment on-chain.

        Args:
            payload: Base64-encoded payment payload.

        Returns:
            JSON string with settlement result.
        """
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{NORY_API_BASE}/api/x402/settle",
            json={"payload": payload},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


class NoryLookupTransactionTool(Tool):
    """Tool to look up transaction status."""

    name = "nory_lookup_transaction"
    description = """Look up the status of a previously submitted payment transaction.

Use this to check the status of a payment including confirmations and
current state (pending, confirmed, failed)."""

    inputs = {
        "transaction_id": {
            "type": "string",
            "description": "Transaction ID or signature.",
        },
        "network": {
            "type": "string",
            "description": "Network where the transaction was submitted. Options: solana-mainnet, solana-devnet, base-mainnet, polygon-mainnet, arbitrum-mainnet, optimism-mainnet, avalanche-mainnet, sei-mainnet, iotex-mainnet.",
        },
    }
    output_type = "string"

    def __init__(self, api_key: str | None = None, **kwargs):
        """Initialize the tool.

        Args:
            api_key: Nory API key (optional for public endpoints).
        """
        super().__init__(**kwargs)
        self.api_key = api_key

    def forward(self, transaction_id: str, network: str) -> str:
        """Look up transaction status.

        Args:
            transaction_id: Transaction ID or signature.
            network: Network where the transaction was submitted.

        Returns:
            JSON string with transaction status.
        """
        import requests

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.get(
            f"{NORY_API_BASE}/api/x402/transactions/{transaction_id}",
            params={"network": network},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


class NoryHealthCheckTool(Tool):
    """Tool to check Nory service health."""

    name = "nory_health_check"
    description = """Check Nory service health and see supported networks.

Use this to verify the payment service is operational before attempting
payments. Returns health status and list of supported blockchain networks."""

    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        """Check Nory service health.

        Returns:
            JSON string with health status and supported networks.
        """
        import requests

        response = requests.get(f"{NORY_API_BASE}/api/x402/health", timeout=30)
        response.raise_for_status()
        return json.dumps(response.json(), indent=2)


def get_nory_x402_tools(api_key: str | None = None) -> list[Tool]:
    """Get all Nory x402 payment tools.

    Args:
        api_key: Nory API key (optional for public endpoints).

    Returns:
        List of all Nory x402 tools.
    """
    return [
        NoryGetPaymentRequirementsTool(api_key=api_key),
        NoryVerifyPaymentTool(api_key=api_key),
        NorySettlePaymentTool(api_key=api_key),
        NoryLookupTransactionTool(api_key=api_key),
        NoryHealthCheckTool(),
    ]
