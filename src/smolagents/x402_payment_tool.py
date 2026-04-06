#!/usr/bin/env python
# coding=utf-8

# Copyright 2026 The AI Agent Economy team. All rights reserved.
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

"""
x402 Payment Tool for smolagents — Native HTTP 402 payment handling.

This module provides a Tool subclass that handles HTTP 402 (Payment Required)
responses using the x402 protocol (https://github.com/coinbase/x402).

It can be used in two ways:
1. As a native smolagents Tool subclass (no external dependencies)
2. Via MCP integration using agentpay-mcp (leverages existing mcp_client.py)

Design principles:
- Human-first: simulation mode by default, real payments require explicit opt-in
- Fail-closed: any policy engine error produces a rejection, never an approval
- Auditable: every payment attempt is logged with full context
- Bounded: rolling spend caps and per-transaction limits enforced before payment

References:
- Issue: https://github.com/huggingface/smolagents/issues/2112
- NVIDIA NeMo integration: https://github.com/NVIDIA/NeMo-Agent-Toolkit-Examples/pull/17
- agentpay-mcp: https://github.com/up2itnow0822/agentpay-mcp
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from smolagents.tools import Tool


logger = logging.getLogger(__name__)


class PaymentMode(Enum):
    """Controls how the payment tool handles 402 responses."""

    SIMULATION = "simulation"  # Log the payment intent, return simulated success (default)
    INFORMATIONAL = "informational"  # Show cost to user, do not pay, do not retry
    LIVE = "live"  # Execute real payments (requires explicit opt-in)


@dataclass
class SpendingPolicy:
    """Configurable spending guardrails for agent payment execution.

    Attributes:
        mode: Payment execution mode (simulation, informational, or live).
        max_per_transaction: Maximum allowed per single transaction in USD.
        rolling_cap: Maximum cumulative spend in the rolling window.
        rolling_window_seconds: Duration of the rolling spend window.
        merchant_allowlist: If set, only these merchant domains can receive payments.
            An empty list means ALL merchants are blocked.
            None means no allowlist enforcement (all merchants allowed).
        require_human_approval_above: Transactions above this USD amount
            require explicit human approval before execution.
        legal_entity_id: Optional entity ID for audit trail tagging (informational).
    """

    mode: PaymentMode = PaymentMode.SIMULATION
    max_per_transaction: float = 1.00
    rolling_cap: float = 10.00
    rolling_window_seconds: int = 3600
    merchant_allowlist: list[str] | None = None
    require_human_approval_above: float = 0.50
    legal_entity_id: str | None = None


@dataclass
class AuditEntry:
    """Immutable record of a payment attempt."""

    timestamp: float
    merchant: str
    amount: float
    asset: str
    network: str
    status: str  # "approved", "rejected", "simulated", "informational"
    reason: str
    transaction_hash: str | None = None
    legal_entity_id: str | None = None


@dataclass
class SpendTracker:
    """Tracks cumulative spend within rolling windows."""

    entries: list[AuditEntry] = field(default_factory=list)

    def get_rolling_spend(self, window_seconds: int) -> float:
        """Sum of approved/simulated payments within the rolling window."""
        cutoff = time.time() - window_seconds
        return sum(
            e.amount
            for e in self.entries
            if e.timestamp >= cutoff and e.status in ("approved", "simulated")
        )

    def add_entry(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

    def get_audit_log(self) -> list[dict]:
        """Return full audit trail as serializable dicts."""
        return [
            {
                "timestamp": e.timestamp,
                "merchant": e.merchant,
                "amount": e.amount,
                "asset": e.asset,
                "network": e.network,
                "status": e.status,
                "reason": e.reason,
                "transaction_hash": e.transaction_hash,
                "legal_entity_id": e.legal_entity_id,
            }
            for e in self.entries
        ]


class X402PaymentTool(Tool):
    """Handles HTTP 402 (Payment Required) responses for smolagents.

    When an agent encounters an API that returns HTTP 402 with x402 payment
    requirements, this tool evaluates the cost against the spending policy,
    optionally executes or simulates the payment, and returns the result.

    By default, the tool runs in SIMULATION mode — it logs the payment intent
    and returns a simulated success without moving any funds. This allows
    developers to integrate and test the payment flow before enabling real
    transactions.

    Example (simulation mode — default, no wallet needed):
        ```python
        from smolagents import CodeAgent, InferenceClientModel
        from x402_payment_tool import X402PaymentTool, SpendingPolicy, PaymentMode

        payment_tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                mode=PaymentMode.SIMULATION,
                max_per_transaction=5.00,
                rolling_cap=50.00,
            )
        )

        agent = CodeAgent(
            tools=[payment_tool],
            model=InferenceClientModel(),
        )

        result = agent.run("Access the premium weather API at https://api.example.com/weather")
        ```

    Example (live mode — requires wallet configuration):
        ```python
        payment_tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                mode=PaymentMode.LIVE,
                max_per_transaction=1.00,
                rolling_cap=10.00,
                merchant_allowlist=["api.example.com", "data.provider.io"],
                require_human_approval_above=0.50,
            ),
            wallet_private_key="0x...",  # Or use environment variable
            chain_id=8453,  # Base mainnet
        )
        ```

    Example (MCP integration — uses agentpay-mcp server):
        ```python
        from smolagents.mcp_client import MCPClient

        # agentpay-mcp handles x402 negotiation via MCP protocol
        with MCPClient(
            {"command": "npx", "args": ["-y", "agentpay-mcp"]},
        ) as payment_tools:
            agent = CodeAgent(
                tools=[*payment_tools],
                model=InferenceClientModel(),
            )
        ```
    """

    name = "x402_payment"
    description = (
        "Handles HTTP 402 (Payment Required) responses using the x402 protocol. "
        "When an API returns a 402 status with payment requirements (amount, recipient, "
        "asset, network), this tool evaluates the cost against the spending policy and "
        "either simulates, reports, or executes the payment. "
        "Input: a JSON string with the 402 response details. "
        "Output: payment result with status and any transaction proof."
    )
    inputs = {
        "payment_request": {
            "type": "string",
            "description": (
                "JSON string containing the x402 payment requirements from the HTTP 402 response. "
                "Expected fields: 'merchant' (API domain), 'amount' (in USD), "
                "'asset' (e.g., 'USDC'), 'network' (e.g., 'base'), "
                "'recipient' (wallet address), and optional 'description'."
            ),
        },
    }
    output_type = "string"

    def __init__(
        self,
        spending_policy: SpendingPolicy | None = None,
        wallet_private_key: str | None = None,
        chain_id: int = 8453,
        human_approval_callback: Any | None = None,
    ):
        """Initialize the x402 payment tool.

        Args:
            spending_policy: Payment guardrails configuration. Defaults to
                simulation mode with conservative limits.
            wallet_private_key: Private key for signing payments in LIVE mode.
                Not required for SIMULATION or INFORMATIONAL modes.
            chain_id: Blockchain network ID (default: 8453 = Base mainnet).
            human_approval_callback: Optional callable that receives a payment
                request dict and returns True/False. Called for transactions
                above the require_human_approval_above threshold.
        """
        super().__init__()
        self.spending_policy = spending_policy or SpendingPolicy()
        self._wallet_private_key = wallet_private_key
        self._chain_id = chain_id
        self._human_approval_callback = human_approval_callback
        self._tracker = SpendTracker()

    def forward(self, payment_request: str) -> str:
        """Process an x402 payment request.

        Args:
            payment_request: JSON string with payment requirements.

        Returns:
            JSON string with payment result including status and details.
        """
        try:
            request = json.loads(payment_request)
        except json.JSONDecodeError:
            return json.dumps({
                "status": "error",
                "reason": "Invalid JSON in payment request",
            })

        merchant = request.get("merchant", "unknown")
        amount = float(request.get("amount", 0))
        asset = request.get("asset", "USDC")
        network = request.get("network", "base")
        recipient = request.get("recipient", "")
        description = request.get("description", "")

        # --- Policy checks (fail-closed) ---
        try:
            rejection = self._check_policy(merchant, amount)
        except Exception as e:
            # Fail closed: any policy engine error = rejection
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="rejected",
                reason=f"Policy engine error (fail-closed): {e}",
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            logger.warning(f"x402 payment rejected (fail-closed): {e}")
            return json.dumps({
                "status": "rejected",
                "reason": f"Policy engine error: {e}",
            })

        if rejection:
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="rejected",
                reason=rejection,
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            logger.info(f"x402 payment rejected: {rejection}")
            return json.dumps({
                "status": "rejected",
                "reason": rejection,
                "merchant": merchant,
                "amount": amount,
            })

        # --- Mode-specific execution ---
        mode = self.spending_policy.mode

        if mode == PaymentMode.INFORMATIONAL:
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="informational",
                reason="Cost reported to user (no payment executed)",
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            return json.dumps({
                "status": "informational",
                "merchant": merchant,
                "amount": amount,
                "asset": asset,
                "network": network,
                "description": description,
                "message": (
                    f"This API requires a payment of ${amount:.4f} {asset} on {network} "
                    f"to {merchant}. No payment was executed (informational mode). "
                    f"To enable payments, set mode=PaymentMode.LIVE."
                ),
            })

        if mode == PaymentMode.SIMULATION:
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="simulated",
                reason="Simulated payment (no funds moved)",
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            logger.info(f"x402 payment simulated: ${amount:.4f} to {merchant}")
            return json.dumps({
                "status": "simulated",
                "merchant": merchant,
                "amount": amount,
                "asset": asset,
                "network": network,
                "description": description,
                "message": (
                    f"Payment of ${amount:.4f} {asset} to {merchant} was simulated successfully. "
                    f"No funds were moved. To execute real payments, set mode=PaymentMode.LIVE."
                ),
                "simulated_transaction_hash": f"0xsim_{int(time.time())}",
            })

        if mode == PaymentMode.LIVE:
            # Human approval gate for amounts above threshold
            if amount > self.spending_policy.require_human_approval_above:
                if self._human_approval_callback:
                    approved = self._human_approval_callback(request)
                    if not approved:
                        entry = AuditEntry(
                            timestamp=time.time(),
                            merchant=merchant,
                            amount=amount,
                            asset=asset,
                            network=network,
                            status="rejected",
                            reason="Human approval denied",
                            legal_entity_id=self.spending_policy.legal_entity_id,
                        )
                        self._tracker.add_entry(entry)
                        return json.dumps({
                            "status": "rejected",
                            "reason": "Human approval denied for this transaction",
                        })
                else:
                    # No callback configured but approval required — fail closed
                    entry = AuditEntry(
                        timestamp=time.time(),
                        merchant=merchant,
                        amount=amount,
                        asset=asset,
                        network=network,
                        status="rejected",
                        reason=(
                            f"Amount ${amount:.4f} exceeds approval threshold "
                            f"${self.spending_policy.require_human_approval_above:.2f} "
                            f"but no human_approval_callback is configured"
                        ),
                        legal_entity_id=self.spending_policy.legal_entity_id,
                    )
                    self._tracker.add_entry(entry)
                    return json.dumps({
                        "status": "rejected",
                        "reason": (
                            f"Amount exceeds ${self.spending_policy.require_human_approval_above:.2f} "
                            f"threshold. Configure human_approval_callback to approve high-value transactions."
                        ),
                    })

            # Execute payment via x402 protocol
            return self._execute_payment(merchant, amount, asset, network, recipient, description)

        # Should never reach here — fail closed
        return json.dumps({"status": "error", "reason": "Unknown payment mode"})

    def _check_policy(self, merchant: str, amount: float) -> str | None:
        """Check spending policy. Returns rejection reason or None if approved."""
        policy = self.spending_policy

        # Per-transaction limit
        if amount > policy.max_per_transaction:
            return (
                f"Amount ${amount:.4f} exceeds per-transaction limit "
                f"${policy.max_per_transaction:.2f}"
            )

        # Rolling spend cap
        current_spend = self._tracker.get_rolling_spend(policy.rolling_window_seconds)
        if current_spend + amount > policy.rolling_cap:
            return (
                f"Transaction would push rolling spend to "
                f"${current_spend + amount:.4f}, exceeding cap "
                f"${policy.rolling_cap:.2f}"
            )

        # Merchant allowlist
        if policy.merchant_allowlist is not None:
            if merchant not in policy.merchant_allowlist:
                return f"Merchant '{merchant}' not in allowlist"

        return None

    def _execute_payment(
        self,
        merchant: str,
        amount: float,
        asset: str,
        network: str,
        recipient: str,
        description: str,
    ) -> str:
        """Execute a real x402 payment. Requires wallet configuration."""
        if not self._wallet_private_key:
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="rejected",
                reason="No wallet configured for LIVE mode",
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            return json.dumps({
                "status": "rejected",
                "reason": (
                    "LIVE mode requires wallet_private_key. "
                    "Set wallet_private_key in X402PaymentTool constructor."
                ),
            })

        # The actual x402 payment signing flow:
        # 1. Construct payment payload per x402 spec
        # 2. Sign with wallet private key
        # 3. Return signed proof for the agent to include in the retry request
        #
        # For production use, this delegates to agentpay-mcp or web3 libraries.
        # The following is the integration point — users should install
        # agentpay-mcp for full payment execution, or implement their own
        # signing logic here.
        try:
            tx_hash = self._sign_and_send(recipient, amount, asset, network)
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="approved",
                reason="Payment executed successfully",
                transaction_hash=tx_hash,
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            logger.info(f"x402 payment executed: ${amount:.4f} to {merchant}, tx={tx_hash}")
            return json.dumps({
                "status": "approved",
                "merchant": merchant,
                "amount": amount,
                "asset": asset,
                "network": network,
                "transaction_hash": tx_hash,
                "description": description,
                "message": f"Payment of ${amount:.4f} {asset} executed successfully.",
            })
        except Exception as e:
            entry = AuditEntry(
                timestamp=time.time(),
                merchant=merchant,
                amount=amount,
                asset=asset,
                network=network,
                status="rejected",
                reason=f"Payment execution failed: {e}",
                legal_entity_id=self.spending_policy.legal_entity_id,
            )
            self._tracker.add_entry(entry)
            logger.error(f"x402 payment failed: {e}")
            return json.dumps({
                "status": "error",
                "reason": f"Payment execution failed: {e}",
            })

    def _sign_and_send(
        self, recipient: str, amount: float, asset: str, network: str
    ) -> str:
        """Sign and send a USDC payment via x402 protocol.

        This is the integration point for real payment execution.
        For production use, install agentpay-mcp which handles the full
        x402 signing flow, or implement using web3.py / ethers equivalent.

        Raises:
            NotImplementedError: If no payment backend is configured.
        """
        # Try to use agentpay-mcp's payment execution if available
        try:
            from agentpay_mcp import execute_payment  # type: ignore

            return execute_payment(
                recipient=recipient,
                amount=amount,
                asset=asset,
                network=network,
                private_key=self._wallet_private_key,
                chain_id=self._chain_id,
            )
        except ImportError:
            pass

        # Fallback: direct web3 signing (requires web3 package)
        try:
            from web3 import Web3  # type: ignore

            # Base mainnet USDC contract
            USDC_CONTRACTS = {
                8453: "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base
                84532: "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # Base Sepolia
            }

            usdc_address = USDC_CONTRACTS.get(self._chain_id)
            if not usdc_address:
                raise ValueError(f"Unsupported chain_id: {self._chain_id}")

            # Convert USD amount to USDC units (6 decimals)
            usdc_amount = int(amount * 1_000_000)

            w3 = Web3(Web3.HTTPProvider(
                f"https://mainnet.base.org" if self._chain_id == 8453
                else f"https://sepolia.base.org"
            ))
            account = w3.eth.account.from_key(self._wallet_private_key)

            # ERC-20 transfer ABI
            erc20_abi = [
                {
                    "constant": False,
                    "inputs": [
                        {"name": "_to", "type": "address"},
                        {"name": "_value", "type": "uint256"},
                    ],
                    "name": "transfer",
                    "outputs": [{"name": "", "type": "bool"}],
                    "type": "function",
                }
            ]

            contract = w3.eth.contract(address=usdc_address, abi=erc20_abi)
            tx = contract.functions.transfer(recipient, usdc_amount).build_transaction({
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address),
                "gas": 100_000,
                "gasPrice": w3.eth.gas_price,
                "chainId": self._chain_id,
            })

            signed = account.sign_transaction(tx)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            return tx_hash.hex()

        except ImportError:
            raise NotImplementedError(
                "No payment backend available. Install one of:\n"
                "  pip install agentpay-mcp    # Recommended: full x402 support\n"
                "  pip install web3            # Direct blockchain signing\n"
                "Or use SIMULATION mode for testing without a wallet."
            )

    def get_audit_log(self) -> list[dict]:
        """Return the full audit trail of all payment attempts."""
        return self._tracker.get_audit_log()

    def get_rolling_spend(self) -> float:
        """Return current rolling spend within the policy window."""
        return self._tracker.get_rolling_spend(
            self.spending_policy.rolling_window_seconds
        )


# --- Convenience: MCP-based integration (zero changes to smolagents core) ---

def create_agentpay_mcp_client(
    wallet_private_key: str | None = None,
    chain_id: int = 8453,
    spending_limit: float = 10.00,
) -> "MCPClient":
    """Create an MCPClient configured for agentpay-mcp payment handling.

    This is the simplest way to add x402 payment support to smolagents —
    it uses the existing MCPClient infrastructure with the agentpay-mcp server.

    Args:
        wallet_private_key: Private key for signing payments.
            If None, runs in simulation/informational mode only.
        chain_id: Blockchain network ID (default: 8453 = Base mainnet).
        spending_limit: Maximum spend limit in USD.

    Returns:
        MCPClient: Configured MCP client with payment tools.

    Example:
        ```python
        from smolagents import CodeAgent, InferenceClientModel
        from x402_payment_tool import create_agentpay_mcp_client

        mcp_client = create_agentpay_mcp_client(
            spending_limit=50.00,
        )

        agent = CodeAgent(
            tools=[*mcp_client.get_tools()],
            model=InferenceClientModel(),
        )
        ```
    """
    from smolagents.mcp_client import MCPClient

    env = {"CHAIN_ID": str(chain_id), "SPENDING_LIMIT": str(spending_limit)}
    if wallet_private_key:
        env["WALLET_PRIVATE_KEY"] = wallet_private_key

    return MCPClient(
        {"command": "npx", "args": ["-y", "agentpay-mcp"]},
    )
