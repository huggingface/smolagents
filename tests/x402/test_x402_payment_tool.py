"""Tests for the x402 payment tool integration with smolagents."""

import json
import time

import pytest

from x402_payment_tool import (
    AuditEntry,
    PaymentMode,
    SpendingPolicy,
    SpendTracker,
    X402PaymentTool,
)


class TestSpendingPolicy:
    def test_default_policy_is_simulation(self):
        policy = SpendingPolicy()
        assert policy.mode == PaymentMode.SIMULATION
        assert policy.max_per_transaction == 1.00
        assert policy.rolling_cap == 10.00

    def test_custom_policy(self):
        policy = SpendingPolicy(
            mode=PaymentMode.LIVE,
            max_per_transaction=5.00,
            rolling_cap=100.00,
            merchant_allowlist=["api.example.com"],
        )
        assert policy.mode == PaymentMode.LIVE
        assert policy.merchant_allowlist == ["api.example.com"]


class TestSpendTracker:
    def test_empty_tracker(self):
        tracker = SpendTracker()
        assert tracker.get_rolling_spend(3600) == 0.0

    def test_tracks_spend(self):
        tracker = SpendTracker()
        entry = AuditEntry(
            timestamp=time.time(),
            merchant="api.example.com",
            amount=1.50,
            asset="USDC",
            network="base",
            status="simulated",
            reason="test",
        )
        tracker.add_entry(entry)
        assert tracker.get_rolling_spend(3600) == 1.50

    def test_expired_entries_excluded(self):
        tracker = SpendTracker()
        old_entry = AuditEntry(
            timestamp=time.time() - 7200,  # 2 hours ago
            merchant="api.example.com",
            amount=5.00,
            asset="USDC",
            network="base",
            status="simulated",
            reason="old",
        )
        new_entry = AuditEntry(
            timestamp=time.time(),
            merchant="api.example.com",
            amount=2.00,
            asset="USDC",
            network="base",
            status="simulated",
            reason="new",
        )
        tracker.add_entry(old_entry)
        tracker.add_entry(new_entry)
        assert tracker.get_rolling_spend(3600) == 2.00

    def test_rejected_entries_not_counted(self):
        tracker = SpendTracker()
        entry = AuditEntry(
            timestamp=time.time(),
            merchant="api.example.com",
            amount=3.00,
            asset="USDC",
            network="base",
            status="rejected",
            reason="policy",
        )
        tracker.add_entry(entry)
        assert tracker.get_rolling_spend(3600) == 0.0


class TestX402PaymentTool:
    def test_simulation_mode_default(self):
        tool = X402PaymentTool()
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.50,
            "asset": "USDC",
            "network": "base",
            "recipient": "0x1234",
        })))
        assert result["status"] == "simulated"
        assert result["amount"] == 0.50
        assert "simulated_transaction_hash" in result

    def test_informational_mode(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(mode=PaymentMode.INFORMATIONAL)
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.25,
            "asset": "USDC",
            "network": "base",
        })))
        assert result["status"] == "informational"
        assert "No payment was executed" in result["message"]

    def test_per_transaction_limit(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(max_per_transaction=1.00)
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 5.00,
        })))
        assert result["status"] == "rejected"
        assert "per-transaction limit" in result["reason"]

    def test_rolling_cap_enforcement(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                rolling_cap=2.00,
                max_per_transaction=1.50,
            )
        )
        # First transaction: should succeed (simulated)
        r1 = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 1.50,
        })))
        assert r1["status"] == "simulated"

        # Second transaction: should be rejected (would exceed cap)
        r2 = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 1.00,
        })))
        assert r2["status"] == "rejected"
        assert "rolling spend" in r2["reason"]

    def test_merchant_allowlist_blocks(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                merchant_allowlist=["trusted-api.com"],
            )
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "evil-api.com",
            "amount": 0.10,
        })))
        assert result["status"] == "rejected"
        assert "not in allowlist" in result["reason"]

    def test_merchant_allowlist_allows(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                merchant_allowlist=["trusted-api.com"],
            )
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "trusted-api.com",
            "amount": 0.10,
        })))
        assert result["status"] == "simulated"

    def test_invalid_json_handled(self):
        tool = X402PaymentTool()
        result = json.loads(tool.forward("not json"))
        assert result["status"] == "error"
        assert "Invalid JSON" in result["reason"]

    def test_audit_log(self):
        tool = X402PaymentTool()
        tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.50,
        }))
        log = tool.get_audit_log()
        assert len(log) == 1
        assert log[0]["merchant"] == "api.example.com"
        assert log[0]["status"] == "simulated"

    def test_live_mode_no_wallet_rejected(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                mode=PaymentMode.LIVE,
                require_human_approval_above=100.00,
            ),
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.10,
            "recipient": "0x1234",
        })))
        assert result["status"] == "rejected"
        assert "wallet_private_key" in result["reason"]

    def test_live_mode_human_approval_no_callback(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                mode=PaymentMode.LIVE,
                require_human_approval_above=0.01,
            ),
            wallet_private_key="0xfake",
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.50,
            "recipient": "0x1234",
        })))
        assert result["status"] == "rejected"
        assert "human_approval_callback" in result["reason"]

    def test_live_mode_human_approval_denied(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                mode=PaymentMode.LIVE,
                require_human_approval_above=0.01,
            ),
            wallet_private_key="0xfake",
            human_approval_callback=lambda req: False,
        )
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.50,
            "recipient": "0x1234",
        })))
        assert result["status"] == "rejected"
        assert "Human approval denied" in result["reason"]

    def test_legal_entity_id_in_audit(self):
        tool = X402PaymentTool(
            spending_policy=SpendingPolicy(
                legal_entity_id="corpo_ent_test123",
            )
        )
        tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.25,
        }))
        log = tool.get_audit_log()
        assert log[0]["legal_entity_id"] == "corpo_ent_test123"

    def test_fail_closed_on_policy_error(self):
        """Verify that policy engine errors result in rejection, not approval."""
        tool = X402PaymentTool()
        # Corrupt the policy to trigger an error
        tool.spending_policy.max_per_transaction = "not_a_number"  # type: ignore
        result = json.loads(tool.forward(json.dumps({
            "merchant": "api.example.com",
            "amount": 0.50,
        })))
        assert result["status"] == "rejected"
        assert "Policy engine error" in result["reason"]
