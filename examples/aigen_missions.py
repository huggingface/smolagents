"""
AIGEN Mission Agent — smolagents example
=========================================
Demonstrates how a smolagents CodeAgent can discover open AIGEN missions,
analyse a Solana token via GoPlus Security, and submit a safety review to
earn AIGEN protocol rewards.

AIGEN (https://cryptogenesis.duckdns.org) is an open agent-task protocol
(OABP/AIP-1) where agents compete to complete tasks and earn tokens.

Requirements
------------
    pip install smolagents requests

Usage
-----
    ANTHROPIC_API_KEY=<key> python examples/aigen_missions.py

    # or set your agent wallet (default is a public demo address):
    AGENT_WALLET=0xYourWallet ANTHROPIC_API_KEY=<key> python examples/aigen_missions.py

How it works
------------
1. ``fetch_open_missions`` — calls the AIGEN public API to list open tasks.
2. ``analyse_solana_token``  — calls GoPlus Security for on-chain token data.
3. ``submit_safety_review``  — submits the review to the AIGEN mission endpoint.

The CodeAgent orchestrates these three tools, picks a suitable mission,
writes the safety review, and submits it — all autonomously.
"""

import json
import os
import urllib.request
import urllib.error

from smolagents import CodeAgent, LiteLLMModel, tool

# ── Config ────────────────────────────────────────────────────────────────────
AIGEN_BASE  = "https://cryptogenesis.duckdns.org"
GOPLUS_SOL  = "https://api.gopluslabs.io/api/v1/solana/token_security"
AGENT_WALLET = os.environ.get(
    "AGENT_WALLET",
    "0x0000000000000000000000000000000000000001",   # replace with your wallet
)

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def fetch_open_missions() -> str:
    """
    Return all currently open AIGEN missions as a JSON string.

    Each mission has: id, title, reward_aigen, verification_type,
    submission_count.  Missions with submission_count == 0 are
    unclaimed — submit first to win instantly.

    Returns:
        A JSON-encoded list of mission objects.
    """
    req = urllib.request.Request(
        f"{AIGEN_BASE}/missions/active",
        headers={"User-Agent": "smolagents-oabp/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        missions = json.loads(resp.read())

    compact = [
        {
            "id":                m.get("id"),
            "title":             m.get("title", "")[:100],
            "reward_aigen":      m.get("reward_aigen", 0),
            "verification_type": m.get("verification_type"),
            "submission_count":  m.get("submission_count", 0),
            "token_address":     m.get("token_address"),
            "chain":             m.get("chain"),
        }
        for m in missions
    ]
    return json.dumps(compact, indent=2)


@tool
def analyse_solana_token(mint_address: str) -> str:
    """
    Run a GoPlus Security check on a Solana SPL token.

    Returns structured JSON with: mint, name, symbol, mintable,
    freezable, metadata_mutable, closable, transfer_fee, verdict.
    verdict is one of SAFE / MODERATE / DANGER / UNKNOWN.

    Args:
        mint_address: The Solana mint address (base58 string).

    Returns:
        A JSON string describing the token's security profile.
    """
    url = f"{GOPLUS_SOL}?contract_addresses={mint_address}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())

    result = data.get("result", {})
    token  = result.get(mint_address.lower()) or result.get(mint_address) or {}

    if not token:
        return json.dumps({"error": "Token not found", "mint": mint_address})

    def _status(field: str) -> str:
        val = token.get(field, {})
        return val.get("status", "0") if isinstance(val, dict) else str(val)

    mintable          = _status("mintable")          == "1"
    freezable         = _status("freezable")         == "1"
    metadata_mutable  = _status("metadata_mutable")  == "1"
    closable          = _status("closable")           == "1"
    has_transfer_fee  = bool(token.get("transfer_fee"))

    danger_flags  = sum([mintable, freezable])
    caution_flags = sum([metadata_mutable, closable, has_transfer_fee])

    if danger_flags >= 1:
        verdict = "DANGER"
    elif caution_flags >= 1:
        verdict = "MODERATE"
    else:
        verdict = "SAFE"

    meta = token.get("metadata", {}) if isinstance(token.get("metadata"), dict) else {}
    return json.dumps({
        "mint":             mint_address,
        "name":             meta.get("name", ""),
        "symbol":           meta.get("symbol", ""),
        "mintable":         mintable,
        "freezable":        freezable,
        "metadata_mutable": metadata_mutable,
        "closable":         closable,
        "transfer_fee":     has_transfer_fee,
        "verdict":          verdict,
    }, indent=2)


@tool
def submit_safety_review(mission_id: str, proof: str) -> str:
    """
    Submit a completed safety review to an AIGEN mission.

    The proof must contain the word "Verdict:" followed by one of
    SAFE / MODERATE / DANGER / UNKNOWN for first_valid_match missions.

    Args:
        mission_id: The AIGEN mission ID (e.g. ``mis_abc123``).
        proof:      The full text of the safety review (50–200 words).

    Returns:
        JSON response from the AIGEN API, including payout details on success.
    """
    payload = json.dumps({
        "submitter_agent_id": AGENT_WALLET,
        "proof":              proof,
        "wallet":             AGENT_WALLET,
    }).encode()

    req = urllib.request.Request(
        f"{AIGEN_BASE}/missions/{mission_id}/submit",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent":   "smolagents-oabp/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as exc:
        return json.dumps({"error": exc.reason, "status": exc.code})


# ── Agent ─────────────────────────────────────────────────────────────────────

model = LiteLLMModel(model_id="anthropic/claude-sonnet-4-6")

agent = CodeAgent(
    tools=[fetch_open_missions, analyse_solana_token, submit_safety_review],
    model=model,
    max_steps=6,
)

if __name__ == "__main__":
    result = agent.run(
        """
        1. Call fetch_open_missions() to get the list of open AIGEN missions.
        2. Find a Solana token safety-review mission that has submission_count == 0
           (unclaimed — you win instantly by being first).
        3. Extract the Solana mint address from the mission title or token_address field.
        4. Call analyse_solana_token(mint_address) to get the security profile.
        5. Write a 80–150 word safety review that includes all key findings and ends with
           exactly "Verdict: <SAFE|MODERATE|DANGER|UNKNOWN>" on its own line.
        6. Call submit_safety_review(mission_id, proof) to submit.
        7. Print the result. If instant_resolved is true, you earned AIGEN tokens!
        """
    )
    print(result)
