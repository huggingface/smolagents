# Verifiable receipts via `ChainAnchorCallback`

[[open-in-colab]]

> [!TIP]
> If you're new to smolagents, start with the [intro to agents](../conceptual_guides/intro_agents) and the [guided tour](../guided_tour).

## Why anchor agent runs?

In regulated or cross-organization deployments, agent runs sometimes need an audit trail that doesn't depend on the operator's word, the operator's logs, or the operator's continued operation. The agent did *something* — for compliance, dispute resolution, or simple accountability, a third party may need to verify, weeks or years later, *what the agent committed to* at a given point in time.

A common shape for this is to compute a hash over each step, build a merkle root over the run, and commit that root somewhere durable: a public chain, an RFC 3161 trusted timestamping authority, or a notary service. The receipt sidecar travels with the run; a verifier with just the sidecar and one external lookup can reconstruct what the agent did.

smolagents already exposes a [`step_callbacks`](../reference/agents#smolagents.MultiStepAgent.step_callbacks) hook on every agent. The [`ChainAnchorCallback`] is a small utility that snaps into that hook and emits a [`chain-anchor/v1`](https://proof.satsignal.cloud/spec-chain-anchor) sidecar for each run.

smolagents takes no network dependency itself. The anchor backend — Satsignal, RFC 3161, your own notary — is a function you supply.

## Quick start

```python
from smolagents import (
    CodeAgent, InferenceClientModel,
    ActionStep, FinalAnswerStep, ChainAnchorCallback,
)
from smolagents.memory import PlanningStep

def my_anchor_fn(root_hex: str) -> dict:
    # Plug in any backend conforming to chain-anchor/v1.
    # This stub just records the root locally; replace with
    # your real POST to Satsignal / RFC 3161 TSA / etc.
    return {
        "v": 1, "system": "my-service", "chain": "bsv-mainnet",
        "txid": "<from your anchor backend>",
        "root_hash": root_hex,
    }

cb = ChainAnchorCallback(
    anchor_fn=my_anchor_fn,
    out_dir="./receipts",
)

agent = CodeAgent(
    tools=[],
    model=InferenceClientModel(),
    step_callbacks={
        ActionStep: cb,
        FinalAnswerStep: cb,
        PlanningStep: cb,  # optional; only fires if the agent plans
    },
)
agent.run("What's 2 + 2?")
# → ./receipts/run-<uuid>.chain-anchor.json
```

The callback is registered against multiple step types so it sees both the body of the run (`ActionStep`) and the end (`FinalAnswerStep`, when the merkle root is computed and the anchor backend is called).

## What the sidecar looks like

```json
{
  "version": "chain-anchor-v1-sidecar",
  "run_id": "0e7d…",
  "started_at_epoch": 1778611200.0,
  "finished_at_epoch": 1778611207.4,
  "leaf_count": 4,
  "leaf_hashing": "leaf = sha256(JCS({label, sha256_hex}))",
  "tree": "satsignal merkle row, dup-last-when-odd",
  "merkle_root_hex": "686c620b9b0e7219bf943bf31f8ffa3be2b23fc21384aef4ae63a163ff3ebfb7",
  "items": [
    {"index": 0, "label": "action-step-1", "sha256_hex": "..."},
    {"index": 1, "label": "action-step-2", "sha256_hex": "..."},
    {"index": 2, "label": "action-step-3", "sha256_hex": "..."},
    {"index": 3, "label": "final-answer",  "sha256_hex": "..."}
  ],
  "chain_anchor": {
    "v": 1, "system": "my-service", "chain": "bsv-mainnet",
    "txid": "...", "root_hash": "686c620b..."
  },
  "anchor_error": null
}
```

A verifier with just this file plus one explorer lookup can:

1. Re-hash each step independently (smolagents step dicts canonicalize deterministically via the same JCS rule).
2. Walk the merkle root over the per-step leaves.
3. Confirm the resulting root equals `merkle_root_hex` *and* equals the `root_hash` committed on chain.

## Pluggable backends

The `anchor_fn` is a single function with signature `(root_hex: str) -> dict | None`. Anything that returns a dict conforming to [chain-anchor/v1](https://proof.satsignal.cloud/spec-chain-anchor) — at minimum `v`, `system`, `chain`, `txid`, `root_hash` — slots in. Two common shapes:

### Satsignal (BSV public-chain anchor)

```python
import requests

def satsignal_anchor(root_hex: str) -> dict:
    r = requests.post(
        "https://app.satsignal.cloud/api/v1/anchors",
        headers={"Authorization": f"Bearer {os.environ['SATSIGNAL_API_KEY']}"},
        json={
            "matter_slug": "agent-runs",
            "category": "commitment",
            "sha256_hex": root_hex,
        },
    )
    r.raise_for_status()
    a = r.json()
    return {
        "v": 1, "system": "satsignal", "chain": "bsv-mainnet",
        "txid": a["txid"], "root_hash": root_hex,
        "anchor_id": a["bundle_id"],
    }
```

### Local RFC 3161 / TSA path

```python
import base64, requests

def tsa_anchor(root_hex: str) -> dict:
    # Submit the root to an RFC 3161 TSA, return the token along with
    # a chain-anchor-shaped envelope that points at the TSA instead.
    # Example uses freetsa.org for demonstration.
    tsq = build_rfc3161_request(bytes.fromhex(root_hex))
    r = requests.post(
        "https://freetsa.org/tsr",
        headers={"Content-Type": "application/timestamp-query"},
        data=tsq,
    )
    return {
        "v": 1, "system": "rfc3161", "chain": "tsa:freetsa.org",
        "txid": "tsa:" + hashlib.sha256(r.content).hexdigest(),
        "root_hash": root_hex,
        "_tsa_token_b64": base64.b64encode(r.content).decode(),
    }
```

If your `anchor_fn` raises, the callback records the error in `anchor_error` instead of crashing the run. Local hashes still get the sidecar so you can backfill.

## Including planning steps

By default, planning steps are excluded from the leaves (they tend to be verbose and don't carry actions). To include them:

```python
cb = ChainAnchorCallback(anchor_fn=..., include_planning=True)
```

## Reproducible roots

The callback canonicalizes each step's `.dict()` representation via [RFC 8785 JSON Canonicalization Scheme](https://datatracker.ietf.org/doc/html/rfc8785) (sorted keys, no whitespace, UTF-8, no `NaN`/`Infinity`), then sha256s it. Two runs that produce *byte-identical* step contents produce the same root. Two runs with different timings or different LLM outputs do not — which is what you want; the receipt commits to the run that actually happened.
