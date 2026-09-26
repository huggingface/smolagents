import hashlib
import json

from smolagents.chain_anchor import (
    ChainAnchorCallback,
    _jcs,
    _leaf_hash,
    _merkle_root,
)
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import Timing


def _make_action_step(n: int, output: str) -> ActionStep:
    return ActionStep(
        step_number=n,
        timing=Timing(start_time=0.0, end_time=1.0),
        action_output=output,
    )


class TestHelpers:
    def test_jcs_sorts_keys_and_strips_whitespace(self):
        out = _jcs({"b": 2, "a": 1})
        assert out == b'{"a":1,"b":2}'

    def test_jcs_utf8(self):
        out = _jcs({"k": "café"})
        assert b"caf\xc3\xa9" in out

    def test_leaf_hash_binds_label(self):
        # Same sha256_hex but different labels → different leaf hashes
        l1 = _leaf_hash("step-1", "deadbeef" * 8)
        l2 = _leaf_hash("step-2", "deadbeef" * 8)
        assert l1 != l2

    def test_merkle_root_single_leaf(self):
        leaf = b"\xaa" * 32
        assert _merkle_root([leaf]) == leaf

    def test_merkle_root_two_leaves(self):
        a, b = b"\x01" * 32, b"\x02" * 32
        expected = hashlib.sha256(a + b).digest()
        assert _merkle_root([a, b]) == expected

    def test_merkle_root_three_leaves_dups_last(self):
        a, b, c = b"\x01" * 32, b"\x02" * 32, b"\x03" * 32
        # Level 1: pair(a,b), pair(c,c)
        l1 = hashlib.sha256(a + b).digest()
        l2 = hashlib.sha256(c + c).digest()
        expected = hashlib.sha256(l1 + l2).digest()
        assert _merkle_root([a, b, c]) == expected

    def test_merkle_root_empty(self):
        assert _merkle_root([]) == b"\x00" * 32


class TestChainAnchorCallback:
    def test_callback_writes_sidecar_with_local_root(self, tmp_path):
        cb = ChainAnchorCallback(out_dir=str(tmp_path), run_id="testrun")
        cb(_make_action_step(1, "alpha"))
        cb(_make_action_step(2, "beta"))
        cb(FinalAnswerStep(output="done"))
        path = tmp_path / "run-testrun.chain-anchor.json"
        assert path.exists()
        sidecar = json.loads(path.read_text())
        assert sidecar["version"] == "chain-anchor-v1-sidecar"
        assert sidecar["run_id"] == "testrun"
        assert sidecar["leaf_count"] == 3
        assert sidecar["chain_anchor"] is None
        assert sidecar["anchor_error"] is None
        # Merkle root is deterministic from the 3 leaves
        assert len(sidecar["merkle_root_hex"]) == 64
        labels = [it["label"] for it in sidecar["items"]]
        assert labels == ["action-step-1", "action-step-2", "final-answer"]

    def test_callback_calls_anchor_fn_with_root(self, tmp_path):
        captured: dict[str, str] = {}

        def fake_anchor_fn(root_hex: str) -> dict:
            captured["root"] = root_hex
            return {
                "v": 1, "system": "test", "chain": "bsv-mainnet",
                "txid": "deadbeef" * 8, "root_hash": root_hex,
            }

        cb = ChainAnchorCallback(
            anchor_fn=fake_anchor_fn, out_dir=str(tmp_path), run_id="r2",
        )
        cb(_make_action_step(1, "x"))
        cb(FinalAnswerStep(output="done"))
        sidecar = json.loads((tmp_path / "run-r2.chain-anchor.json").read_text())
        assert sidecar["merkle_root_hex"] == captured["root"]
        assert sidecar["chain_anchor"]["txid"] == "deadbeef" * 8
        assert sidecar["chain_anchor"]["root_hash"] == captured["root"]

    def test_anchor_fn_exception_recorded_not_raised(self, tmp_path):
        def boom(_root: str) -> dict:
            raise RuntimeError("network unreachable")

        cb = ChainAnchorCallback(
            anchor_fn=boom, out_dir=str(tmp_path), run_id="r3",
        )
        cb(_make_action_step(1, "x"))
        cb(FinalAnswerStep(output="done"))
        sidecar = json.loads((tmp_path / "run-r3.chain-anchor.json").read_text())
        assert sidecar["chain_anchor"] is None
        assert "RuntimeError" in sidecar["anchor_error"]
        assert "network unreachable" in sidecar["anchor_error"]

    def test_planning_steps_excluded_by_default(self, tmp_path):
        cb = ChainAnchorCallback(out_dir=str(tmp_path), run_id="r4")
        cb(_make_action_step(1, "x"))
        cb(PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan body"),
            plan="some plan",
            timing=Timing(start_time=0.0, end_time=1.0),
        ))
        cb(FinalAnswerStep(output="done"))
        sidecar = json.loads((tmp_path / "run-r4.chain-anchor.json").read_text())
        assert sidecar["leaf_count"] == 2  # action + final, no planning
        labels = [it["label"] for it in sidecar["items"]]
        assert "planning-step-1" not in labels

    def test_planning_steps_included_when_opted_in(self, tmp_path):
        cb = ChainAnchorCallback(
            out_dir=str(tmp_path), run_id="r5", include_planning=True,
        )
        cb(_make_action_step(1, "x"))
        cb(PlanningStep(
            model_input_messages=[],
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan body"),
            plan="some plan",
            timing=Timing(start_time=0.0, end_time=1.0),
        ))
        cb(FinalAnswerStep(output="done"))
        sidecar = json.loads((tmp_path / "run-r5.chain-anchor.json").read_text())
        assert sidecar["leaf_count"] == 3
        labels = [it["label"] for it in sidecar["items"]]
        assert "planning-step-1" in labels

    def test_flush_is_idempotent(self, tmp_path):
        cb = ChainAnchorCallback(out_dir=str(tmp_path), run_id="r6")
        cb(_make_action_step(1, "x"))
        cb(FinalAnswerStep(output="done"))
        cb(FinalAnswerStep(output="done"))  # second flush should be a no-op
        sidecar = json.loads((tmp_path / "run-r6.chain-anchor.json").read_text())
        # leaf_count would have been 3 if the second flush had recorded
        # another final-answer leaf
        assert sidecar["leaf_count"] == 2

    def test_merkle_root_matches_independent_recomputation(self, tmp_path):
        """Sanity check: an independent stdlib re-implementation produces
        the same root the callback writes. This is the property a
        chain-anchor-v1 verifier checks."""
        cb = ChainAnchorCallback(out_dir=str(tmp_path), run_id="r7")
        cb(_make_action_step(1, "alpha"))
        cb(_make_action_step(2, "beta"))
        cb(FinalAnswerStep(output="gamma"))
        sidecar = json.loads((tmp_path / "run-r7.chain-anchor.json").read_text())

        # Re-derive from sidecar items.
        def jcs(o):
            return json.dumps(o, sort_keys=True, separators=(",", ":"),
                              ensure_ascii=False, allow_nan=False).encode("utf-8")

        def sha(b):
            return hashlib.sha256(b).digest()

        leaves = [
            sha(jcs({"label": it["label"], "sha256_hex": it["sha256_hex"]}))
            for it in sidecar["items"]
        ]
        # dup-last-when-odd merkle
        level = list(leaves)
        while len(level) > 1:
            if len(level) % 2 == 1:
                level.append(level[-1])
            level = [sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
        assert level[0].hex() == sidecar["merkle_root_hex"]
