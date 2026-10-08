#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Verifiable-receipt callback that anchors a batch of step hashes to any
third-party anchor backend conforming to ``chain-anchor-v1``.

Plug-and-play with :class:`MultiStepAgent` via the ``step_callbacks``
argument. smolagents takes no network dependency; the anchor backend
(BSV, RFC 3161 TSA, local timestamp service, anything) is supplied by
the caller as ``anchor_fn``.

Format reference: https://proof.satsignal.cloud/spec-chain-anchor
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from typing import Any, Callable

from smolagents.memory import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
)


__all__ = ["ChainAnchorCallback"]


def _jcs(obj: Any) -> bytes:
    """RFC 8785-ish JSON Canonicalization Scheme — sorted keys, no
    whitespace, UTF-8, no NaN/Infinity. Sufficient for the
    sha256-of-canonical-bytes pattern used by chain-anchor-v1."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _leaf_hash(label: str, sha256_hex: str) -> bytes:
    """Manifest-mode leaf hashing per chain-anchor-v1 §4.1: bind
    label and sha together so a swapped label can't preserve the leaf."""
    return _sha(_jcs({"label": label, "sha256_hex": sha256_hex}))


def _merkle_root(leaves: list[bytes]) -> bytes:
    """Satsignal merkle row, dup-last-when-odd."""
    if not leaves:
        return b"\x00" * 32
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_sha(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return level[0]


def _step_label(step: MemoryStep, index: int) -> str:
    if isinstance(step, ActionStep):
        return f"action-step-{step.step_number}"
    if isinstance(step, PlanningStep):
        return f"planning-step-{index}"
    if isinstance(step, FinalAnswerStep):
        return "final-answer"
    return f"step-{index}-{type(step).__name__.lower()}"


class ChainAnchorCallback:
    """Step callback that produces a ``chain-anchor-v1`` JSON sidecar
    per agent run.

    The callback collects a sha256 of each (canonicalized) step's
    serialized form as a leaf, computes a merkle root over the batch
    when the final answer fires, hands the root to a user-supplied
    ``anchor_fn``, and writes a sidecar JSON file conforming to
    ``chain-anchor-v1``.

    smolagents itself takes no network dependency; the anchor backend
    (BSV, RFC 3161 TSA, local timestamp service, anything else) lives
    entirely in the caller's ``anchor_fn``.

    Args:
        anchor_fn (`Callable[[str], dict | None]`): User-supplied function
            that takes the hex-encoded merkle root and returns a dict
            conforming to chain-anchor-v1 (``v``, ``system``, ``chain``,
            ``txid``, ``root_hash``, plus any optional fields). May
            return ``None`` if the caller does not have a backend yet —
            the sidecar still records the local merkle root + leaves
            so an anchor can be backfilled later.
        out_dir (`str`, *optional*, defaults to ``"."``): Directory the
            sidecar is written to. Filename is
            ``run-<uuid>.chain-anchor.json``.
        include_planning (`bool`, *optional*, defaults to ``False``):
            Whether to commit planning steps as leaves alongside action
            steps.
        run_id (`str`, *optional*): Override the generated run UUID.
            Useful in tests.
        clock (`Callable[[], float]`, *optional*, defaults to
            ``time.time``): Time source. Useful in tests.

    Register against both ``ActionStep`` and ``FinalAnswerStep`` so
    the callback sees the end of a run::

        from smolagents import (
            CodeAgent, ActionStep, FinalAnswerStep, ChainAnchorCallback,
        )
        from smolagents.memory import PlanningStep

        def my_anchor_fn(root_hex):
            # Plug in any anchor service.
            return {"v": 1, "system": "my-service", "chain": "bsv-mainnet",
                    "txid": "<from POST>", "root_hash": root_hex}

        cb = ChainAnchorCallback(anchor_fn=my_anchor_fn, out_dir="./receipts")
        agent = CodeAgent(
            tools=[...],
            model=...,
            step_callbacks={ActionStep: cb, FinalAnswerStep: cb,
                            PlanningStep: cb},
        )
        agent.run("Some task")
        # → ./receipts/run-<uuid>.chain-anchor.json
    """

    def __init__(
        self,
        anchor_fn: Callable[[str], dict | None] | None = None,
        out_dir: str = ".",
        include_planning: bool = False,
        run_id: str | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.anchor_fn = anchor_fn
        self.out_dir = out_dir
        self.include_planning = include_planning
        self.run_id = run_id or str(uuid.uuid4())
        self.clock = clock
        self._items: list[dict[str, str]] = []
        self._leaves: list[bytes] = []
        self._started_at = self.clock()
        self._flushed = False

    def __call__(self, memory_step: MemoryStep, agent=None) -> None:
        # Skip planning steps unless explicitly requested.
        if isinstance(memory_step, PlanningStep) and not self.include_planning:
            return
        # Record the step as a leaf.
        if isinstance(memory_step, (ActionStep, PlanningStep, FinalAnswerStep)):
            label = _step_label(memory_step, len(self._items))
            canonical = _jcs(memory_step.dict())
            step_sha = hashlib.sha256(canonical).hexdigest()
            self._items.append({"label": label, "sha256_hex": step_sha})
            self._leaves.append(_leaf_hash(label, step_sha))
        # Flush the anchor on the final-answer step.
        if isinstance(memory_step, FinalAnswerStep):
            self._flush()

    def _flush(self) -> None:
        if self._flushed:
            return
        self._flushed = True
        root_hex = _merkle_root(self._leaves).hex()
        chain_anchor: dict | None = None
        anchor_error: str | None = None
        if self.anchor_fn is not None:
            try:
                chain_anchor = self.anchor_fn(root_hex)
            except Exception as e:
                anchor_error = f"{type(e).__name__}: {e}"
        sidecar = {
            "version": "chain-anchor-v1-sidecar",
            "run_id": self.run_id,
            "started_at_epoch": self._started_at,
            "finished_at_epoch": self.clock(),
            "leaf_count": len(self._leaves),
            "leaf_hashing": "leaf = sha256(JCS({label, sha256_hex}))",
            "tree": "satsignal merkle row, dup-last-when-odd",
            "merkle_root_hex": root_hex,
            "items": [
                {"index": i, **it} for i, it in enumerate(self._items)
            ],
            "chain_anchor": chain_anchor,
            "anchor_error": anchor_error,
        }
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"run-{self.run_id}.chain-anchor.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, sort_keys=False)
        self.sidecar_path = path
        self.sidecar = sidecar
