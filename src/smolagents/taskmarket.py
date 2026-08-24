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
"""
TaskMarket integration for smolagents.

`TaskMarketTool` lets a *human-operated* smolagent act as a TaskMarket **requester**:
configure a Base-only session, create a USDC-escrowed task (only after fresh, explicit
user authorization), and retrieve live task status plus submissions for **human review**.
It never silently accepts or rejects work, never handles private keys, and always shells
out to the official `taskmarket` CLI (first-party tooling).

Setup
-----
    npm install -g @lucid-agents/taskmarket
    taskmarket init

Usage (inside an agent)
-----------------------
    tool = TaskMarketTool()
    # 1) configure (Base only, set a spend cap)
    tool.forward("configure", '{"max_spend_usdc": 50}')
    # 2) create a task — requires explicit fresh authorization
    tool.forward("create_task", json.dumps({
        "description": "Summarize the Q3 roadmap.",
        "reward_usdc": 5,
        "deadline_unix": int(time.time()) + 86400,
        "deliverables": "one markdown file",
        "authorized_by_user": True,
    }))
    # 3) inspect status / submissions for human review
    tool.forward("get_status", '{"task_id": "0x..."}')
    tool.forward("get_submissions", '{"task_id": "0x..."}')

Reproduction
------------
    pytest tests/test_taskmarket.py
"""
from __future__ import annotations

import json
import shutil
import subprocess

from .tools import Tool


class TaskMarketTool(Tool):
    """
    On-chain (Base) TaskMarket requester integration.

    Exposes four guarded actions through a single `forward(action, params_json)` entry point:
      - `configure`       : lock the network to Base (8453) and set a maximum spend in USDC.
      - `create_task`     : create + escrow a task. Requires `authorized_by_user=true`
                            (fresh, explicit human authorization) and respects the spend cap.
      - `get_status`      : return the live task status (parsed JSON).
      - `get_submissions` : return submissions for human review (never auto-accept/reject).

    Private keys are never read, stored, logged, or committed; all on-chain actions are
    delegated to the first-party `taskmarket` CLI.
    """

    name = "taskmarket"
    description = (
        "Integrate with TaskMarket, an on-chain (Base) agent task marketplace that escrows "
        "USDC before work begins. Use this to configure a Base-only requester session, create "
        "a funded task (only with explicit fresh user authorization), and retrieve live task "
        "status and submissions for human review. This tool never auto-accepts or rejects "
        "submissions and never handles private keys. Requires the `taskmarket` CLI "
        "(`npm install -g @lucid-agents/taskmarket`) on PATH."
    )
    inputs = {
        "action": {
            "type": "string",
            "description": (
                "One of: 'configure' (Base network + max spend), "
                "'create_task' (escrow a task; requires authorized_by_user=true), "
                "'get_status' (live task status), 'get_submissions' (submissions for human review)."
            ),
        },
        "params_json": {
            "type": "string",
            "description": (
                "JSON object of action parameters. "
                "configure: {\"max_spend_usdc\": float}. "
                "create_task: {\"description\": str, \"reward_usdc\": float, \"deadline_unix\": int, "
                "\"deliverables\": str, \"authorized_by_user\": bool, \"chain_id\": int?}. "
                "get_status/get_submissions: {\"task_id\": str}."
            ),
        },
    }
    output_type = "string"

    BASE_CHAIN_ID = 8453  # Base mainnet

    def __init__(self, cli: str = "taskmarket"):
        super().__init__()
        self.cli = cli
        self._max_spend_usdc: float | None = None
        self._configured = False

    # ------------------------------------------------------------------ helpers
    def _resolve_cmd(self) -> list[str]:
        """Prefer the `taskmarket` binary; fall back to `npx taskmarket`."""
        if shutil.which(self.cli):
            return [self.cli]
        if shutil.which("npx"):
            return ["npx", "taskmarket"]
        raise RuntimeError(
            "TaskMarket CLI not found on PATH. Install it with "
            "`npm install -g @lucid-agents/taskmarket`."
        )

    def _run(self, args: list[str]) -> str:
        cmd = self._resolve_cmd() + args
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError("TaskMarket CLI timed out.") from None
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "TaskMarket CLI failed.")
        return proc.stdout

    @staticmethod
    def _parse_task_id(out: str) -> str | None:
        for line in out.splitlines():
            if "0x" in line:
                tail = line.split("0x")[-1]
                if len(tail) >= 60:
                    return "0x" + tail[:64]
        return None

    # ------------------------------------------------------------------ forward
    def forward(self, action: str, params_json: str) -> str:
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError as e:
            return f"Invalid params_json: {e}"

        action = str(action).strip().lower()

        if action == "configure":
            chain_id = int(params.get("chain_id", self.BASE_CHAIN_ID))
            if chain_id != self.BASE_CHAIN_ID:
                raise ValueError(f"Only Base (8453) is allowed; got {chain_id}.")
            self._max_spend_usdc = float(params["max_spend_usdc"])
            self._configured = True
            return json.dumps(
                {"network": "Base", "chain_id": chain_id, "max_spend_usdc": self._max_spend_usdc}
            )

        if action == "create_task":
            if not params.get("authorized_by_user"):
                raise PermissionError(
                    "FRESH user authorization required before funding a task "
                    "(set authorized_by_user=true with an explicit human confirmation)."
                )
            chain_id = int(params.get("chain_id", self.BASE_CHAIN_ID))
            if chain_id != self.BASE_CHAIN_ID:
                raise ValueError(f"Only Base (8453) is allowed; got {chain_id}.")
            reward = float(params["reward_usdc"])
            if reward <= 0:
                raise ValueError("Reward must be greater than 0.")
            if self._configured and self._max_spend_usdc is not None and reward > self._max_spend_usdc:
                raise ValueError(
                    f"Reward {reward} USDC exceeds configured max spend {self._max_spend_usdc} USDC."
                )
            out = self._run(
                [
                    "task",
                    "create",
                    "--description",
                    str(params["description"]),
                    "--reward",
                    str(int(reward * 1_000_000)),  # micro-USDC
                    "--deadline",
                    str(int(params["deadline_unix"])),
                    "--deliverables",
                    str(params["deliverables"]),
                ]
            )
            task_id = self._parse_task_id(out)
            return json.dumps({"task_id": task_id, "raw": out})

        if action in ("get_status", "get_submissions"):
            out = self._run(["task", "get", str(params["task_id"])])
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                return out
            payload = data.get("data", data)
            if action == "get_status":
                return json.dumps(payload)
            # submissions for human review — never auto-decide
            return json.dumps(payload.get("submissions", payload.get("awards", [])))

        return (
            f"Unknown action '{action}'. "
            "Use: configure | create_task | get_status | get_submissions."
        )
