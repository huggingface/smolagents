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
import json
import subprocess
import unittest
from unittest import mock

from smolagents.taskmarket import TaskMarketTool


CREATE_OUT = (
    '{"ok":true,"data":{"taskId":"0x'
    + "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f901"
    + '"}}\n'
    "  Uploading proof.json...\n"
    "  proof.json: 100%\n"
)
STATUS_OUT = json.dumps({"ok": True, "data": {"taskId": "0xabc123", "phase": "active", "awardCount": 0}})
SUBS_OUT = json.dumps({"ok": True, "data": {"submissions": [{"worker": "0x1", "file": "a.md"}]}})


class TestTaskMarketTool(unittest.TestCase):
    def setUp(self):
        self.tool = TaskMarketTool(cli="taskmarket")

    # -------------------------------------------------------------- configure
    def test_configure_rejects_non_base_chain(self):
        with self.assertRaises(ValueError):
            self.tool.forward("configure", json.dumps({"max_spend_usdc": 10, "chain_id": 1}))

    def test_configure_sets_base_and_spend(self):
        out = json.loads(self.tool.forward("configure", json.dumps({"max_spend_usdc": 25})))
        self.assertEqual(out["chain_id"], 8453)
        self.assertEqual(out["max_spend_usdc"], 25)
        self.assertTrue(self.tool._configured)

    # ------------------------------------------------------------ create_task
    def test_create_task_requires_authorization(self):
        with self.assertRaises(PermissionError):
            self.tool.forward(
                "create_task",
                json.dumps(
                    {
                        "description": "x",
                        "reward_usdc": 1,
                        "deadline_unix": 1,
                        "deliverables": "y",
                        "authorized_by_user": False,
                    }
                ),
            )

    def test_create_task_rejects_negative_reward(self):
        with self.assertRaises(ValueError):
            self.tool.forward(
                "create_task",
                json.dumps(
                    {
                        "description": "x",
                        "reward_usdc": -1,
                        "deadline_unix": 1,
                        "deliverables": "y",
                        "authorized_by_user": True,
                    }
                ),
            )

    def test_create_task_enforces_spend_cap(self):
        self.tool.forward("configure", json.dumps({"max_spend_usdc": 2}))
        with self.assertRaises(ValueError):
            self.tool.forward(
                "create_task",
                json.dumps(
                    {
                        "description": "x",
                        "reward_usdc": 5,
                        "deadline_unix": 1,
                        "deliverables": "y",
                        "authorized_by_user": True,
                    }
                ),
            )

    def test_create_task_invokes_cli_with_micro_usdc_and_parses_id(self):
        with mock.patch.object(self.tool, "_resolve_cmd", return_value=["taskmarket"]), mock.patch(
            "smolagents.taskmarket.subprocess.run"
        ) as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=CREATE_OUT, stderr=""
            )
            out = json.loads(
                self.tool.forward(
                    "create_task",
                    json.dumps(
                        {
                            "description": "Write a report",
                            "reward_usdc": 3,
                            "deadline_unix": 1700000000,
                            "deliverables": "one markdown file",
                            "authorized_by_user": True,
                        }
                    ),
                )
            )
            self.assertTrue(out["task_id"].startswith("0x"))
            self.assertEqual(len(out["task_id"]), 66)  # 0x + 64 hex
            called = run.call_args[0][0]
            self.assertEqual(called[0:2], ["taskmarket", "task"])
            # reward 3 USDC -> 3_000_000 micro-USDC
            self.assertIn("3000000", called)
            self.assertIn("Write a report", called)

    def test_create_task_rejects_non_base_chain(self):
        with self.assertRaises(ValueError):
            self.tool.forward(
                "create_task",
                json.dumps(
                    {
                        "description": "x",
                        "reward_usdc": 1,
                        "deadline_unix": 1,
                        "deliverables": "y",
                        "authorized_by_user": True,
                        "chain_id": 137,
                    }
                ),
            )

    # --------------------------------------------------------------- get_*
    def test_get_status_parses_json(self):
        with mock.patch.object(self.tool, "_resolve_cmd", return_value=["taskmarket"]), mock.patch(
            "smolagents.taskmarket.subprocess.run"
        ) as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=STATUS_OUT, stderr=""
            )
            out = json.loads(self.tool.forward("get_status", json.dumps({"task_id": "0xabc123"})))
            self.assertEqual(out["phase"], "active")

    def test_get_submissions_returns_for_human_review(self):
        with mock.patch.object(self.tool, "_resolve_cmd", return_value=["taskmarket"]), mock.patch(
            "smolagents.taskmarket.subprocess.run"
        ) as run:
            run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=SUBS_OUT, stderr=""
            )
            out = json.loads(
                self.tool.forward("get_submissions", json.dumps({"task_id": "0xabc123"}))
            )
            self.assertEqual(len(out), 1)
            self.assertEqual(out[0]["file"], "a.md")

    # ------------------------------------------------------------- guardrails
    def test_unknown_action(self):
        self.assertIn("Unknown action", self.tool.forward("frobnicate", "{}"))

    def test_invalid_params_json(self):
        self.assertIn("Invalid params_json", self.tool.forward("configure", "not-json"))

    def test_missing_cli_raises_helpful_error(self):
        tool = TaskMarketTool(cli="definitely-not-a-real-cli-xyz")
        with mock.patch("smolagents.taskmarket.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                tool.forward("get_status", json.dumps({"task_id": "0x1"}))

    def test_tool_contract_valid(self):
        # The base class validates name/inputs/output_type/forward signature on init.
        self.assertEqual(self.tool.name, "taskmarket")
        self.assertEqual(self.tool.output_type, "string")
        self.assertIn("action", self.tool.inputs)


if __name__ == "__main__":
    unittest.main()
