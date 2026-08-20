# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import pytest

from smolagents.taskmarket_tools import (
    BASE_USDC_CONTRACT,
    TaskmarketCLIError,
    TaskmarketRequesterSession,
    TaskmarketToolCollection,
)


def _completed(command, data, returncode=0):
    if returncode == 0:
        envelope = {"ok": True, "data": data}
    else:
        envelope = {"ok": False, "error": str(data)}
    return subprocess.CompletedProcess(command, returncode, stdout=json.dumps(envelope), stderr="")


@pytest.fixture
def cli_runner(monkeypatch):
    calls = []

    def fake_run(command, *, capture_output, text, timeout, check):
        calls.append(command)
        args = command[1:]
        if args == ["deposit"]:
            return _completed(
                command,
                {
                    "address": "0xRequester",
                    "network": "Base Mainnet",
                    "chainId": 8453,
                    "currency": "USDC",
                    "usdcContract": BASE_USDC_CONTRACT,
                },
            )
        if args == ["stats"]:
            return _completed(command, {"address": "0xRequester", "balanceUsdc": "25.000000"})
        if args[:2] == ["task", "create"]:
            return _completed(command, {"taskId": "0xTask"})
        return _completed(command, "unexpected command", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


@pytest.fixture
def api_reads(monkeypatch):
    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, *, params=None, timeout=None):
        if url.endswith("/api/tasks/0xTask/submissions"):
            return FakeResponse([{"id": "s1", "workerAddress": "0xWorker"}])
        if url.endswith("/api/tasks/0xTask"):
            return FakeResponse({"id": "0xTask", "status": "open", "reward": "2000000"})
        if url.endswith("/api/tasks"):
            return FakeResponse([{"id": "0xTask", "status": params["status"]}])
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("requests.get", fake_get)


def _preview(session):
    return session.preview_task(
        description="Produce a deterministic report.",
        deliverables=["report.md", "tests.txt"],
        reward_usdc=2,
        duration_hours=6,
    )


def test_preview_shows_exact_spend_network_and_deliverables(cli_runner):
    session = TaskmarketRequesterSession(max_spend_usdc=5)
    preview = _preview(session)

    assert preview["reward_usdc"] == "2"
    assert preview["maximum_spend_usdc"] == "2"
    assert preview["configured_spend_cap_usdc"] == "5"
    assert preview["chain_id"] == 8453
    assert preview["network"] == "Base Mainnet"
    assert preview["human_approval_required"] is True
    assert preview["deliverables"] == ["report.md", "tests.txt"]
    assert preview["canonical_task_description"].endswith("Deliverables:\n- report.md\n- tests.txt")
    assert cli_runner[:2] == [["taskmarket", "deposit"], ["taskmarket", "stats"]]


def test_max_spend_is_enforced_before_cli_calls(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("CLI must not run when local spend validation fails.")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    session = TaskmarketRequesterSession(max_spend_usdc=1)

    with pytest.raises(ValueError, match="exceeds configured maximum spend"):
        session.preview_task(
            description="Too expensive",
            deliverables=["out.txt"],
            reward_usdc=2,
            duration_hours=1,
        )


def test_create_requires_host_authorization_and_consumes_it(cli_runner, api_reads):
    collection = TaskmarketToolCollection(max_spend_usdc=5)
    preview = _preview(collection.session)

    with pytest.raises(PermissionError, match="No fresh human authorization"):
        collection.session.create_approved_task(preview["preview_id"])

    collection.authorize(preview["preview_id"])
    result = collection.session.create_approved_task(preview["preview_id"])

    assert result["ok"] is True
    assert result["task_id"] == "0xTask"
    assert result["task"]["status"] == "open"
    assert result["api_url"].endswith("/api/tasks/0xTask")

    create_calls = [call for call in cli_runner if call[1:3] == ["task", "create"]]
    assert len(create_calls) == 1
    command = create_calls[0]
    assert command[command.index("--reward") + 1] == "2"
    assert command[command.index("--duration") + 1] == "6"
    assert command[command.index("--description") + 1].endswith("Deliverables:\n- report.md\n- tests.txt")

    with pytest.raises(PermissionError, match="No fresh human authorization"):
        collection.session.create_approved_task(preview["preview_id"])
    assert len([call for call in cli_runner if call[1:3] == ["task", "create"]]) == 1


def test_wrong_network_refuses_to_spend(monkeypatch):
    def fake_run(command, *, capture_output, text, timeout, check):
        return _completed(
            command,
            {
                "address": "0xRequester",
                "network": "Ethereum Mainnet",
                "chainId": 1,
                "currency": "USDC",
                "usdcContract": "0xWrong",
            },
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    session = TaskmarketRequesterSession(max_spend_usdc=5)

    with pytest.raises(TaskmarketCLIError, match="Refusing to spend"):
        session.preview_task(
            description="Network guard",
            deliverables=["out.txt"],
            reward_usdc=1,
            duration_hours=1,
        )


def test_payment_timeout_is_ambiguous_and_not_retried(monkeypatch, api_reads):
    create_attempts = 0

    def fake_run(command, *, capture_output, text, timeout, check):
        nonlocal create_attempts
        args = command[1:]
        if args == ["deposit"]:
            return _completed(
                command,
                {
                    "address": "0xRequester",
                    "network": "Base Mainnet",
                    "chainId": 8453,
                    "currency": "USDC",
                    "usdcContract": BASE_USDC_CONTRACT,
                },
            )
        if args == ["stats"]:
            return _completed(command, {"address": "0xRequester", "balanceUsdc": "25.000000"})
        if args[:2] == ["task", "create"]:
            create_attempts += 1
            raise subprocess.TimeoutExpired(command, timeout)
        return _completed(command, "unexpected", returncode=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    collection = TaskmarketToolCollection(max_spend_usdc=5)
    preview = _preview(collection.session)
    collection.authorize(preview["preview_id"])

    result = collection.session.create_approved_task(preview["preview_id"])

    assert result["ok"] is False
    assert result["settlement_status"] == "unknown"
    assert result["do_not_retry"] is True
    assert create_attempts == 1

    with pytest.raises(PermissionError, match="No fresh human authorization"):
        collection.session.create_approved_task(preview["preview_id"])
    assert create_attempts == 1

    with pytest.raises(ValueError, match="ambiguous settlement"):
        collection.authorize(preview["preview_id"])


def test_submissions_are_read_only_and_collection_has_no_accept_tool(api_reads):
    collection = TaskmarketToolCollection(max_spend_usdc=5)
    submissions_tool = next(tool for tool in collection.tools if tool.name == "taskmarket_list_submissions")
    result = submissions_tool(task_id="0xTask")

    assert result["submissions"][0]["id"] == "s1"
    assert result["human_review_required"] is True
    assert result["automatic_acceptance"] is False
    assert result["automatic_rejection"] is False

    names = {tool.name for tool in collection.tools}
    assert names == {
        "taskmarket_list_tasks",
        "taskmarket_preview_task",
        "taskmarket_create_task",
        "taskmarket_task_status",
        "taskmarket_list_submissions",
    }
    assert all("accept" not in name and "reject" not in name for name in names)
