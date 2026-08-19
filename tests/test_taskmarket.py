# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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
from unittest.mock import MagicMock, patch

import pytest

from smolagents.taskmarket import (
    BrowseTaskMarketTool,
    CreateTaskMarketTaskTool,
    GetTaskMarketTaskTool,
    ListTaskMarketSubmissionsTool,
    taskmarket_tools,
)
from smolagents.tool_validation import validate_tool_attributes
from smolagents.tools import Tool


SAMPLE_TASK = {
    "id": "0xabc",
    "description": "Write a CSV of animal-built structures.",
    "reward": "2000000",
    "netReward": "1850000",
    "mode": "bounty",
    "status": "open",
    "phase": "active",
    "expiryTime": "2026-08-22T11:58:25.795Z",
    "submissionCount": 3,
    "awardCount": 0,
    "requester": "0x1111",
    "tags": ["csv"],
    "pendingActions": [{"role": "worker", "action": "submit"}],
}


@pytest.mark.parametrize(
    "tool_class",
    [
        BrowseTaskMarketTool,
        GetTaskMarketTaskTool,
        ListTaskMarketSubmissionsTool,
        CreateTaskMarketTaskTool,
    ],
)
def test_taskmarket_tools_are_valid_tool_classes(tool_class):
    assert Tool in tool_class.__mro__
    assert validate_tool_attributes(tool_class) is None


def test_taskmarket_tools_helper_returns_four_tools():
    tools = taskmarket_tools(max_spend_usdc=5.0)
    assert [tool.name for tool in tools] == [
        "taskmarket_browse",
        "taskmarket_get_task",
        "taskmarket_list_submissions",
        "taskmarket_create_task",
    ]
    assert tools[-1].max_spend_usdc == 5.0


def test_browse_open_tasks():
    payload = {"tasks": [SAMPLE_TASK], "hasMore": False, "nextCursor": None}
    tool = BrowseTaskMarketTool()
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = payload
        mock_get.return_value.raise_for_status = lambda: None
        result = tool(limit=5, mode="bounty", min_reward_usdc=1.0)

    assert "0xabc" in result
    assert "2.0000 USDC" in result
    assert "Write a CSV" in result
    call_kwargs = mock_get.call_args
    assert call_kwargs.args[0] == "https://api.taskmarket.dev/api/tasks"
    assert call_kwargs.kwargs["params"]["status"] == "open"
    assert call_kwargs.kwargs["params"]["mode"] == "bounty"
    assert call_kwargs.kwargs["params"]["minReward"] == "1000000"
    assert call_kwargs.kwargs["params"]["limit"] == 5


def test_get_task_includes_tracking_fields():
    tool = GetTaskMarketTaskTool()
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = SAMPLE_TASK
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status = lambda: None
        result = tool("0xabc")

    assert "status: open" in result
    assert "2.0000 USDC" in result
    assert "net 1.8500 USDC" in result
    assert "pendingActions" in result
    assert "accept" not in result.lower() or "never auto-run accept/reject" in result
    mock_get.assert_called_once_with("https://api.taskmarket.dev/api/tasks/0xabc", timeout=20)


def test_list_submissions_is_review_only():
    tool = ListTaskMarketSubmissionsTool()
    rows = [
        {
            "workerAddress": "0xworker",
            "workerAgentId": "63929",
            "submittedAt": "2026-08-19T15:00:00.000Z",
            "rejectedAt": None,
            "deliverableHash": "0xhash",
            "fileName": "animal_architecture.csv",
        }
    ]
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = rows
        mock_get.return_value.raise_for_status = lambda: None
        result = tool("0xabc")

    assert "0xworker" in result
    assert "animal_architecture.csv" in result
    assert "Do not accept or reject from this tool." in result
    mock_get.assert_called_once_with(
        "https://api.taskmarket.dev/api/tasks/0xabc/submissions", timeout=20
    )


def test_create_refuses_without_confirm():
    tool = CreateTaskMarketTaskTool(max_spend_usdc=10.0)
    with patch("subprocess.run") as mock_run:
        result = tool(description="Ship a CSV", reward_usdc=2.0, confirm=False)
    assert "Refused" in result
    mock_run.assert_not_called()


def test_create_refuses_over_spend_cap():
    tool = CreateTaskMarketTaskTool(max_spend_usdc=1.0)
    with patch("subprocess.run") as mock_run:
        result = tool(description="Ship a CSV", reward_usdc=2.5, confirm=True)
    assert "exceeds this tool's max_spend_usdc" in result
    mock_run.assert_not_called()


def test_create_refuses_when_cli_missing():
    tool = CreateTaskMarketTaskTool()
    with patch("shutil.which", return_value=None), patch("subprocess.run") as mock_run:
        result = tool(description="Ship a CSV", reward_usdc=1.0, confirm=True)
    assert "CLI not found" in result
    assert "never accepts a private key" in result
    mock_run.assert_not_called()


def test_create_invokes_official_cli_when_confirmed():
    tool = CreateTaskMarketTaskTool(max_spend_usdc=10.0)
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = '{"ok":true,"data":{"id":"0xnew"}}'
    completed.stderr = ""
    with patch("shutil.which", return_value="/usr/bin/taskmarket"), patch(
        "subprocess.run", return_value=completed
    ) as mock_run:
        result = tool(
            description="Ship a CSV",
            reward_usdc=2.0,
            confirm=True,
            duration_hours=48,
            mode="bounty",
            tags="csv,data",
        )

    assert "CLI accepted" in result
    assert "0xnew" in result
    command = mock_run.call_args.args[0]
    assert command[:3] == ["/usr/bin/taskmarket", "task", "create"]
    assert "--reward" in command and "2.0" in command
    assert "--duration" in command and "48" in command
    assert mock_run.call_args.kwargs["check"] is False
