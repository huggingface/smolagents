#!/usr/bin/env python
# coding=utf-8

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
"""Taskmarket tools for delegating work to funded external workers.

Taskmarket (https://taskmarket.dev) is an onchain task marketplace on Base.
These tools let a smolagents agent browse funded work, inspect a brief,
present submissions for human review, and create a task only after explicit
confirmation and within a hard USDC spend cap.

They never hold private keys, never auto-accept work, and never spend funds
unless `confirm=True` and the reward is within `max_spend_usdc`.
"""

from __future__ import annotations

from .tools import Tool

DEFAULT_TASKMARKET_API_URL = "https://api.taskmarket.dev"


class BrowseTaskMarketTool(Tool):
    """List open Taskmarket tasks so an agent can decide whether to delegate.

    Args:
        api_url (`str`, default `"https://api.taskmarket.dev"`): Taskmarket REST origin.
        timeout (`int`, default `20`): HTTP timeout in seconds.

    Examples:
        ```python
        >>> from smolagents import BrowseTaskMarketTool
        >>> print(BrowseTaskMarketTool()(limit=3, mode="bounty"))
        ```
    """

    name = "taskmarket_browse"
    description = (
        "Browse open Taskmarket tasks (USDC on Base). Use this when a request is "
        "better delegated to external workers than solved unreliably in-process. "
        "Read-only: does not spend funds or create tasks."
    )
    inputs = {
        "limit": {
            "type": "integer",
            "description": "Maximum number of open tasks to return (1-25).",
            "nullable": True,
        },
        "mode": {
            "type": "string",
            "description": "Optional mode filter: bounty, claim, pitch, benchmark, or auction.",
            "nullable": True,
        },
        "min_reward_usdc": {
            "type": "number",
            "description": "Optional minimum reward in USDC.",
            "nullable": True,
        },
        "max_reward_usdc": {
            "type": "number",
            "description": "Optional maximum reward in USDC.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, api_url="https://api.taskmarket.dev", timeout=20):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def forward(
        self,
        limit: int | None = None,
        mode: str | None = None,
        min_reward_usdc: float | None = None,
        max_reward_usdc: float | None = None,
    ) -> str:
        import requests
        from requests.exceptions import RequestException

        page_size = 10 if limit is None else int(limit)
        page_size = max(1, min(page_size, 25))
        params = {"status": "open", "limit": page_size, "sort": "newest"}
        if mode:
            params["mode"] = mode
        if min_reward_usdc is not None:
            params["minReward"] = str(int(float(min_reward_usdc) * 1000000))
        if max_reward_usdc is not None:
            params["maxReward"] = str(int(float(max_reward_usdc) * 1000000))
        try:
            response = requests.get(f"{self.api_url}/api/tasks", params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except RequestException as error:
            return f"Error browsing Taskmarket: {error}"

        tasks = payload.get("tasks") or []
        lines = [
            f"Open Taskmarket tasks ({len(tasks)} shown"
            + (", more available" if payload.get("hasMore") else "")
            + "). Inspect a task with taskmarket_get_task before acting."
        ]
        for task in tasks:
            reward = 0.0
            try:
                reward = int(task.get("reward") or 0) / 1000000
            except (TypeError, ValueError):
                reward = 0.0
            description = str(task.get("description") or "").replace("\n", " ").strip()
            if len(description) > 500:
                description = description[:500] + "..."
            task_id = task.get("id")
            lines.append(
                f"- {task_id} | {reward:.4f} USDC | mode={task.get('mode')} | "
                f"submissions={task.get('submissionCount')} | expiry={task.get('expiryTime')}\n"
                f"  {description}\n  https://taskmarket.dev/tasks/{task_id}"
            )
        if len(lines) == 1:
            lines.append("No open tasks matched these filters.")
        return "\n".join(lines)


class GetTaskMarketTaskTool(Tool):
    """Fetch one Taskmarket task, including status suitable for tracking.

    Args:
        api_url (`str`, default `"https://api.taskmarket.dev"`): Taskmarket REST origin.
        timeout (`int`, default `20`): HTTP timeout in seconds.
    """

    name = "taskmarket_get_task"
    description = (
        "Get a Taskmarket task by id: brief, reward, status, expiry, submission "
        "count, and next actions. Read-only. Does not accept, reject, or pay."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "Task id, typically a 0x-prefixed 64-hex hash.",
        }
    }
    output_type = "string"

    def __init__(self, api_url="https://api.taskmarket.dev", timeout=20):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def forward(self, task_id: str) -> str:
        import requests
        from requests.exceptions import RequestException

        task_id = (task_id or "").strip()
        if not task_id:
            return "Error: task_id is required."
        try:
            response = requests.get(f"{self.api_url}/api/tasks/{task_id}", timeout=self.timeout)
            if response.status_code == 404:
                return f"No Taskmarket task found for {task_id}."
            response.raise_for_status()
            task = response.json()
        except RequestException as error:
            return f"Error fetching Taskmarket task: {error}"

        if not isinstance(task, dict) or not task.get("id"):
            data = task.get("data") if isinstance(task, dict) else None
            task = data if isinstance(data, dict) else task
        if not isinstance(task, dict):
            return f"Unexpected Taskmarket response for {task_id}."

        reward = 0.0
        try:
            reward = int(task.get("reward") or 0) / 1000000
        except (TypeError, ValueError):
            reward = 0.0
        net = task.get("netReward")
        net_usdc = None
        try:
            net_usdc = int(net) / 1000000 if net is not None else None
        except (TypeError, ValueError):
            net_usdc = None
        description = str(task.get("description") or "")
        if len(description) > 4000:
            description = description[:4000] + "\n...[truncated to 4000 characters]..."
        pending = task.get("pendingActions") or []
        pending_parts = []
        for action in pending:
            if isinstance(action, dict):
                pending_parts.append(f"{action.get('role')}:{action.get('action')}")
        pending_text = ", ".join(pending_parts) or "none"
        return (
            f"id: {task.get('id')}\n"
            f"url: https://taskmarket.dev/tasks/{task.get('id')}\n"
            f"status: {task.get('status')} | phase: {task.get('phase')} | mode: {task.get('mode')}\n"
            f"reward: {reward:.4f} USDC"
            + (f" (net {net_usdc:.4f} USDC)" if net_usdc is not None else "")
            + "\n"
            f"expiry: {task.get('expiryTime')}\n"
            f"submissions: {task.get('submissionCount')} | awards: {task.get('awardCount')}\n"
            f"requester: {task.get('requester')}\n"
            f"pendingActions (never auto-run accept/reject): {pending_text}\n"
            f"tags: {task.get('tags')}\n\n"
            f"{description}"
        )


class ListTaskMarketSubmissionsTool(Tool):
    """Present Taskmarket submissions for human review. Never auto-accepts.

    Args:
        api_url (`str`, default `"https://api.taskmarket.dev"`): Taskmarket REST origin.
        timeout (`int`, default `20`): HTTP timeout in seconds.
    """

    name = "taskmarket_list_submissions"
    description = (
        "List submissions on a Taskmarket task for human review. Returns worker, "
        "time, hashes, rejection status, and artifact names. Does not accept, "
        "reject, rate, or pay anyone."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "Task id whose submissions should be listed.",
        }
    }
    output_type = "string"

    def __init__(self, api_url="https://api.taskmarket.dev", timeout=20):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def forward(self, task_id: str) -> str:
        import requests
        from requests.exceptions import RequestException

        task_id = (task_id or "").strip()
        if not task_id:
            return "Error: task_id is required."
        try:
            response = requests.get(
                f"{self.api_url}/api/tasks/{task_id}/submissions", timeout=self.timeout
            )
            response.raise_for_status()
            payload = response.json()
        except RequestException as error:
            return f"Error listing Taskmarket submissions: {error}"

        rows = payload if isinstance(payload, list) else payload.get("data") or payload.get("submissions") or []
        if not isinstance(rows, list):
            return f"Unexpected submissions payload for {task_id}."
        lines = [
            f"Submissions for {task_id} ({len(rows)}). "
            "Present these to a human reviewer. Do not accept or reject from this tool."
        ]
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            artifacts = row.get("artifacts") or row.get("files") or []
            names = []
            if isinstance(artifacts, list):
                for artifact in artifacts:
                    if isinstance(artifact, dict):
                        names.append(str(artifact.get("fileName") or artifact.get("name") or "artifact"))
                    else:
                        names.append(str(artifact))
            rejected = row.get("rejectedAt")
            lines.append(
                f"{index}. worker={row.get('workerAddress') or row.get('worker')} "
                f"agent={row.get('workerAgentId')} submittedAt={row.get('submittedAt')} "
                f"rejectedAt={rejected} deliverableHash={row.get('deliverableHash')} "
                f"files={names or row.get('fileName')}"
            )
        if len(lines) == 1:
            lines.append("No submissions yet.")
        return "\n".join(lines)


class CreateTaskMarketTaskTool(Tool):
    """Create a funded Taskmarket task via the first-party CLI, never from raw keys.

    Spending is refused unless `confirm=True` and `reward_usdc` is within
    `max_spend_usdc`. The official `taskmarket` CLI owns the wallet.

    Args:
        api_url (`str`, default `"https://api.taskmarket.dev"`): Unused for writes; kept for a consistent constructor.
        timeout (`int`, default `20`): CLI timeout in seconds.
        max_spend_usdc (`float`, default `10.0`): Hard cap on reward USDC this tool will escrow.
        cli_path (`str`, default `"taskmarket"`): First-party CLI executable.
    """

    name = "taskmarket_create_task"
    description = (
        "Create and fund a Taskmarket task using the official taskmarket CLI. "
        "Requires confirm=True. Refuses rewards above this tool's max_spend_usdc. "
        "Does not hold private keys, does not auto-accept submissions, and will "
        "not spend without explicit confirmation."
    )
    inputs = {
        "description": {
            "type": "string",
            "description": "Public task brief. Be specific about deliverable and acceptance.",
        },
        "reward_usdc": {
            "type": "number",
            "description": "USDC to escrow. Must be > 0 and <= the tool's max_spend_usdc.",
        },
        "confirm": {
            "type": "boolean",
            "description": "Must be true. The tool refuses to spend unless the user explicitly confirms.",
        },
        "duration_hours": {
            "type": "integer",
            "description": "How long the task stays open, in hours. Defaults to 72.",
            "nullable": True,
        },
        "mode": {
            "type": "string",
            "description": "Task mode: bounty (default), claim, pitch, benchmark, or auction.",
            "nullable": True,
        },
        "tags": {
            "type": "string",
            "description": "Optional comma-separated tags.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(
        self,
        api_url="https://api.taskmarket.dev",
        timeout=20,
        max_spend_usdc=10.0,
        cli_path="taskmarket",
    ):
        super().__init__()
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.max_spend_usdc = float(max_spend_usdc)
        self.cli_path = cli_path

    def forward(
        self,
        description: str,
        reward_usdc: float,
        confirm: bool,
        duration_hours: int | None = None,
        mode: str | None = None,
        tags: str | None = None,
    ) -> str:
        import shutil
        import subprocess

        if confirm is not True:
            return (
                "Refused: creating a Taskmarket task spends USDC from the CLI wallet. "
                "Re-run with confirm=True after the user authorizes the spend. "
                f"Requested {reward_usdc} USDC, cap {self.max_spend_usdc} USDC."
            )
        try:
            reward = float(reward_usdc)
        except (TypeError, ValueError):
            return "Error: reward_usdc must be a number."
        if reward <= 0:
            return "Error: reward_usdc must be greater than 0."
        if reward > self.max_spend_usdc:
            return (
                f"Refused: {reward} USDC exceeds this tool's max_spend_usdc "
                f"({self.max_spend_usdc}). Raise the cap in the constructor only with "
                "user authorization."
            )
        if not (description or "").strip():
            return "Error: description is required."

        cli = shutil.which(self.cli_path) if self.cli_path == "taskmarket" else self.cli_path
        if self.cli_path == "taskmarket" and not cli:
            return (
                "Refused to spend: official CLI not found. Install and authorize it "
                "yourself, then retry:\n"
                "  npm install -g @lucid-agents/taskmarket && taskmarket init\n"
                "This tool never accepts a private key."
            )
        executable = cli or self.cli_path
        hours = 72 if duration_hours is None else int(duration_hours)
        task_mode = (mode or "bounty").strip() or "bounty"
        command = [
            executable,
            "task",
            "create",
            "--description",
            description.strip(),
            "--reward",
            str(reward),
            "--duration",
            str(hours),
            "--mode",
            task_mode,
        ]
        if tags:
            command.extend(["--tags", tags])
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "Error: taskmarket CLI timed out before the create call settled."
        except OSError as error:
            return f"Error running taskmarket CLI: {error}"

        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            return (
                f"CLI exited {completed.returncode}. Funds were not assumed spent. "
                f"Output:\n{output.strip() or '(empty)'}"
            )
        return f"Taskmarket CLI accepted the create command.\n{output.strip()}"


def taskmarket_tools(max_spend_usdc: float = 10.0, api_url: str = "https://api.taskmarket.dev"):
    """Return the Taskmarket toolset for a CodeAgent or ToolCallingAgent."""
    return [
        BrowseTaskMarketTool(api_url=api_url),
        GetTaskMarketTaskTool(api_url=api_url),
        ListTaskMarketSubmissionsTool(api_url=api_url),
        CreateTaskMarketTaskTool(api_url=api_url, max_spend_usdc=max_spend_usdc),
    ]


__all__ = [
    "BrowseTaskMarketTool",
    "CreateTaskMarketTaskTool",
    "DEFAULT_TASKMARKET_API_URL",
    "GetTaskMarketTaskTool",
    "ListTaskMarketSubmissionsTool",
    "taskmarket_tools",
]
