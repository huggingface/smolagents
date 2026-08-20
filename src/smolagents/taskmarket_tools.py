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

"""Taskmarket tools with explicit human approval around requester spending."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any

import requests

from .tools import Tool, ToolCollection


TASKMARKET_API_BASE = "https://api.taskmarket.dev"
BASE_CHAIN_ID = 8453
BASE_NETWORK_NAME = "Base Mainnet"
BASE_USDC_CONTRACT = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
APPROVAL_TTL_SECONDS = 300
SUPPORTED_MODES = {"bounty", "claim", "pitch", "benchmark"}
SUPPORTED_TASK_VISIBILITY = {"public", "unlisted"}
SUPPORTED_SUBMISSION_VISIBILITY = {"public", "reveal_all", "winner_only", "never"}


class TaskmarketError(RuntimeError):
    """Base error for the Taskmarket integration."""


class TaskmarketCLIError(TaskmarketError):
    """Raised when a read-only Taskmarket CLI command fails."""


class AmbiguousSettlementError(TaskmarketError):
    """Raised when a paid CLI command may have settled but its result is unknown."""


def _parse_decimal(value: str | int | float | Decimal, field: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a valid decimal number.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite.")
    return parsed


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@dataclass(frozen=True)
class _TaskSpec:
    description: str
    deliverables: tuple[str, ...]
    reward_usdc: Decimal
    duration_hours: Decimal
    mode: str
    task_visibility: str
    submission_visibility: str

    @property
    def canonical_description(self) -> str:
        deliverables = "\n".join(f"- {item}" for item in self.deliverables)
        return f"{self.description}\n\nDeliverables:\n{deliverables}"


class _TaskmarketClient:
    """Process and HTTP boundary for public reads and first-party CLI writes."""

    def __init__(
        self,
        *,
        api_base: str = TASKMARKET_API_BASE,
        executable: str = "taskmarket",
        timeout_seconds: int = 90,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def _get_json(self, path: str, *, params: dict[str, str] | None = None) -> Any:
        try:
            response = requests.get(
                f"{self.api_base}{path}",
                params=params,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise TaskmarketError(f"Taskmarket API read failed for {path}: {exc}") from exc

    def _run_json(self, args: list[str], *, payment_command: bool = False) -> Any:
        command = [self.executable, *args]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            if payment_command:
                raise AmbiguousSettlementError(
                    "Taskmarket payment command timed out. Settlement is unknown; do not retry blindly."
                ) from exc
            raise TaskmarketCLIError(f"Taskmarket command timed out: {' '.join(args)}") from exc
        except OSError as exc:
            raise TaskmarketCLIError(f"Unable to execute Taskmarket CLI '{self.executable}': {exc}") from exc

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        try:
            envelope = json.loads(stdout) if stdout else None
        except json.JSONDecodeError as exc:
            if payment_command:
                raise AmbiguousSettlementError(
                    "Taskmarket payment command returned unreadable output. Settlement is unknown; do not retry blindly."
                ) from exc
            raise TaskmarketCLIError(f"Taskmarket returned invalid JSON. stderr={stderr!r}") from exc

        if result.returncode != 0 or not isinstance(envelope, dict) or envelope.get("ok") is not True:
            error_text = stderr
            if isinstance(envelope, dict):
                error_text = str(envelope.get("error") or envelope)
            if payment_command:
                raise AmbiguousSettlementError(
                    f"Taskmarket payment command did not confirm success ({error_text}). "
                    "Settlement is unknown; do not retry blindly."
                )
            raise TaskmarketCLIError(f"Taskmarket command failed: {error_text}")

        return envelope.get("data")

    def list_tasks(self, status: str, limit: int) -> Any:
        return self._get_json("/api/tasks", params={"status": status, "limit": str(limit)})

    def get_task(self, task_id: str) -> Any:
        return self._get_json(f"/api/tasks/{task_id}")

    def list_submissions(self, task_id: str) -> Any:
        return self._get_json(f"/api/tasks/{task_id}/submissions")

    def network_info(self) -> dict[str, Any]:
        data = self._run_json(["deposit"])
        if not isinstance(data, dict):
            raise TaskmarketCLIError("Unexpected response from `taskmarket deposit`.")
        return data

    def stats(self) -> dict[str, Any]:
        data = self._run_json(["stats"])
        if not isinstance(data, dict):
            raise TaskmarketCLIError("Unexpected response from `taskmarket stats`.")
        return data

    def create_task(self, spec: _TaskSpec) -> dict[str, Any]:
        args = [
            "task",
            "create",
            "--description",
            spec.canonical_description,
            "--reward",
            _decimal_text(spec.reward_usdc),
            "--duration",
            _decimal_text(spec.duration_hours),
            "--mode",
            spec.mode,
            "--task-visibility",
            spec.task_visibility,
            "--submission-visibility",
            spec.submission_visibility,
        ]
        data = self._run_json(args, payment_command=True)
        if not isinstance(data, dict) or not data.get("taskId"):
            raise AmbiguousSettlementError(
                "Taskmarket did not return a taskId after the paid create command. "
                "Settlement is unknown; do not retry blindly."
            )
        return data


class TaskmarketRequesterSession:
    """Stateful requester safety boundary shared by the Taskmarket tools."""

    def __init__(
        self,
        *,
        max_spend_usdc: str | int | float | Decimal,
        api_base: str = TASKMARKET_API_BASE,
        executable: str = "taskmarket",
        timeout_seconds: int = 90,
    ) -> None:
        max_spend = _parse_decimal(max_spend_usdc, "max_spend_usdc")
        if max_spend <= 0:
            raise ValueError("max_spend_usdc must be greater than zero.")
        self.max_spend_usdc = max_spend
        self.client = _TaskmarketClient(
            api_base=api_base,
            executable=executable,
            timeout_seconds=timeout_seconds,
        )
        self._previews: dict[str, _TaskSpec] = {}
        self._approved: dict[str, datetime] = {}
        self._ambiguous: set[str] = set()

    def _validate_network(self) -> dict[str, Any]:
        network = self.client.network_info()
        chain_id = network.get("chainId")
        name = str(network.get("network", ""))
        currency = str(network.get("currency", ""))
        contract = str(network.get("usdcContract", ""))

        if (
            str(chain_id) != str(BASE_CHAIN_ID)
            or name.lower() not in {"base", "base mainnet"}
            or currency.upper() != "USDC"
            or contract.lower() != BASE_USDC_CONTRACT.lower()
        ):
            raise TaskmarketCLIError(
                "Refusing to spend: Taskmarket CLI is not configured for Base Mainnet "
                f"(chainId {BASE_CHAIN_ID}) and the expected USDC contract."
            )
        return network

    def preview_task(
        self,
        *,
        description: str,
        deliverables: list[str],
        reward_usdc: str | int | float | Decimal,
        duration_hours: str | int | float | Decimal,
        mode: str = "bounty",
        task_visibility: str = "public",
        submission_visibility: str = "public",
    ) -> dict[str, Any]:
        description = description.strip()
        clean_deliverables = tuple(item.strip() for item in deliverables if item.strip())
        reward = _parse_decimal(reward_usdc, "reward_usdc")
        duration = _parse_decimal(duration_hours, "duration_hours")

        if not description:
            raise ValueError("description must not be empty.")
        if not clean_deliverables:
            raise ValueError("At least one deliverable is required.")
        if reward <= 0:
            raise ValueError("reward_usdc must be greater than zero.")
        if reward > self.max_spend_usdc:
            raise ValueError(
                f"reward_usdc {_decimal_text(reward)} exceeds configured maximum spend "
                f"{_decimal_text(self.max_spend_usdc)} USDC."
            )
        if duration <= 0:
            raise ValueError("duration_hours must be greater than zero.")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode: {mode}.")
        if task_visibility not in SUPPORTED_TASK_VISIBILITY:
            raise ValueError("Only public and unlisted tasks are supported by this integration.")
        if submission_visibility not in SUPPORTED_SUBMISSION_VISIBILITY:
            raise ValueError(f"Unsupported submission visibility: {submission_visibility}.")

        spec = _TaskSpec(
            description=description,
            deliverables=clean_deliverables,
            reward_usdc=reward,
            duration_hours=duration,
            mode=mode,
            task_visibility=task_visibility,
            submission_visibility=submission_visibility,
        )

        network = self._validate_network()
        stats = self.client.stats()
        preview_id = uuid.uuid4().hex
        self._previews[preview_id] = spec
        now = datetime.now(UTC)
        deadline_estimate = now + timedelta(hours=float(duration))

        return {
            "preview_id": preview_id,
            "description": description,
            "deliverables": list(clean_deliverables),
            "canonical_task_description": spec.canonical_description,
            "reward_usdc": _decimal_text(reward),
            "duration_hours": _decimal_text(duration),
            "deadline_estimate_utc": deadline_estimate.isoformat(),
            "deadline_rule": "Taskmarket sets expiry from the create transaction time using duration_hours.",
            "mode": mode,
            "task_visibility": task_visibility,
            "submission_visibility": submission_visibility,
            "network": BASE_NETWORK_NAME,
            "chain_id": BASE_CHAIN_ID,
            "currency": "USDC",
            "usdc_contract": BASE_USDC_CONTRACT,
            "maximum_spend_usdc": _decimal_text(reward),
            "configured_spend_cap_usdc": _decimal_text(self.max_spend_usdc),
            "wallet_address": stats.get("address") or network.get("address"),
            "wallet_balance_usdc": stats.get("balanceUsdc"),
            "human_approval_required": True,
            "approval_expires_seconds": APPROVAL_TTL_SECONDS,
            "approval_instruction": (
                "Show this preview to the user, then have trusted host code call "
                "TaskmarketToolCollection.authorize(preview_id). The authorize method is not an agent tool."
            ),
        }

    def authorize(self, preview_id: str) -> None:
        """Record fresh host-side human approval for one exact preview."""
        if preview_id not in self._previews:
            raise ValueError("Unknown preview_id. Generate a fresh preview first.")
        if preview_id in self._ambiguous:
            raise ValueError("This preview has ambiguous settlement state and cannot be re-authorized.")
        self._approved[preview_id] = datetime.now(UTC)

    def create_approved_task(self, preview_id: str) -> dict[str, Any]:
        spec = self._previews.get(preview_id)
        if spec is None:
            raise ValueError("Unknown preview_id. Generate a fresh preview first.")
        approved_at = self._approved.get(preview_id)
        if approved_at is None:
            raise PermissionError("No fresh human authorization exists for this preview.")

        age = (datetime.now(UTC) - approved_at).total_seconds()
        if age > APPROVAL_TTL_SECONDS:
            self._approved.pop(preview_id, None)
            raise PermissionError("Human authorization expired. Generate and approve a fresh preview.")

        self._validate_network()
        stats = self.client.stats()
        available = _parse_decimal(stats.get("balanceUsdc", "0"), "wallet balance")
        if available < spec.reward_usdc:
            raise TaskmarketCLIError(
                f"Insufficient USDC balance: have {_decimal_text(available)}, need {_decimal_text(spec.reward_usdc)}."
            )

        # Consume approval before the paid command. The same approval can never be
        # used to retry an unknown settlement.
        self._approved.pop(preview_id, None)
        try:
            created = self.client.create_task(spec)
        except AmbiguousSettlementError as exc:
            self._ambiguous.add(preview_id)
            return {
                "ok": False,
                "preview_id": preview_id,
                "settlement_status": "unknown",
                "do_not_retry": True,
                "error": str(exc),
                "guidance": "Inspect Taskmarket requester state manually before taking any further paid action.",
            }

        task_id = str(created["taskId"])
        result: dict[str, Any] = {
            "ok": True,
            "preview_id": preview_id,
            "task_id": task_id,
            "api_url": f"{self.client.api_base}/api/tasks/{task_id}",
            "funded_usdc": _decimal_text(spec.reward_usdc),
            "network": BASE_NETWORK_NAME,
            "chain_id": BASE_CHAIN_ID,
            "human_review_required_for_submissions": True,
        }
        try:
            result["task"] = self.client.get_task(task_id)
        except TaskmarketError as exc:
            result["status_read_error"] = str(exc)
        return result


class TaskmarketListTasksTool(Tool):
    name = "taskmarket_list_tasks"
    description = "List public Taskmarket tasks without spending funds."
    inputs = {
        "status": {"type": "string", "description": "Task status, for example 'open'."},
        "limit": {"type": "integer", "description": "Maximum number of tasks to return, from 1 to 100."},
    }
    output_type = "object"

    def __init__(self, session: TaskmarketRequesterSession):
        self.session = session
        super().__init__()

    def forward(self, status: str, limit: int) -> dict[str, Any]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")
        return {"tasks": self.session.client.list_tasks(status, limit)}


class TaskmarketPreviewTaskTool(Tool):
    name = "taskmarket_preview_task"
    description = (
        "Preview an exact Taskmarket requester task before spending. Returns the description, deliverables, reward, "
        "deadline estimate/rule, Base/USDC network, wallet balance, maximum spend, and a preview_id. "
        "This tool cannot authorize spending."
    )
    inputs = {
        "description": {"type": "string", "description": "Complete task description."},
        "deliverables": {"type": "array", "description": "Concrete deliverables expected from workers."},
        "reward_usdc": {"type": "number", "description": "USDC reward to escrow."},
        "duration_hours": {"type": "number", "description": "Task duration in hours."},
    }
    output_type = "object"

    def __init__(self, session: TaskmarketRequesterSession):
        self.session = session
        super().__init__()

    def forward(
        self,
        description: str,
        deliverables: list[str],
        reward_usdc: float,
        duration_hours: float,
    ) -> dict[str, Any]:
        return self.session.preview_task(
            description=description,
            deliverables=deliverables,
            reward_usdc=reward_usdc,
            duration_hours=duration_hours,
        )


class TaskmarketCreateTaskTool(Tool):
    name = "taskmarket_create_task"
    description = (
        "Create and fund a Taskmarket task from a preview_id only after trusted host code has recorded fresh "
        "explicit user authorization. Approval is one-time and consumed before the paid command."
    )
    inputs = {
        "preview_id": {"type": "string", "description": "Opaque preview_id returned by taskmarket_preview_task."}
    }
    output_type = "object"

    def __init__(self, session: TaskmarketRequesterSession):
        self.session = session
        super().__init__()

    def forward(self, preview_id: str) -> dict[str, Any]:
        return self.session.create_approved_task(preview_id)


class TaskmarketTaskStatusTool(Tool):
    name = "taskmarket_task_status"
    description = "Retrieve live Taskmarket task status and details. Read-only."
    inputs = {"task_id": {"type": "string", "description": "Taskmarket task ID."}}
    output_type = "object"

    def __init__(self, session: TaskmarketRequesterSession):
        self.session = session
        super().__init__()

    def forward(self, task_id: str) -> dict[str, Any]:
        return {"task_id": task_id, "task": self.session.client.get_task(task_id)}


class TaskmarketSubmissionsTool(Tool):
    name = "taskmarket_list_submissions"
    description = (
        "List Taskmarket submissions for human review. Read-only; this integration intentionally provides no "
        "automatic accept or reject operation."
    )
    inputs = {"task_id": {"type": "string", "description": "Taskmarket task ID."}}
    output_type = "object"

    def __init__(self, session: TaskmarketRequesterSession):
        self.session = session
        super().__init__()

    def forward(self, task_id: str) -> dict[str, Any]:
        submissions = self.session.client.list_submissions(task_id)
        return {
            "task_id": task_id,
            "submissions": submissions,
            "human_review_required": True,
            "automatic_acceptance": False,
            "automatic_rejection": False,
        }


class TaskmarketToolCollection(ToolCollection):
    """
    Collection of safe Taskmarket requester tools for smolagents.

    The collection exposes read-only discovery/status/submission tools plus a
    preview/create flow. A model can generate a preview, but only trusted host
    code can call :meth:`authorize`; the authorization method is not included in
    ``tools`` and therefore is not available to the agent.
    """

    def __init__(
        self,
        *,
        max_spend_usdc: str | int | float | Decimal,
        api_base: str = TASKMARKET_API_BASE,
        executable: str = "taskmarket",
        timeout_seconds: int = 90,
    ) -> None:
        self.session = TaskmarketRequesterSession(
            max_spend_usdc=max_spend_usdc,
            api_base=api_base,
            executable=executable,
            timeout_seconds=timeout_seconds,
        )
        super().__init__(
            [
                TaskmarketListTasksTool(self.session),
                TaskmarketPreviewTaskTool(self.session),
                TaskmarketCreateTaskTool(self.session),
                TaskmarketTaskStatusTool(self.session),
                TaskmarketSubmissionsTool(self.session),
            ]
        )

    def authorize(self, preview_id: str) -> None:
        """Record fresh explicit user authorization for one exact preview."""
        self.session.authorize(preview_id)


__all__ = [
    "AmbiguousSettlementError",
    "TaskmarketCLIError",
    "TaskmarketCreateTaskTool",
    "TaskmarketError",
    "TaskmarketListTasksTool",
    "TaskmarketPreviewTaskTool",
    "TaskmarketRequesterSession",
    "TaskmarketSubmissionsTool",
    "TaskmarketTaskStatusTool",
    "TaskmarketToolCollection",
]
