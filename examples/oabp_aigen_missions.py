"""Discover and optionally submit AIGEN OABP missions with smolagents.

Run the default scripted workflow from the repository root with:

    PYTHONPATH=src python examples/oabp_aigen_missions.py

The default workflow calls the same smolagents tools directly, so it can be run
without an LLM API key and without accidentally submitting to a live mission.
Use ``--run-agent`` to let a CodeAgent orchestrate those tools.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import requests

from smolagents import CodeAgent, tool


DEFAULT_BASE_URL = "https://cryptogenesis.duckdns.org"
DEFAULT_AGENT_ID = "example-smolagents-oabp-agent"
DEFAULT_TIMEOUT = 20
USER_AGENT = "smolagents-oabp-example/1.0"


class OABPRequestError(RuntimeError):
    """Raised when no compatible OABP endpoint returns usable JSON."""


@dataclass(frozen=True)
class OABPClient:
    base_url: str = DEFAULT_BASE_URL
    timeout: int = DEFAULT_TIMEOUT

    def url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def request_json(
        self,
        method: str,
        candidate_paths: list[str],
        payload: dict[str, Any] | None = None,
    ) -> tuple[str, Any]:
        errors: list[str] = []
        for path in candidate_paths:
            try:
                response = requests.request(
                    method,
                    self.url(path),
                    json=payload,
                    timeout=self.timeout,
                    headers={"User-Agent": USER_AGENT},
                )
                if response.status_code >= 400:
                    errors.append(f"{method} {path}: HTTP {response.status_code}")
                    continue
                return path, response.json()
            except requests.RequestException as exc:
                errors.append(f"{method} {path}: {exc}")
            except ValueError as exc:
                errors.append(f"{method} {path}: invalid JSON ({exc})")
        raise OABPRequestError("; ".join(errors))


def normalize_missions(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [mission for mission in data if isinstance(mission, dict)]
    if isinstance(data, dict):
        for key in ("missions", "data", "items"):
            missions = data.get(key)
            if isinstance(missions, list):
                return [mission for mission in missions if isinstance(mission, dict)]
    return []


def mission_submission_count(mission: dict[str, Any]) -> int:
    submissions = mission.get("submissions")
    if isinstance(submissions, list):
        return len(submissions)
    try:
        return int(mission.get("submission_count") or 0)
    except (TypeError, ValueError):
        return 0


def mission_reward(mission: dict[str, Any]) -> int:
    reward = mission.get("reward") if isinstance(mission.get("reward"), dict) else {}
    amount = mission.get("reward_aigen") or mission.get("reward_amount") or reward.get("amount") or 0
    try:
        return int(amount)
    except (TypeError, ValueError):
        return 0


def choose_mission(missions: list[dict[str, Any]], requested_mission_id: str | None = None) -> dict[str, Any] | None:
    if requested_mission_id:
        return next((mission for mission in missions if mission.get("id") == requested_mission_id), None)
    if not missions:
        return None
    return sorted(missions, key=lambda mission: (mission_submission_count(mission), -mission_reward(mission)))[0]


@tool
def fetch_open_aigen_missions(base_url: str = DEFAULT_BASE_URL) -> str:
    """
    Fetch open missions from an OABP-compatible AIGEN server.

    Args:
        base_url: Base URL of the OABP server.

    Returns:
        A JSON string containing the endpoint used and the open missions.
    """
    endpoint, data = OABPClient(base_url).request_json("GET", ["/missions/active", "/api/missions", "/missions"])
    missions = normalize_missions(data)
    return json.dumps({"endpoint": endpoint, "count": len(missions), "missions": missions}, sort_keys=True)


@tool
def read_aigen_mission(mission_id: str, base_url: str = DEFAULT_BASE_URL) -> str:
    """
    Read details for one OABP mission.

    Args:
        mission_id: Mission identifier returned by the OABP mission list.
        base_url: Base URL of the OABP server.

    Returns:
        A JSON string containing mission details.
    """
    endpoint, data = OABPClient(base_url).request_json(
        "GET", [f"/missions/{mission_id}", f"/api/missions/{mission_id}"]
    )
    return json.dumps({"endpoint": endpoint, "mission": data}, sort_keys=True)


@tool
def submit_aigen_mission(
    mission_id: str,
    agent_id: str,
    content: str,
    submitter_wallet: str = "",
    base_url: str = DEFAULT_BASE_URL,
) -> str:
    """
    Submit solution content to an OABP mission.

    Args:
        mission_id: Mission identifier to submit to.
        agent_id: Public OABP agent identifier.
        content: Proof URL or solution content requested by the mission.
        submitter_wallet: Optional wallet address to include with the submission.
        base_url: Base URL of the OABP server.

    Returns:
        A JSON string containing the submit endpoint and response.
    """
    payload: dict[str, Any] = {
        "submitter_agent_id": agent_id,
        "proof": content,
        "metadata": {"client": "smolagents-oabp-example", "framework": "smolagents"},
    }
    if submitter_wallet:
        payload["submitter_wallet"] = submitter_wallet

    endpoint, data = OABPClient(base_url).request_json(
        "POST",
        [f"/api/missions/{mission_id}/submit", f"/missions/{mission_id}/submit"],
        payload=payload,
    )
    return json.dumps({"endpoint": endpoint, "response": data}, sort_keys=True)


def build_oabp_agent(model: Any) -> CodeAgent:
    """Create a CodeAgent with OABP tools attached."""
    return CodeAgent(tools=[fetch_open_aigen_missions, read_aigen_mission, submit_aigen_mission], model=model)


def build_agent_task(args: argparse.Namespace) -> str:
    mission_hint = f"Use mission id {args.mission_id!r}." if args.mission_id else "Pick one open mission."
    submit_instruction = (
        f"Submit this proof content: {args.content!r}."
        if args.submit
        else "Do not submit anything; only report the selected mission and endpoint data."
    )
    wallet_instruction = (
        f"Include submitter wallet {args.submitter_wallet!r} if you submit." if args.submitter_wallet else ""
    )
    return (
        f"Use the OABP tools against {args.base_url}. Fetch open AIGEN missions. "
        f"{mission_hint} Read that mission's details. {submit_instruction} "
        f"Use public agent id {args.agent_id!r}. {wallet_instruction} Return a concise JSON-like summary."
    )


def run_agent_workflow(args: argparse.Namespace) -> Any:
    """Let a CodeAgent call the OABP tools. This requires a configured model provider."""
    from smolagents import InferenceClientModel

    agent = build_oabp_agent(InferenceClientModel(model_id=args.model_id))
    return agent.run(build_agent_task(args))


def run_scripted_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Run the same OABP tools deterministically so the example works without an API key."""
    mission_listing = json.loads(fetch_open_aigen_missions.forward(args.base_url))
    selected = choose_mission(mission_listing["missions"], args.mission_id)
    if selected is None:
        return {"ok": False, "error": "No active missions returned by the OABP server."}

    mission_id = selected["id"]
    mission_detail = json.loads(read_aigen_mission.forward(mission_id, args.base_url))
    content = args.content or (
        f"smolagents OABP example selected mission {mission_id}. "
        "Pass --content with the mission's requested proof and --submit to send it."
    )

    submission = None
    if args.submit:
        submission = json.loads(
            submit_aigen_mission.forward(mission_id, args.agent_id, content, args.submitter_wallet, args.base_url)
        )

    return {
        "ok": True,
        "agent_id": args.agent_id,
        "mission_list_endpoint": mission_listing["endpoint"],
        "mission_count": mission_listing["count"],
        "selected_mission": selected,
        "mission_detail_endpoint": mission_detail["endpoint"],
        "submitted": submission is not None,
        "submission": submission,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and optionally submit AIGEN OABP missions.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--agent-id", default=DEFAULT_AGENT_ID)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--mission-id", default=None)
    parser.add_argument("--submitter-wallet", default="")
    parser.add_argument("--content", default="")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--run-agent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    result = run_agent_workflow(parsed_args) if parsed_args.run_agent else run_scripted_workflow(parsed_args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
