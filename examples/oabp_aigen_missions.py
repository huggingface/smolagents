# How to run with uv:
#   uv run oabp_aigen_missions.py
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "smolagents @ file:///<path-to-smolagents>",
# ]
# ///

"""Discover and submit AIGEN OABP missions with a smolagents tool.

This example uses the Open Agent Bounty Protocol (OABP/AIP-1) reference API at
https://cryptogenesis.duckdns.org/api. It defaults to dry-run mode so you can
inspect the selected mission before submitting anything.

Set these environment variables to submit a real proof:

    OABP_AGENT_ID=your_agent_id
    OABP_PROOF=https://github.com/your/repo-or-pr
    OABP_SUBMIT=1

Optional:

    OABP_BASE_URL=https://cryptogenesis.duckdns.org/api
    OABP_KEYWORD=smolagents
    OABP_INSECURE_SKIP_VERIFY=1  # only for local test environments with custom certificates
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from smolagents import CodeAgent, InferenceClientModel, Tool


class OABPMissionTool(Tool):
    name = "oabp_mission_tool"
    description = (
        "Discover open OABP/AIGEN missions, pick one by keyword, and optionally submit a proof URL."
    )
    inputs = {
        "keyword": {
            "type": "string",
            "description": "Keyword used to rank missions by title and description.",
            "nullable": True,
        },
        "proof": {
            "type": "string",
            "description": "Proof URL to submit. Leave blank for dry-run discovery.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, base_url: str, agent_id: str, submit: bool = False):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.agent_id = agent_id
        self.submit = submit

    def forward(self, keyword: str | None = None, proof: str | None = None) -> str:
        missions = self._get("/missions").get("missions", [])
        mission = self._pick_mission(missions, keyword or "")
        if mission is None:
            return "No open OABP missions found."

        result: dict[str, Any] = {
            "selected_mission": {
                "id": mission["id"],
                "title": mission["title"],
                "reward_aigen": mission.get("reward_aigen"),
            },
            "dry_run": not self.submit,
        }

        if not self.submit:
            result["next_step"] = (
                "Set OABP_SUBMIT=1 and OABP_PROOF to submit a real proof URL for this mission."
            )
            return json.dumps(result, indent=2)

        if not proof:
            raise ValueError("proof is required when OABP_SUBMIT=1")

        result["submission"] = self._post(
            f"/missions/{urllib.parse.quote(mission['id'], safe='')}/submit",
            {
                "submitter_agent_id": self.agent_id,
                "proof": proof,
                "metadata": {"framework": "smolagents", "example": "oabp_aigen_missions.py"},
            },
        )
        return json.dumps(result, indent=2)

    def _pick_mission(self, missions: list[dict[str, Any]], keyword: str) -> dict[str, Any] | None:
        open_missions = [mission for mission in missions if mission.get("status", "open") == "open"]
        if not open_missions:
            return None

        if not keyword:
            return max(open_missions, key=lambda mission: int(mission.get("reward_aigen") or 0))

        keyword = keyword.lower()

        def score(mission: dict[str, Any]) -> tuple[int, int]:
            haystack = f"{mission.get('title', '')} {mission.get('description', '')}".lower()
            return (keyword in haystack, int(mission.get("reward_aigen") or 0))

        return max(open_missions, key=score)

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        context = None
        if os.getenv("OABP_INSECURE_SKIP_VERIFY") == "1":
            context = ssl._create_unverified_context()

        try:
            with urllib.request.urlopen(request, timeout=20, context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OABP request failed: HTTP {error.code} {body}") from error


def main() -> None:
    base_url = os.getenv("OABP_BASE_URL", "https://cryptogenesis.duckdns.org/api")
    agent_id = os.getenv("OABP_AGENT_ID", "example_smolagent")
    keyword = os.getenv("OABP_KEYWORD", "smolagents")
    proof = os.getenv("OABP_PROOF")
    submit = os.getenv("OABP_SUBMIT") == "1"

    tool = OABPMissionTool(base_url=base_url, agent_id=agent_id, submit=submit)
    model = InferenceClientModel()
    agent = CodeAgent(tools=[tool], model=model)

    agent.run(
        "Use the OABP mission tool to find an open AIGEN mission matching "
        f"{keyword!r}. If a proof URL is configured, submit it; otherwise report the dry-run result."
    )


if __name__ == "__main__":
    main()
