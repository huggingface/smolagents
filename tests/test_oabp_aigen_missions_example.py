import argparse
import importlib.util
import sys
from pathlib import Path


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "oabp_aigen_missions.py"


def load_example_module():
    spec = importlib.util.spec_from_file_location("oabp_aigen_missions", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_missions_accepts_common_response_shapes():
    example = load_example_module()

    missions = [{"id": "mission-1"}, {"id": "mission-2"}, "ignored"]

    assert example.normalize_missions(missions) == missions[:2]
    assert example.normalize_missions({"missions": missions}) == missions[:2]
    assert example.normalize_missions({"data": missions}) == missions[:2]
    assert example.normalize_missions({"items": missions}) == missions[:2]
    assert example.normalize_missions({"unexpected": missions}) == []


def test_choose_mission_prefers_fewer_submissions_then_higher_reward():
    example = load_example_module()

    missions = [
        {"id": "busy", "submission_count": 2, "reward_aigen": 1000},
        {"id": "low-reward", "submission_count": 0, "reward_aigen": 50},
        {"id": "best", "submission_count": 0, "reward_aigen": 200},
    ]

    assert example.choose_mission(missions)["id"] == "best"
    assert example.choose_mission(missions, requested_mission_id="busy")["id"] == "busy"
    assert example.choose_mission([], requested_mission_id="missing") is None


def test_build_agent_task_keeps_default_run_read_only():
    example = load_example_module()
    args = argparse.Namespace(
        agent_id="agent-1",
        base_url="https://example.test",
        content="proof",
        mission_id="mission-1",
        submit=False,
        submitter_wallet="",
    )

    task = example.build_agent_task(args)

    assert "Do not submit anything" in task
    assert "mission-1" in task
    assert "agent-1" in task
