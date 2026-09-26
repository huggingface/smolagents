from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from smolagents.cli import RunConfig, interactive_mode, load_model, parse_arguments, run_smolagent
from smolagents.local_python_executor import CodeOutput, LocalPythonExecutor
from smolagents.memory import MemoryStep
from smolagents.models import InferenceClientModel, LiteLLMModel, Model, OpenAIModel, TransformersModel
from smolagents.monitoring import LogLevel


@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_fireworks_api_key")
    monkeypatch.setenv("HF_TOKEN", "test_hf_api_key")


def make_run_config(tmp_path: Path, **kwargs) -> RunConfig:
    return replace(
        RunConfig.default(),
        checkpoint_path=str(tmp_path / "runs" / "latest.json"),
        memory_path=str(tmp_path / "memory.md"),
        **kwargs,
    )


def test_load_model_openai_model(set_env_vars):
    with patch("openai.OpenAI") as mock_openai:
        model = load_model("OpenAIModel", "test_model_id")
    assert isinstance(model, OpenAIModel)
    assert model.model_id == "test_model_id"
    assert mock_openai.call_count == 1
    assert mock_openai.call_args.kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "test_fireworks_api_key"


def test_load_model_litellm_model():
    with patch("smolagents.models.LiteLLMModel.create_client", return_value=object()):
        model = load_model("LiteLLMModel", "test_model_id", api_key="test_api_key", api_base="https://api.test.com")
    assert isinstance(model, LiteLLMModel)
    assert model.api_key == "test_api_key"
    assert model.api_base == "https://api.test.com"
    assert model.model_id == "test_model_id"


def test_load_model_transformers_model():
    with (
        patch(
            "transformers.AutoModelForImageTextToText.from_pretrained",
            side_effect=ValueError("Unrecognized configuration class"),
        ),
        patch("transformers.AutoModelForCausalLM.from_pretrained"),
        patch("transformers.AutoTokenizer.from_pretrained"),
    ):
        model = load_model("TransformersModel", "test_model_id")
    assert isinstance(model, TransformersModel)
    assert model.model_id == "test_model_id"


def test_load_model_hf_api_model(set_env_vars):
    with patch("huggingface_hub.InferenceClient") as huggingface_hub_inference_client:
        model = load_model("InferenceClientModel", "test_model_id")
    assert isinstance(model, InferenceClientModel)
    assert model.model_id == "test_model_id"
    assert huggingface_hub_inference_client.call_count == 1
    assert huggingface_hub_inference_client.call_args.kwargs["token"] == "test_hf_api_key"


def test_load_model_invalid_model_type():
    with pytest.raises(ValueError, match="Unsupported model type: InvalidModel"):
        load_model("InvalidModel", "test_model_id")


def test_parse_arguments_defaults_to_long_horizon_profile():
    args = parse_arguments(["hello"])
    run_config = RunConfig.from_args(args)

    assert run_config.autonomy_profile == "long-horizon"
    assert run_config.max_steps == 64
    assert run_config.planning_interval == 4
    assert run_config.use_structured_internal_output is True
    assert run_config.tool_retry_limit == 2
    assert run_config.stagnation_window == 3


def test_run_config_flag_precedence_over_profile():
    args = parse_arguments(
        [
            "hello",
            "--autonomy-profile",
            "legacy",
            "--max-steps",
            "99",
            "--planning-interval",
            "7",
            "--no-use-structured-internal-output",
            "--tool-retry-limit",
            "5",
            "--stagnation-window",
            "4",
        ]
    )
    run_config = RunConfig.from_args(args)

    assert run_config.autonomy_profile == "legacy"
    assert run_config.max_steps == 99
    assert run_config.planning_interval == 7
    assert run_config.use_structured_internal_output is False
    assert run_config.tool_retry_limit == 5
    assert run_config.stagnation_window == 4


def test_interactive_mode_keeps_selected_action_type():
    prompt_answers = ["tool_calling", "web_search", "InferenceClientModel", "test-model", "do a task"]
    with (
        patch("smolagents.cli.Prompt.ask", side_effect=prompt_answers),
        patch("smolagents.cli.Confirm.ask", return_value=False),
    ):
        result = interactive_mode()

    assert result[-1] == "tool_calling"


def test_run_smolagent_passes_autonomy_and_verbosity_to_agent(tmp_path):
    run_config = make_run_config(tmp_path)
    dummy_model = Model(model_id="dummy")

    with (
        patch("smolagents.cli.load_model", return_value=dummy_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "test_prompt",
            "last_plan": None,
            "memory_summary": [],
        }

        output = run_smolagent(
            "test_prompt",
            [],
            "InferenceClientModel",
            "test_model_id",
            provider="hf-inference",
            run_config=run_config,
        )

    assert output == "done"
    assert mock_code_agent.call_args.kwargs["max_steps"] == 64
    assert mock_code_agent.call_args.kwargs["planning_interval"] == 4
    assert mock_code_agent.call_args.kwargs["tool_retry_limit"] == 2
    assert mock_code_agent.call_args.kwargs["stagnation_window"] == 3
    assert mock_code_agent.call_args.kwargs["verbosity_level"] == LogLevel.INFO


def test_run_smolagent_instruction_layering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("project-guideline", encoding="utf-8")
    (tmp_path / "extra.md").write_text("extra-guideline", encoding="utf-8")
    (tmp_path / "memory.md").write_text("memory-guideline", encoding="utf-8")

    run_config = make_run_config(
        tmp_path,
        instructions_files=[str(tmp_path / "extra.md")],
        append_instructions=["inline-guideline"],
    )
    dummy_model = Model(model_id="dummy")

    with (
        patch("smolagents.cli.load_model", return_value=dummy_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "test_prompt",
            "last_plan": None,
            "memory_summary": [],
        }

        run_smolagent("test_prompt", [], "InferenceClientModel", "test_model_id", run_config=run_config)

    instructions = mock_code_agent.call_args.kwargs["instructions"]
    assert "Runtime autonomy contract" in instructions
    assert "project-guideline" in instructions
    assert "extra-guideline" in instructions
    assert "memory-guideline" in instructions
    assert "inline-guideline" in instructions


def test_run_smolagent_resume_uses_checkpoint_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint_path = tmp_path / "runs" / "latest.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        '{"task": "resume-task", "run_state": {"completed_action_steps": 2, "memory_summary": []}}',
        encoding="utf-8",
    )

    run_config = make_run_config(tmp_path, resume=True)
    dummy_model = Model(model_id="dummy")

    with (
        patch("smolagents.cli.load_model", return_value=dummy_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "resume-task",
            "last_plan": None,
            "memory_summary": [],
        }

        run_smolagent(None, [], "InferenceClientModel", "test_model_id", run_config=run_config)

    assert mock_code_agent.return_value.run.call_args.args == ("resume-task",)
    assert "Resume context from previous run" in mock_code_agent.call_args.kwargs["instructions"]


def test_run_smolagent_disables_structured_output_when_backend_not_supported(tmp_path):
    run_config = make_run_config(tmp_path, use_structured_internal_output=True)
    inference_model = InferenceClientModel(model_id="test-model", provider=None, token="hf_test")

    with (
        patch("smolagents.cli.load_model", return_value=inference_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "task",
            "last_plan": None,
            "memory_summary": [],
        }

        run_smolagent("task", [], "InferenceClientModel", "test_model_id", run_config=run_config)

    assert mock_code_agent.call_args.kwargs["use_structured_outputs_internally"] is False


def test_run_smolagent_passes_final_schema_check(tmp_path):
    schema_path = tmp_path / "final-schema.json"
    schema_path.write_text(
        '{"type": "object", "required": ["answer"], "properties": {"answer": {"type": "string"}}}',
        encoding="utf-8",
    )
    run_config = make_run_config(tmp_path, final_schema_path=str(schema_path))
    dummy_model = Model(model_id="dummy")

    with (
        patch("smolagents.cli.load_model", return_value=dummy_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "task",
            "last_plan": None,
            "memory_summary": [],
        }

        run_smolagent("task", [], "InferenceClientModel", "test_model_id", run_config=run_config)

    checks = mock_code_agent.call_args.kwargs["final_answer_checks"]
    assert len(checks) == 1
    check = checks[0]
    assert check('{"answer": "ok"}', memory=None, agent=None) is True
    with pytest.raises(ValueError):
        check('{"wrong": "shape"}', memory=None, agent=None)


def test_run_smolagent_json_events_use_memory_step_callbacks(tmp_path):
    run_config = make_run_config(tmp_path, json_events=True)
    dummy_model = Model(model_id="dummy")

    with (
        patch("smolagents.cli.load_model", return_value=dummy_model),
        patch("smolagents.cli.CodeAgent") as mock_code_agent,
    ):
        mock_code_agent.return_value.run.return_value = "done"
        mock_code_agent.return_value.get_run_state_snapshot.return_value = {
            "step_number": 1,
            "completed_action_steps": 1,
            "pending_task": "task",
            "last_plan": None,
            "memory_summary": [],
        }

        run_smolagent("task", [], "InferenceClientModel", "test_model_id", run_config=run_config)

    step_callbacks = mock_code_agent.call_args.kwargs["step_callbacks"]
    assert MemoryStep in step_callbacks
    assert len(step_callbacks[MemoryStep]) == 1


def test_vision_web_browser_main():
    pytest.importorskip("helium")
    pytest.importorskip("selenium")
    import importlib

    vision_web_browser = importlib.import_module("smolagents.vision_web_browser")

    with patch.object(vision_web_browser, "helium"):
        with patch.object(vision_web_browser, "load_model") as mock_load_model:
            mock_load_model.return_value = "mock_model"
            with patch.object(vision_web_browser, "CodeAgent") as mock_code_agent:
                from smolagents.vision_web_browser import helium_instructions, run_webagent

                run_webagent("test_prompt", "InferenceClientModel", "test_model_id", provider="hf-inference")
    # load_model
    assert len(mock_load_model.call_args_list) == 1
    assert mock_load_model.call_args.args == ("InferenceClientModel", "test_model_id")
    # CodeAgent
    assert len(mock_code_agent.call_args_list) == 1
    assert mock_code_agent.call_args.args == ()
    assert len(mock_code_agent.call_args.kwargs["tools"]) == 4
    assert mock_code_agent.call_args.kwargs["model"] == "mock_model"
    assert mock_code_agent.call_args.kwargs["additional_authorized_imports"] == ["helium"]
    # agent.python_executor
    assert len(mock_code_agent.return_value.python_executor.call_args_list) == 1
    assert mock_code_agent.return_value.python_executor.call_args.args == ("from helium import *",)
    assert LocalPythonExecutor(["helium"])("from helium import *") == CodeOutput(
        output=None, logs="", is_final_answer=False
    )
    # agent.run
    assert len(mock_code_agent.return_value.run.call_args_list) == 1
    assert mock_code_agent.return_value.run.call_args.args == ("test_prompt" + helium_instructions,)
