from argparse import Namespace
from unittest.mock import Mock, patch

import pytest

from smolagents.cli import (
    _discover_default_skill_paths,
    _load_tools,
    build_agent,
    load_model,
    main,
    run_smolagent,
)
from smolagents.local_python_executor import CodeOutput, LocalPythonExecutor
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIModel, TransformersModel


@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_fireworks_api_key")
    monkeypatch.setenv("HF_API_KEY", "test_hf_api_key")


def test_load_model_openai_model(set_env_vars):
    with patch("openai.OpenAI") as mock_openai:
        model = load_model("OpenAIModel", "test_model_id")
    assert isinstance(model, OpenAIModel)
    assert model.model_id == "test_model_id"
    assert mock_openai.call_count == 1
    assert mock_openai.call_args.kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert mock_openai.call_args.kwargs["api_key"] == "test_fireworks_api_key"


def test_load_model_litellm_model():
    pytest.importorskip("litellm")
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


def test_run_smolagent_calls_agent_run():
    with patch("smolagents.cli.build_agent") as mock_build_agent:
        mock_agent = Mock()
        mock_build_agent.return_value = mock_agent

        run_smolagent("test_prompt", [], "InferenceClientModel", "test_model_id", provider="hf-inference")

    assert len(mock_build_agent.call_args_list) == 1
    assert mock_build_agent.call_args.args == ([], "InferenceClientModel", "test_model_id")
    assert mock_build_agent.call_args.kwargs == {
        "api_base": None,
        "api_key": None,
        "imports": None,
        "provider": "hf-inference",
        "action_type": "code",
    }
    mock_agent.run.assert_called_once_with("test_prompt")


def test_build_agent_tool_calling_mode():
    with patch("smolagents.cli._discover_default_skill_paths", return_value=["/tmp/release-skill"]), patch(
        "smolagents.cli.load_model"
    ) as mock_load_model:
        mock_load_model.return_value = "mock_model"
        with patch("smolagents.cli.ToolCallingAgent") as mock_tool_calling_agent:
            build_agent([], "InferenceClientModel", "test_model_id", action_type="tool_calling")

    assert len(mock_tool_calling_agent.call_args_list) == 1
    assert mock_tool_calling_agent.call_args.kwargs == {
        "tools": [],
        "model": "mock_model",
        "skills": ["/tmp/release-skill"],
        "stream_outputs": True,
        "logger": None,
    }


def test_build_agent_code_mode_passes_default_skills():
    with patch("smolagents.cli._discover_default_skill_paths", return_value=["/tmp/security-skill"]), patch(
        "smolagents.cli.load_model"
    ) as mock_load_model:
        mock_load_model.return_value = "mock_model"
        with patch("smolagents.cli.CodeAgent") as mock_code_agent:
            build_agent(
                [],
                "InferenceClientModel",
                "test_model_id",
                action_type="code",
                imports=["numpy"],
            )

    assert len(mock_code_agent.call_args_list) == 1
    assert mock_code_agent.call_args.kwargs == {
        "tools": [],
        "model": "mock_model",
        "additional_authorized_imports": ["numpy"],
        "skills": ["/tmp/security-skill"],
        "stream_outputs": True,
        "logger": None,
    }


def test_discover_default_skill_paths_uses_only_working_directory(tmp_path):
    project_root = tmp_path / "project"
    nested_workdir = project_root / "src" / "pkg"
    nested_workdir.mkdir(parents=True)

    first_skill_file = project_root / ".agents" / "skills" / "release-manager" / "SKILL.md"
    second_skill_file = project_root / ".agents" / "skills" / "qa-review" / "SKILL.md"
    first_skill_file.parent.mkdir(parents=True)
    second_skill_file.parent.mkdir(parents=True)
    first_skill_file.write_text("---\nname: release-manager\ndescription: Release workflow\n---\n", encoding="utf-8")
    second_skill_file.write_text("---\nname: qa-review\ndescription: QA workflow\n---\n", encoding="utf-8")

    discovered = _discover_default_skill_paths(nested_workdir)

    assert discovered == []

    direct_skill_file = nested_workdir / ".agents" / "skills" / "local-skill" / "SKILL.md"
    direct_skill_file.parent.mkdir(parents=True)
    direct_skill_file.write_text("---\nname: local-skill\ndescription: Local workflow\n---\n", encoding="utf-8")

    discovered = _discover_default_skill_paths(nested_workdir)

    assert discovered == [str((nested_workdir / ".agents" / "skills" / "local-skill").resolve())]


def test_load_tools_missing_extra_shows_clear_hint():
    class BrokenWebSearchTool:
        def __init__(self):
            raise ImportError("No module named 'ddgs'")

    with patch.dict("smolagents.cli.TOOL_MAPPING", {"web_search": BrokenWebSearchTool}):
        with pytest.raises(ModuleNotFoundError, match=r"smolagents\[toolkit\]"):
            _load_tools(["web_search"])


def test_main_without_prompt_launches_tui():
    args = Namespace(
        prompt=None,
        tools=["web_search"],
        model_type="InferenceClientModel",
        model_id="test_model_id",
        provider=None,
        api_base=None,
        api_key=None,
        imports=[],
        action_type="code",
    )

    with patch("smolagents.cli.parse_arguments", return_value=args), patch(
        "smolagents.cli.launch_terminal_ui"
    ) as mock_launch_terminal_ui:
        main()

    mock_launch_terminal_ui.assert_called_once_with(
        ["web_search"],
        "InferenceClientModel",
        "test_model_id",
        provider=None,
        api_base=None,
        api_key=None,
        imports=[],
        action_type="code",
    )


def test_main_with_prompt_runs_single_task():
    args = Namespace(
        prompt="test prompt",
        tools=["web_search"],
        model_type="InferenceClientModel",
        model_id="test_model_id",
        provider="hf-inference",
        api_base=None,
        api_key=None,
        imports=["pandas"],
        action_type="tool_calling",
    )

    with patch("smolagents.cli.parse_arguments", return_value=args), patch("smolagents.cli.run_smolagent") as mock_run:
        main()

    mock_run.assert_called_once_with(
        "test prompt",
        ["web_search"],
        "InferenceClientModel",
        "test_model_id",
        provider="hf-inference",
        api_base=None,
        api_key=None,
        imports=["pandas"],
        action_type="tool_calling",
    )


def test_vision_web_browser_main():
    pytest.importorskip("helium")
    pytest.importorskip("selenium")
    with patch("smolagents.vision_web_browser.helium"):
        with patch("smolagents.vision_web_browser.load_model") as mock_load_model:
            mock_load_model.return_value = "mock_model"
            with patch("smolagents.vision_web_browser.CodeAgent") as mock_code_agent:
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
