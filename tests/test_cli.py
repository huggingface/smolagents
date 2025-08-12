import subprocess
from unittest.mock import patch

import pytest

from smolagents.cli import load_model
from smolagents.local_python_executor import CodeOutput, LocalPythonExecutor
from smolagents.models import InferenceClientModel, LiteLLMModel, OpenAIServerModel, TransformersModel


@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_fireworks_api_key")
    monkeypatch.setenv("HF_TOKEN", "test_hf_api_key")


def test_multi_agent_cli():
    result = subprocess.run(
        ["smolagent", "test task", "--num-agents", "2", "--model-id", "mock-model"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Agent 0 running" in result.stdout
    assert "Agent 1 running" in result.stdout


def test_load_model_openai_server_model(set_env_vars):
    with patch("openai.OpenAI") as MockOpenAI:
        model = load_model("OpenAIServerModel", "test_model_id")
    assert isinstance(model, OpenAIServerModel)
    assert model.model_id == "test_model_id"
    assert MockOpenAI.call_count == 1
    assert MockOpenAI.call_args.kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"
    assert MockOpenAI.call_args.kwargs["api_key"] == "test_fireworks_api_key"


def test_load_model_litellm_model():
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
    with patch("huggingface_hub.InferenceClient") as huggingface_hub_InferenceClient:
        model = load_model("InferenceClientModel", "test_model_id")
    assert isinstance(model, InferenceClientModel)
    assert model.model_id == "test_model_id"
    assert huggingface_hub_InferenceClient.call_count == 1
    assert huggingface_hub_InferenceClient.call_args.kwargs["token"] == "test_hf_api_key"


def test_load_model_invalid_model_type():
    with pytest.raises(ValueError, match="Unsupported model type: InvalidModel"):
        load_model("InvalidModel", "test_model_id")


def test_cli_main(capsys):
    with patch("smolagents.cli.load_model") as mock_load_model:
        mock_load_model.return_value = "mock_model"
        with patch("smolagents.cli.CodeAgent") as mock_code_agent:

            class DummyProcess:
                def __init__(self, target, args):
                    self.target = target
                    self.args = args

                def start(self):
                    self.target(*self.args)

                def join(self):
                    pass

            with patch("smolagents.cli.Process", DummyProcess):
                from smolagents.cli import run_smolagent

                run_smolagent(
                    "test_prompt",
                    [],
                    "InferenceClientModel",
                    "test_model_id",
                    1,
                    provider="hf-inference",
                )
    # load_model
    assert len(mock_load_model.call_args_list) == 1
    assert mock_load_model.call_args.args == ("InferenceClientModel", "test_model_id")
    assert mock_load_model.call_args.kwargs == {"api_base": None, "api_key": None, "provider": "hf-inference"}
    # CodeAgent
    assert len(mock_code_agent.call_args_list) == 1
    assert mock_code_agent.call_args.args == ()
    assert mock_code_agent.call_args.kwargs["tools"] == []
    assert mock_code_agent.call_args.kwargs["model"] == "mock_model"
    assert mock_code_agent.call_args.kwargs["additional_authorized_imports"] is None
    assert mock_code_agent.call_args.kwargs["agent_id"] == 0
    assert "queue_dict" in mock_code_agent.call_args.kwargs
    # agent.run
    assert len(mock_code_agent.return_value.run.call_args_list) == 1
    assert mock_code_agent.return_value.run.call_args.args == ("test_prompt",)
    # print
    captured = capsys.readouterr()
    assert "Agent 0 running with tools: []" in captured.out


def test_vision_web_browser_main():
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
