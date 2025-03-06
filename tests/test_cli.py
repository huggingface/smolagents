from unittest.mock import patch

import pytest

from smolagents.cli import load_model
from smolagents.models import HfApiModel, LiteLLMModel, OpenAIServerModel, TransformersModel


@pytest.fixture
def set_env_vars(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test_fireworks_api_key")
    monkeypatch.setenv("HF_TOKEN", "test_hf_api_key")


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
        patch("transformers.AutoModelForCausalLM.from_pretrained"),
        patch("transformers.AutoTokenizer.from_pretrained"),
    ):
        model = load_model("TransformersModel", "test_model_id")
    assert isinstance(model, TransformersModel)
    assert model.model_id == "test_model_id"


def test_load_model_hf_api_model(set_env_vars):
    with patch("huggingface_hub.InferenceClient") as huggingface_hub_InferenceClient:
        model = load_model("HfApiModel", "test_model_id")
    assert isinstance(model, HfApiModel)
    assert model.model_id == "test_model_id"
    assert huggingface_hub_InferenceClient.call_count == 1
    assert huggingface_hub_InferenceClient.call_args.kwargs["token"] == "test_hf_api_key"


def test_load_model_invalid_model_type():
    with pytest.raises(ValueError, match="Unsupported model type: InvalidModel"):
        load_model("InvalidModel", "test_model_id")
