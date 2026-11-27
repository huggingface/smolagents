"""InferenceClientModel must forward and validate structured output params."""

from unittest.mock import MagicMock

import pytest

from smolagents.models import ChatMessage, InferenceClientModel, MessageRole


def _model(provider):
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider=provider, token="hf_dummy"
    )
    model.client = MagicMock()
    model.client.chat_completion.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    role="assistant", content='{"thought":"ok","code":"print(1)"}', tool_calls=None
                )
            )
        ],
        usage=MagicMock(prompt_tokens=10, completion_tokens=10),
    )
    return model


def test_generate_forwards_response_format():
    model = _model("cerebras")
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    model.generate(messages, response_format={"type": "json_object"})

    assert model.client.chat_completion.call_args.kwargs["response_format"] == {
        "type": "json_object"
    }


def test_generate_stream_validates_provider():
    model = _model("together")
    messages = [ChatMessage(role=MessageRole.USER, content="Hello")]

    with pytest.raises(ValueError, match="only supports structured outputs"):
        list(model.generate_stream(messages, response_format={"type": "json_object"}))
