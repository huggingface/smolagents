# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from transformers.testing_utils import get_tests_dir

from smolagents.models import (
    AzureOpenAIServerModel,
    ChatMessage,
    HfApiModel,
    LiteLLMModel,
    MessageRole,
    MLXModel,
    OpenAIServerModel,
    TransformersModel,
    get_clean_message_list,
    get_tool_json_schema,
    parse_json_if_needed,
    parse_tool_args_if_needed,
)
from smolagents.tools import tool

from .utils.markers import require_run_all


class ModelTests(unittest.TestCase):
    def test_get_json_schema_has_nullable_args(self):
        @tool
        def get_weather(location: str, celsius: Optional[bool] = False) -> str:
            """
            Get weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert "nullable" in get_tool_json_schema(get_weather)["function"]["parameters"]["properties"]["celsius"]

    def test_chatmessage_has_model_dumps_json(self):
        message = ChatMessage("user", [{"type": "text", "text": "Hello!"}])
        data = json.loads(message.model_dump_json())
        assert data["content"] == [{"type": "text", "text": "Hello!"}]

    @unittest.skipUnless(sys.platform.startswith("darwin"), "requires macOS")
    def test_get_mlx_message_no_tool(self):
        model = MLXModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        output = model(messages, stop_sequences=["great"]).content
        assert output.startswith("Hello")

    @unittest.skipUnless(sys.platform.startswith("darwin"), "requires macOS")
    def test_get_mlx_message_tricky_stop_sequence(self):
        # In this test HuggingFaceTB/SmolLM2-135M-Instruct generates the token ">'"
        # which is required to test capturing stop_sequences that have extra chars at the end.
        model = MLXModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct", max_tokens=100)
        stop_sequence = " print '>"
        messages = [{"role": "user", "content": [{"type": "text", "text": f"Please{stop_sequence}'"}]}]
        # check our assumption that that ">" is followed by "'"
        assert model.tokenizer.vocab[">'"]
        assert model(messages, stop_sequences=[]).content == f"I'm ready to help you{stop_sequence}'"
        # check stop_sequence capture when output has trailing chars
        assert model(messages, stop_sequences=[stop_sequence]).content == "I'm ready to help you"

    def test_transformers_message_no_tool(self):
        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=5,
            device_map="cpu",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        output = model(messages, stop_sequences=["great"]).content
        assert output == "assistant\nHello"

    def test_transformers_message_vl_no_tool(self):
        from PIL import Image

        img = Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png")
        model = TransformersModel(
            model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
            max_new_tokens=5,
            device_map="cpu",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "image": img}]}]
        output = model(messages, stop_sequences=["great"]).content
        assert output == "Hello! How can"

    def test_parse_tool_args_if_needed(self):
        original_message = ChatMessage(role="user", content=[{"type": "text", "text": "Hello!"}])
        parsed_message = parse_tool_args_if_needed(original_message)
        assert parsed_message == original_message

    def test_parse_json_if_needed(self):
        args = "abc"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == "abc"

        args = '{"a": 3}'
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == {"a": 3}

        args = "3"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3

        args = 3
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3


class TestHfApiModel:
    def test_call_with_custom_role_conversions(self):
        custom_role_conversions = {MessageRole.USER: MessageRole.SYSTEM}
        model = HfApiModel(model_id="test-model", custom_role_conversions=custom_role_conversions)
        model.client = MagicMock()
        messages = [{"role": "user", "content": "Test message"}]
        _ = model(messages)
        # Verify that the role conversion was applied
        assert model.client.chat_completion.call_args.kwargs["messages"][0]["role"] == "system", (
            "role conversion should be applied"
        )

    @require_run_all
    def test_get_hfapi_message_no_tool(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_no_tool_external_provider(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])


class TestLiteLLMModel:
    @pytest.mark.parametrize(
        "model_id, error_flag",
        [
            ("groq/llama-3.3-70b", "Missing API Key"),
            ("cerebras/llama-3.3-70b", "The api_key client option must be set"),
            ("mistral/mistral-tiny", "The api_key client option must be set"),
        ],
    )
    def test_call_different_providers_without_key(self, model_id, error_flag):
        model = LiteLLMModel(model_id=model_id)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test message"}]}]
        with pytest.raises(Exception) as e:
            # This should raise 401 error because of missing API key, not fail for any "bad format" reason
            model(messages)
        assert error_flag in str(e)

    def test_passing_flatten_messages(self):
        model = LiteLLMModel(model_id="groq/llama-3.3-70b", flatten_messages_as_text=False)
        assert not model.flatten_messages_as_text

        model = LiteLLMModel(model_id="fal/llama-3.3-70b", flatten_messages_as_text=True)
        assert model.flatten_messages_as_text


class TestOpenAIServerModel:
    def test_client_kwargs_passed_correctly(self):
        model_id = "gpt-3.5-turbo"
        api_base = "https://api.openai.com/v1"
        api_key = "test_api_key"
        organization = "test_org"
        project = "test_project"
        client_kwargs = {"max_retries": 5}

        with patch("openai.OpenAI") as MockOpenAI:
            _ = OpenAIServerModel(
                model_id=model_id,
                api_base=api_base,
                api_key=api_key,
                organization=organization,
                project=project,
                client_kwargs=client_kwargs,
            )
            MockOpenAI.assert_called_once_with(
                base_url=api_base, api_key=api_key, organization=organization, project=project, max_retries=5
            )


class TestAzureOpenAIServerModel:
    def test_client_kwargs_passed_correctly(self):
        model_id = "gpt-3.5-turbo"
        api_key = "test_api_key"
        api_version = "2023-12-01-preview"
        azure_endpoint = "https://example-resource.azure.openai.com/"
        organization = "test_org"
        project = "test_project"
        client_kwargs = {"max_retries": 5}

        with patch("openai.OpenAI") as MockOpenAI, patch("openai.AzureOpenAI") as MockAzureOpenAI:
            _ = AzureOpenAIServerModel(
                model_id=model_id,
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                organization=organization,
                project=project,
                client_kwargs=client_kwargs,
            )
        assert MockOpenAI.call_count == 0
        MockAzureOpenAI.assert_called_once_with(
            base_url=None,
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            organization=organization,
            project=project,
            max_retries=5,
        )


def test_get_clean_message_list_basic():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
    ]
    result = get_clean_message_list(messages)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"][0]["text"] == "Hello!"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"][0]["text"] == "Hi there!"


def test_get_clean_message_list_role_conversions():
    messages = [
        {"role": "tool-call", "content": [{"type": "text", "text": "Calling tool..."}]},
        {"role": "tool-response", "content": [{"type": "text", "text": "Tool response"}]},
    ]
    result = get_clean_message_list(messages, role_conversions={"tool-call": "assistant", "tool-response": "user"})
    assert len(result) == 2
    assert result[0]["role"] == "assistant"
    assert result[0]["content"][0]["text"] == "Calling tool..."
    assert result[1]["role"] == "user"
    assert result[1]["content"][0]["text"] == "Tool response"


@pytest.mark.parametrize(
    "convert_images_to_image_urls, expected_clean_message",
    [
        (
            False,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "encoded_image"},
                    {"type": "image", "image": "second_encoded_image"},
                ],
            },
        ),
        (
            True,
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,encoded_image"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,second_encoded_image"}},
                ],
            },
        ),
    ],
)
def test_get_clean_message_list_image_encoding(convert_images_to_image_urls, expected_clean_message):
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": b"image_data"}, {"type": "image", "image": b"second_image_data"}],
        }
    ]
    with patch("smolagents.models.encode_image_base64") as mock_encode:
        mock_encode.side_effect = ["encoded_image", "second_encoded_image"]
        result = get_clean_message_list(messages, convert_images_to_image_urls=convert_images_to_image_urls)
        mock_encode.assert_any_call(b"image_data")
        mock_encode.assert_any_call(b"second_image_data")
        assert len(result) == 1
        assert result[0] == expected_clean_message


def test_get_clean_message_list_flatten_messages_as_text():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
    ]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello!How are you?"


@pytest.mark.parametrize(
    "model_class, model_kwargs, patching, expected_flatten_messages_as_text",
    [
        (AzureOpenAIServerModel, {}, ("openai.AzureOpenAI", {}), False),
        (HfApiModel, {}, ("huggingface_hub.InferenceClient", {}), False),
        (LiteLLMModel, {}, None, False),
        (LiteLLMModel, {"model_id": "ollama"}, None, True),
        (LiteLLMModel, {"model_id": "groq"}, None, True),
        (LiteLLMModel, {"model_id": "cerebras"}, None, True),
        (MLXModel, {}, ("mlx_lm.load", {"return_value": (MagicMock(), MagicMock())}), True),
        (OpenAIServerModel, {}, ("openai.OpenAI", {}), False),
        (OpenAIServerModel, {"flatten_messages_as_text": True}, ("openai.OpenAI", {}), True),
        (
            TransformersModel,
            {},
            [
                ("transformers.AutoModelForCausalLM.from_pretrained", {}),
                ("transformers.AutoTokenizer.from_pretrained", {}),
            ],
            True,
        ),
        (
            TransformersModel,
            {},
            [
                (
                    "transformers.AutoModelForCausalLM.from_pretrained",
                    {"side_effect": ValueError("Unrecognized configuration class")},
                ),
                ("transformers.AutoModelForImageTextToText.from_pretrained", {}),
                ("transformers.AutoProcessor.from_pretrained", {}),
            ],
            False,
        ),
    ],
)
def test_flatten_messages_as_text_for_all_models(
    model_class, model_kwargs, patching, expected_flatten_messages_as_text
):
    with ExitStack() as stack:
        if isinstance(patching, list):
            for target, kwargs in patching:
                stack.enter_context(patch(target, **kwargs))
        elif patching:
            target, kwargs = patching
            stack.enter_context(patch(target, **kwargs))

        model = model_class(**{"model_id": "test-model", **model_kwargs})
    assert model.flatten_messages_as_text is expected_flatten_messages_as_text, f"{model_class.__name__} failed"
