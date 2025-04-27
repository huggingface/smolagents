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
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import ChatCompletionOutputMessage

from smolagents.models import (
    AmazonBedrockServerModel,
    AzureOpenAIServerModel,
    ChatMessage,
    ChatMessageToolCall,
    HfApiModel,
    InferenceClientModel,
    LiteLLMModel,
    LiteLLMRouterModel,
    MessageRole,
    MLXModel,
    Model,
    OpenAIServerModel,
    TransformersModel,
    get_clean_message_list,
    get_tool_call_from_text,
    get_tool_json_schema,
    parse_json_if_needed,
    supports_stop_parameter,
)
from smolagents.tools import tool

from .utils.markers import require_run_all


class TestModel:
    @pytest.mark.parametrize(
        "model_id, stop_sequences, should_contain_stop",
        [
            ("regular-model", ["stop1", "stop2"], True),  # Regular model should include stop
            ("openai/o3", ["stop1", "stop2"], False),  # o3 model should not include stop
            ("openai/o4-mini", ["stop1", "stop2"], False),  # o4-mini model should not include stop
            ("something/else/o3", ["stop1", "stop2"], False),  # Path ending with o3 should not include stop
            ("something/else/o4-mini", ["stop1", "stop2"], False),  # Path ending with o4-mini should not include stop
            ("o3", ["stop1", "stop2"], False),  # Exact o3 model should not include stop
            ("o4-mini", ["stop1", "stop2"], False),  # Exact o4-mini model should not include stop
            ("regular-model", None, False),  # None stop_sequences should not add stop parameter
        ],
    )
    def test_prepare_completion_kwargs_stop_sequences(self, model_id, stop_sequences, should_contain_stop):
        model = Model()
        model.model_id = model_id
        completion_kwargs = model._prepare_completion_kwargs(
            messages=[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}], stop_sequences=stop_sequences
        )
        # Verify that the stop parameter is only included when appropriate
        if should_contain_stop:
            assert "stop" in completion_kwargs
            assert completion_kwargs["stop"] == stop_sequences
        else:
            assert "stop" not in completion_kwargs

    def test_get_json_schema_has_nullable_args(self):
        @tool
        def get_weather(location: str, celsius: bool | None = False) -> str:
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

    def test_transformers_message_no_tool(self, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT", 30)  # instead of 10
        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=5,
            device_map="cpu",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        output = model.generate(messages, stop_sequences=["great"]).content
        assert output == "assistant\nHello"

        output = model.generate_stream(messages, stop_sequences=["great"])
        output_str = ""
        for el in output:
            output_str += el.content
        assert output_str == "assistant\nHello"

    def test_transformers_message_vl_no_tool(self, shared_datadir, monkeypatch):
        monkeypatch.setattr("huggingface_hub.constants.HF_HUB_DOWNLOAD_TIMEOUT", 30)  # instead of 10
        import PIL.Image

        img = PIL.Image.open(shared_datadir / "000000039769.png")
        model = TransformersModel(
            model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
            max_new_tokens=4,
            device_map="cpu",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "image": img}]}]
        output = model.generate(messages, stop_sequences=["great"]).content
        assert output == "I am"

        output = model.generate_stream(messages, stop_sequences=["great"])
        output_str = ""
        for el in output:
            output_str += el.content
        assert output_str == "I am"

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


class TestInferenceClientModel:
    def test_call_with_custom_role_conversions(self):
        custom_role_conversions = {MessageRole.USER: MessageRole.SYSTEM}
        model = InferenceClientModel(model_id="test-model", custom_role_conversions=custom_role_conversions)
        model.client = MagicMock()
        mock_response = model.client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(role="assistant")
        messages = [{"role": "user", "content": "Test message"}]
        _ = model(messages)
        # Verify that the role conversion was applied
        assert model.client.chat_completion.call_args.kwargs["messages"][0]["role"] == "system", (
            "role conversion should be applied"
        )

    def test_init_model_with_tokens(self):
        model = InferenceClientModel(model_id="test-model", token="abc")
        assert model.client.token == "abc"

        model = InferenceClientModel(model_id="test-model", api_key="abc")
        assert model.client.token == "abc"

        with pytest.raises(ValueError, match="Received both `token` and `api_key` arguments."):
            InferenceClientModel(model_id="test-model", token="abc", api_key="def")

    @require_run_all
    def test_get_hfapi_message_no_tool(self):
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_no_tool_external_provider(self):
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])


class TestHfApiModel:
    def test_init_model_with_tokens(self):
        model = HfApiModel(model_id="test-model", token="abc")
        assert model.client.token == "abc"

        model = HfApiModel(model_id="test-model", api_key="abc")
        assert model.client.token == "abc"

        with pytest.raises(ValueError) as e:
            _ = HfApiModel(model_id="test-model", token="abc", api_key="def")
        assert "Received both `token` and `api_key` arguments." in str(e)

    @require_run_all
    def test_get_hfapi_message_no_tool(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model.generate(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_no_tool_external_provider(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model.generate(messages, stop_sequences=["great"])

    @require_run_all
    def test_get_hfapi_message_stream_no_tool(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        for el in model.generate_stream(messages, stop_sequences=["great"]):
            assert el.content is not None

    @require_run_all
    def test_get_hfapi_message_stream_no_tool_external_provider(self):
        model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        for el in model.generate_stream(messages, stop_sequences=["great"]):
            assert el.content is not None


class TestLiteLLMModel:
    @pytest.mark.parametrize(
        "model_id, error_flag",
        [
            ("groq/llama-3.3-70b", "Invalid API Key"),
            ("cerebras/llama-3.3-70b", "The api_key client option must be set"),
            ("mistral/mistral-tiny", "The api_key client option must be set"),
        ],
    )
    def test_call_different_providers_without_key(self, model_id, error_flag):
        model = LiteLLMModel(model_id=model_id)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Test message"}]}]
        with pytest.raises(Exception) as e:
            # This should raise 401 error because of missing API key, not fail for any "bad format" reason
            model.generate(messages)
        assert error_flag in str(e)
        with pytest.raises(Exception) as e:
            # This should raise 401 error because of missing API key, not fail for any "bad format" reason
            for el in model.generate_stream(messages):
                assert el.content is not None
        assert error_flag in str(e)

    def test_passing_flatten_messages(self):
        model = LiteLLMModel(model_id="groq/llama-3.3-70b", flatten_messages_as_text=False)
        assert not model.flatten_messages_as_text

        model = LiteLLMModel(model_id="fal/llama-3.3-70b", flatten_messages_as_text=True)
        assert model.flatten_messages_as_text


class TestLiteLLMRouterModel:
    @pytest.mark.parametrize(
        "model_id, expected",
        [
            ("llama-3.3-70b", False),
            ("llama-3.3-70b", True),
            ("mistral-tiny", True),
        ],
    )
    def test_flatten_messages_as_text(self, model_id, expected):
        model_list = [
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "groq/llama-3.3-70b"}},
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "cerebras/llama-3.3-70b"}},
            {"model_name": "mistral-tiny", "litellm_params": {"model": "mistral/mistral-tiny"}},
        ]
        model = LiteLLMRouterModel(model_id=model_id, model_list=model_list, flatten_messages_as_text=expected)
        assert model.flatten_messages_as_text is expected

    def test_create_client(self):
        model_list = [
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "groq/llama-3.3-70b"}},
            {"model_name": "llama-3.3-70b", "litellm_params": {"model": "cerebras/llama-3.3-70b"}},
        ]
        with patch("litellm.Router") as mock_router:
            router_model = LiteLLMRouterModel(
                model_id="model-group-1", model_list=model_list, client_kwargs={"routing_strategy": "simple-shuffle"}
            )
            # Ensure that the Router constructor was called with the expected keyword arguments
            mock_router.assert_called_once()
            assert mock_router.call_count == 1
            assert mock_router.call_args.kwargs["model_list"] == model_list
            assert mock_router.call_args.kwargs["routing_strategy"] == "simple-shuffle"
            assert router_model.client == mock_router.return_value


class TestOpenAIServerModel:
    def test_client_kwargs_passed_correctly(self):
        model_id = "gpt-3.5-turbo"
        api_base = "https://api.openai.com/v1"
        api_key = "test_api_key"
        organization = "test_org"
        project = "test_project"
        client_kwargs = {"max_retries": 5}

        with patch("openai.OpenAI") as MockOpenAI:
            model = OpenAIServerModel(
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
        assert model.client == MockOpenAI.return_value


class TestAmazonBedrockServerModel:
    def test_client_for_bedrock(self):
        model_id = "us.amazon.nova-pro-v1:0"

        with patch("boto3.client") as MockBoto3:
            model = AmazonBedrockServerModel(
                model_id=model_id,
            )

        assert model.client == MockBoto3.return_value


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
            model = AzureOpenAIServerModel(
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
        assert model.client == MockAzureOpenAI.return_value


class TestTransformersModel:
    @pytest.mark.parametrize(
        "patching",
        [
            [
                (
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    {"side_effect": ValueError("Unrecognized configuration class")},
                ),
                ("transformers.AutoModelForCausalLM.from_pretrained", {}),
                ("transformers.AutoTokenizer.from_pretrained", {}),
            ],
            [
                ("transformers.AutoModelForImageTextToText.from_pretrained", {}),
                ("transformers.AutoProcessor.from_pretrained", {}),
            ],
        ],
    )
    def test_init(self, patching):
        with ExitStack() as stack:
            mocks = {target: stack.enter_context(patch(target, **kwargs)) for target, kwargs in patching}
            model = TransformersModel(
                model_id="test-model", device_map="cpu", torch_dtype="float16", trust_remote_code=True
            )
        assert model.model_id == "test-model"
        if "transformers.AutoTokenizer.from_pretrained" in mocks:
            assert model.model == mocks["transformers.AutoModelForCausalLM.from_pretrained"].return_value
            assert mocks["transformers.AutoModelForCausalLM.from_pretrained"].call_args.kwargs == {
                "device_map": "cpu",
                "torch_dtype": "float16",
                "trust_remote_code": True,
            }
            assert model.tokenizer == mocks["transformers.AutoTokenizer.from_pretrained"].return_value
            assert mocks["transformers.AutoTokenizer.from_pretrained"].call_args.args == ("test-model",)
            assert mocks["transformers.AutoTokenizer.from_pretrained"].call_args.kwargs == {"trust_remote_code": True}
        elif "transformers.AutoProcessor.from_pretrained" in mocks:
            assert model.model == mocks["transformers.AutoModelForImageTextToText.from_pretrained"].return_value
            assert mocks["transformers.AutoModelForImageTextToText.from_pretrained"].call_args.kwargs == {
                "device_map": "cpu",
                "torch_dtype": "float16",
                "trust_remote_code": True,
            }
            assert model.processor == mocks["transformers.AutoProcessor.from_pretrained"].return_value
            assert mocks["transformers.AutoProcessor.from_pretrained"].call_args.args == ("test-model",)
            assert mocks["transformers.AutoProcessor.from_pretrained"].call_args.kwargs == {"trust_remote_code": True}


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
    assert result[0]["content"] == "Hello!\nHow are you?"


@pytest.mark.parametrize(
    "model_class, model_kwargs, patching, expected_flatten_messages_as_text",
    [
        (AzureOpenAIServerModel, {}, ("openai.AzureOpenAI", {}), False),
        (InferenceClientModel, {}, ("huggingface_hub.InferenceClient", {}), False),
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
                (
                    "transformers.AutoModelForImageTextToText.from_pretrained",
                    {"side_effect": ValueError("Unrecognized configuration class")},
                ),
                ("transformers.AutoModelForCausalLM.from_pretrained", {}),
                ("transformers.AutoTokenizer.from_pretrained", {}),
            ],
            True,
        ),
        (
            TransformersModel,
            {},
            [
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


@pytest.mark.parametrize(
    "model_id,expected",
    [
        # Unsupported base models
        ("o3", False),
        ("o4-mini", False),
        # Unsupported versioned models
        ("o3-2025-04-16", False),
        ("o4-mini-2025-04-16", False),
        # Unsupported models with path prefixes
        ("openai/o3", False),
        ("openai/o4-mini", False),
        ("openai/o3-2025-04-16", False),
        ("openai/o4-mini-2025-04-16", False),
        # Supported models
        ("o3-mini", True),  # Different from o3
        ("o3-mini-2025-01-31", True),  # Different from o3
        ("o4", True),  # Different from o4-mini
        ("o4-turbo", True),  # Different from o4-mini
        ("gpt-4", True),
        ("claude-3-5-sonnet", True),
        ("mistral-large", True),
        # Supported models with path prefixes
        ("openai/gpt-4", True),
        ("anthropic/claude-3-5-sonnet", True),
        ("mistralai/mistral-large", True),
        # Edge cases
        ("", True),  # Empty string doesn't match pattern
        ("o3x", True),  # Not exactly o3
        ("o3_mini", True),  # Not o3-mini format
        ("prefix-o3", True),  # o3 not at start
    ],
)
def test_supports_stop_parameter(model_id, expected):
    """Test the supports_stop_parameter function with various model IDs"""
    assert supports_stop_parameter(model_id) == expected, f"Failed for model_id: {model_id}"


class TestGetToolCallFromText:
    @pytest.fixture(autouse=True)
    def mock_uuid4(self):
        with patch("uuid.uuid4", return_value="test-uuid"):
            yield

    def test_get_tool_call_from_text_basic(self):
        text = '{"name": "weather_tool", "arguments": "New York"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert isinstance(result, ChatMessageToolCall)
        assert result.id == "test-uuid"
        assert result.type == "function"
        assert result.function.name == "weather_tool"
        assert result.function.arguments == "New York"

    def test_get_tool_call_from_text_name_key_missing(self):
        text = '{"action": "weather_tool", "arguments": "New York"}'
        with pytest.raises(ValueError) as exc_info:
            get_tool_call_from_text(text, "name", "arguments")
        error_msg = str(exc_info.value)
        assert "Key tool_name_key='name' not found" in error_msg
        assert "'action', 'arguments'" in error_msg

    def test_get_tool_call_from_text_json_object_args(self):
        text = '{"name": "weather_tool", "arguments": {"city": "New York"}}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments == {"city": "New York"}

    def test_get_tool_call_from_text_json_string_args(self):
        text = '{"name": "weather_tool", "arguments": "{\\"city\\": \\"New York\\"}"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments == {"city": "New York"}

    def test_get_tool_call_from_text_missing_args(self):
        text = '{"name": "weather_tool"}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.arguments is None

    def test_get_tool_call_from_text_custom_keys(self):
        text = '{"tool": "weather_tool", "params": "New York"}'
        result = get_tool_call_from_text(text, "tool", "params")
        assert result.function.name == "weather_tool"
        assert result.function.arguments == "New York"

    def test_get_tool_call_from_text_numeric_args(self):
        text = '{"name": "calculator", "arguments": 42}'
        result = get_tool_call_from_text(text, "name", "arguments")
        assert result.function.name == "calculator"
        assert result.function.arguments == 42
