# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from .monitoring import TokenUsage
from .tools import Tool
from .utils import _is_package_available, encode_image_base64, make_image_url, parse_json_blob


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]
CODEAGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "additionalProperties": False,
            "properties": {
                "thought": {
                    "description": "A free form text description of the thought process.",
                    "title": "Thought",
                    "type": "string",
                },
                "code": {
                    "description": "Valid Python code snippet implementing the thought.",
                    "title": "Code",
                    "type": "string",
                },
            },
            "required": ["thought", "code"],
            "title": "ThoughtAndCodeAnswer",
            "type": "object",
        },
        "name": "ThoughtAndCodeAnswer",
        "strict": True,
    },
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API
    token_usage: TokenUsage | None = None

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        return get_dict_from_nested_dataclasses(self)


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageStreamDelta:
    content: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    token_usage: TokenUsage | None = None


@dataclass
class ToolCallStreamDelta:
    """Represents a streaming delta for tool calls during generation."""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: dict[str, Any] | None = None


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: list[dict[str, str | list[dict]]],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, str | list[dict]]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list: list[dict[str, str | list[dict]]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message["content"][0]["text"]
            else:
                for el in message["content"]:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Key {tool_name_key=} not found in the generated tool call. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3 and openai/o4-mini (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    # o3 and o4-mini (including versioned variants, o3-2025-04-16) don't support stop parameter
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*)$"
    return not re.match(pattern, model_name)


class Model:
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self._last_input_token_count: int | None = None
        self._last_output_token_count: int | None = None
        self.model_id: str | None = model_id

    @property
    def last_input_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_input_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.input_tokens instead.",
            FutureWarning,
        )
        return self._last_input_token_count

    @property
    def last_output_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_output_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.output_tokens instead.",
            FutureWarning,
        )
        return self._last_output_token_count

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",  # Configurable tool_choice parameter
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, response_format, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            # Some models do not support stop parameter
            if supports_stop_parameter(self.model_id or ""):
                completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # Handle tools parameter
        if tools_to_call_from:
            tools_config = {
                "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`list[dict[str, str | list[dict]]] | list[ChatMessage]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            response_format (`dict[str, str]`, *optional*):
                The response format to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Sometimes APIs do not return the tool call as a specific object, so we need to parse it."""
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        return cls(**{k: v for k, v in model_dictionary.items()})


class VLLMModel(Model):
    """Model to use [vLLM](https://docs.vllm.ai/) for fast LLM inference and serving.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference.
            This can be a path or model identifier from the Hugging Face model hub.
        model_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the vLLM model (like revision, max_model_len, etc.).
    """

    def __init__(
        self,
        model_id,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not _is_package_available("vllm"):
            raise ModuleNotFoundError("Please install 'vllm' extra to use VLLMModel: `pip install 'smolagents[vllm]'`")

        from vllm import LLM  # type: ignore
        from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

        self.model_kwargs = model_kwargs or {}
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model = LLM(model=model_id, **self.model_kwargs)
        assert self.model is not None
        self.tokenizer = get_tokenizer(model_id)
        self._is_vlm = False  # VLLMModel does not support vision models yet.

    def cleanup(self):
        import gc

        import torch
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        if self.model is not None:
            # taken from https://github.com/vllm-project/vllm/issues/1908#issuecomment-2076870351
            del self.model.llm_engine.model_executor.driver_worker
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        from vllm import SamplingParams  # type: ignore

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            flatten_messages_as_text=(not self._is_vlm),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # Override the OpenAI schema for VLLM compatibility
        guided_options_request = {"guided_json": response_format["json_schema"]["schema"]} if response_format else None

        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )

        sampling_params = SamplingParams(
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
            stop=prepared_stop_sequences,
        )

        out = self.model.generate(
            prompt,
            sampling_params=sampling_params,
            guided_options_request=guided_options_request,
        )

        output_text = out[0].outputs[0].text
        self._last_input_token_count = len(out[0].prompt_token_ids)
        self._last_output_token_count = len(out[0].outputs[0].token_ids)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(out[0].prompt_token_ids),
                output_tokens=len(out[0].outputs[0].token_ids),
            ),
        )


class MLXModel(Model):
    """A class to interact with models loaded using MLX on Apple silicon.

    > [!TIP]
    > You must have `mlx-lm` installed on your machine. Please run `pip install smolagents[mlx-lm]` if it's not the case.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        tool_name_key (str):
            The key, which can usually be found in the model's chat template, for retrieving a tool name.
        tool_arguments_key (str):
            The key, which can usually be found in the model's chat template, for retrieving tool arguments.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        load_kwargs (dict[str, Any], *optional*):
            Additional keyword arguments to pass to the `mlx.lm.load` method when loading the model and tokenizer.
        apply_chat_template_kwargs (dict, *optional*):
            Additional keyword arguments to pass to the `apply_chat_template` method of the tokenizer.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "text", "text": "Explain quantum mechanics in simple terms."}
    ...         ]
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        trust_remote_code: bool = False,
        load_kwargs: dict[str, Any] | None = None,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'smolagents[mlx-lm]'`"
            )
        import mlx_lm

        self.load_kwargs = load_kwargs or {}
        self.load_kwargs.setdefault("tokenizer_config", {}).setdefault("trust_remote_code", trust_remote_code)
        self.apply_chat_template_kwargs = apply_chat_template_kwargs or {}
        self.apply_chat_template_kwargs.setdefault("add_generation_prompt", True)
        # mlx-lm doesn't support vision models: flatten_messages_as_text=True
        super().__init__(model_id=model_id, flatten_messages_as_text=True, **kwargs)

        self.model, self.tokenizer = mlx_lm.load(self.model_id, **self.load_kwargs)
        self.stream_generate = mlx_lm.stream_generate
        self.is_vlm = False  # mlx-lm doesn't support vision models

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("MLX does not support structured outputs.")
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        messages = completion_kwargs.pop("messages")
        stops = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt_ids = self.tokenizer.apply_chat_template(messages, tools=tools, **self.apply_chat_template_kwargs)

        output_tokens = 0
        text = ""
        for response in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            output_tokens += 1
            text += response.text
            if any((stop_index := text.rfind(stop)) != -1 for stop in stops):
                text = text[:stop_index]
                break

        self._last_input_token_count = len(prompt_ids)
        self._last_output_token_count = output_tokens
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw={"out": text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(prompt_ids),
                output_tokens=output_tokens,
            ),
        )


class TransformersModel(Model):
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
            For example, `"Qwen/Qwen2.5-Coder-32B-Instruct"`.
        device_map (`str`, *optional*):
            The device_map to initialize your model with.
        torch_dtype (`str`, *optional*):
            The torch_dtype to initialize your model with.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_new_tokens` or `device`.
        **kwargs:
            Additional keyword arguments to pass to `model.generate()`, for instance `max_new_tokens` or `device`.
    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )

        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                FutureWarning,
            )
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        default_max_tokens = 4096
        max_new_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        if not max_new_tokens:
            kwargs["max_new_tokens"] = default_max_tokens
            logger.warning(
                f"`max_new_tokens` not provided, using this default value for `max_new_tokens`: {default_max_tokens}"
            )

        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self._is_vlm = True
            self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore

        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            else:
                raise e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(flatten_messages_as_text=not self._is_vlm, model_id=model_id, **kwargs)

    def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: list[str], tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def _prepare_completion_args(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)
        tools = completion_kwargs.pop("tools", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,
            tools=tools,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
        )
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_stop_sequences(output_text, stop_sequences)

        self._last_input_token_count = count_prompt_tokens
        self._last_output_token_count = len(generated_tokens)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

        thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
        thread.start()

        # Generate with streaming
        for new_text in self.streamer:
            self._last_input_token_count = count_prompt_tokens
            self._last_output_token_count = 1
            yield ChatMessageStreamDelta(
                content=new_text,
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=count_prompt_tokens, output_tokens=1),
            )
        thread.join()


class ApiModel(Model):
    """
    Base class for API-based language models.

    This class serves as a foundation for implementing models that interact with
    external APIs. It handles the common functionality for managing model IDs,
    custom role mappings, and API client connections.

    Parameters:
        model_id (`str`):
            The identifier for the model to be used with the API.
        custom_role_conversions (`dict[str, str`], **optional**):
            Mapping to convert  between internal role names and API-specific role names. Defaults to None.
        client (`Any`, **optional**):
            Pre-configured API client instance. If not provided, a default client will be created. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
        self, model_id: str, custom_role_conversions: dict[str, str] | None = None, client: Any | None = None, **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()

    def create_client(self):
        """Create the API client for the specific service."""
        raise NotImplementedError("Subclasses must implement this method to create a client")


class LiteLLMModel(ApiModel):
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """Create the LiteLLM client."""
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        response = self.client.completion(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        if tools_to_call_from:
            raise NotImplementedError("Streaming is not yet supported for tool calling")
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        for event in self.client.completion(**completion_kwargs, stream=True, stream_options={"include_usage": True}):
            if event.choices:
                if event.choices[0].delta.content:
                    yield ChatMessageStreamDelta(
                        content=event.choices[0].delta.content,
                    )
            if getattr(event, "usage", None):
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )


class LiteLLMRouterModel(LiteLLMModel):
    """Router‑based client for interacting with the [LiteLLM Python SDK Router](https://docs.litellm.ai/docs/routing).

    This class provides a high-level interface for distributing requests among multiple language models using
    the LiteLLM SDK's routing capabilities. It is responsible for initializing and configuring the router client,
    applying custom role conversions, and managing message formatting to ensure seamless integration with various LLMs.

    Parameters:
        model_id (`str`):
            Identifier for the model group to use from the model list (e.g., "model-group-1").
        model_list (`list[dict[str, Any]]`):
            Model configurations to be used for routing.
            Each configuration should include the model group name and any necessary parameters.
            For more details, refer to the [LiteLLM Routing](https://docs.litellm.ai/docs/routing#quick-start) documentation.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional configuration parameters for the Router client. For more details, see the
            [LiteLLM Routing Configurations](https://docs.litellm.ai/docs/routing).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the LiteLLM Router completion method.

    Example:
    ```python
    >>> import os
    >>> from smolagents import CodeAgent, WebSearchTool, LiteLLMRouterModel
    >>> os.environ["OPENAI_API_KEY"] = ""
    >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    >>> os.environ["AWS_REGION"] = ""
    >>> llm_loadbalancer_model_list = [
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "gpt-4o-mini",
    ...             "api_key": os.getenv("OPENAI_API_KEY"),
    ...         },
    ...     },
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    ...             "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    ...             "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    ...             "aws_region_name": os.getenv("AWS_REGION"),
    ...         },
    ...     },
    >>> ]
    >>> model = LiteLLMRouterModel(
    ...    model_id="model-group-1",
    ...    model_list=llm_loadbalancer_model_list,
    ...    client_kwargs={
    ...        "routing_strategy":"simple-shuffle"
    ...    }
    >>> )
    >>> agent = CodeAgent(tools=[WebSearchTool()], model=model)
    >>> agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    ```
    """

    def __init__(
        self,
        model_id: str,
        model_list: list[dict[str, Any]],
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        self.client_kwargs = {
            "model_list": model_list,
            **(client_kwargs or {}),
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            from litellm import Router
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMRouterModel: `pip install 'smolagents[litellm]'`"
            ) from e
        return Router(**self.client_kwargs)


class InferenceClientModel(ApiModel):
    """A class to interact with Hugging Face's Inference Providers for language model interaction.

    This model allows you to communicate with Hugging Face's models using Inference Providers. It can be used in both serverless mode, with a dedicated endpoint, or even with a local URL, supporting features like stop sequences and grammar customization.

    Providers include Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.

    Parameters:
        model_id (`str`, *optional*, default `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference.
            This can be a model identifier from the Hugging Face model hub or a URL to a deployed Inference Endpoint.
            Currently, it defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`, but this may change in the future.
        provider (`str`, *optional*):
            Name of the provider to use for inference. A list of supported providers can be found in the [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
            Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order [here](https://hf.co/settings/inference-providers).
            If `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference Providers'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the Hugging Face InferenceClient.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        api_key (`str`, *optional*):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
        bill_to (`str`, *optional*):
            The billing account to use for the requests. By default the requests are billed on the user's account. Requests can only be billed to
            an organization the user is a member of, and which has subscribed to Enterprise Hub.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the Hugging Face InferenceClient.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = InferenceClientModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     provider="nebius",
    ...     token="your_hf_token_here",
    ...     max_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        provider: str | None = None,
        token: str | None = None,
        timeout: int = 120,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        api_key: str | None = None,
        bill_to: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        if token is None:
            token = os.getenv("HF_TOKEN")
        self.client_kwargs = {
            **(client_kwargs or {}),
            "model": model_id,
            "provider": provider,
            "token": token,
            "timeout": timeout,
            "bill_to": bill_to,
            "base_url": base_url,
        }
        super().__init__(model_id=model_id, custom_role_conversions=custom_role_conversions, **kwargs)

    def create_client(self):
        """Create the Hugging Face client."""
        from huggingface_hub import InferenceClient

        return InferenceClient(**self.client_kwargs)

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None and self.client_kwargs["provider"] not in STRUCTURED_GENERATION_PROVIDERS:
            raise ValueError(
                "InferenceClientModel only supports structured outputs with these providers:"
                + ", ".join(STRUCTURED_GENERATION_PROVIDERS)
            )
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            # response_format=response_format,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        response = self.client.chat_completion(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            asdict(response.choices[0].message),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Track accumulated tool calls and content
        accumulated_tool_calls = {}
        current_content = ""

        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.choices:
                choice = event.choices[0]
                if choice.delta is None:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
                else:
                    delta = choice.delta

                    # Handle content streaming
                    if delta.content:
                        current_content += delta.content
                        yield ChatMessageStreamDelta(content=delta.content)

                    # Handle tool call streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:  # ?ormally there should be only one call at a time
                            # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls[tool_call_delta.index] = {
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": ""},
                                }

                            # Update the tool call at the specific index
                            tool_call = accumulated_tool_calls[tool_call_delta.index]

                            if tool_call_delta.id:
                                tool_call["id"] = tool_call_delta.index
                            if tool_call_delta.type:
                                tool_call["type"] = tool_call_delta.type
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                        yield ChatMessageStreamDelta(
                            content=current_content,
                            tool_calls=[
                                ToolCallStreamDelta(
                                    id=tool_call["id"],
                                    type=tool_call["type"],
                                    function=ChatMessageToolCallDefinition(
                                        name=tool_call["function"]["name"],
                                        arguments=tool_call["function"]["arguments"],
                                    ),
                                )
                                for tool_call in accumulated_tool_calls.values()
                            ],
                        )

            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )


class HfApiModel(InferenceClientModel):
    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "HfApiModel was renamed to InferenceClientModel in version 1.14.0 and will be removed in 1.17.0.",
            FutureWarning,
        )
        return super().__new__(cls)


class OpenAIServerModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Track accumulated tool calls and content
        accumulated_tool_calls = {}
        current_content = ""

        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.choices:
                choice = event.choices[0]
                if choice.delta is None:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
                else:
                    delta = choice.delta

                    # Handle content streaming
                    if delta.content:
                        current_content += delta.content
                        yield ChatMessageStreamDelta(content=delta.content)

                    # Handle tool call streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:  # ?ormally there should be only one call at a time
                            # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls[tool_call_delta.index] = {
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": ""},
                                }

                            # Update the tool call at the specific index
                            tool_call = accumulated_tool_calls[tool_call_delta.index]

                            if tool_call_delta.id:
                                tool_call["id"] = tool_call_delta.index
                            if tool_call_delta.type:
                                tool_call["type"] = tool_call_delta.type
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                        yield ChatMessageStreamDelta(
                            content=current_content,
                            tool_calls=[
                                ToolCallStreamDelta(
                                    id=tool_call["id"],
                                    type=tool_call["type"],
                                    function=ChatMessageToolCallDefinition(
                                        name=tool_call["function"]["name"],
                                        arguments=tool_call["function"]["arguments"],
                                    ),
                                )
                                for tool_call in accumulated_tool_calls.values()
                            ],
                        )

            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


class AzureOpenAIServerModel(OpenAIServerModel):
    """This model connects to an Azure OpenAI deployment.

    Parameters:
        model_id (`str`):
            The model deployment name to use when connecting (e.g. "gpt-4o-mini").
        azure_endpoint (`str`, *optional*):
            The Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`. If not provided, it will be inferred from the `AZURE_OPENAI_ENDPOINT` environment variable.
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, it will be inferred from the `AZURE_OPENAI_API_KEY` environment variable.
        api_version (`str`, *optional*):
            The API version to use. If not provided, it will be inferred from the `OPENAI_API_VERSION` environment variable.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the AzureOpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Azure OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
            }
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AzureOpenAI(**self.client_kwargs)


class AmazonBedrockServerModel(ApiModel):
    """
    A model class for interacting with Amazon Bedrock Server models through the Bedrock API.

    This class provides an interface to interact with various Bedrock language models,
    allowing for customized model inference, guardrail configuration, message handling,
    and other parameters allowed by boto3 API.

    Parameters:
        model_id (`str`):
            The model identifier to use on Bedrock (e.g. "us.amazon.nova-pro-v1:0").
        client (`boto3.client`, *optional*):
            A custom boto3 client for AWS interactions. If not provided, a default client will be created.
        client_kwargs (dict[str, Any], *optional*):
            Keyword arguments used to configure the boto3 client if it needs to be created internally.
            Examples include `region_name`, `config`, or `endpoint_url`.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
            Defaults to converting all roles to "user" role to enable using all the Bedrock models.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs
            Additional keyword arguments passed directly to the underlying API calls.

    Example:
        Creating a model instance with default settings:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0'
        ... )

        Creating a model instance with a custom boto3 client:
        >>> import boto3
        >>> client = boto3.client('bedrock-runtime', region_name='us-west-2')
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client=client
        ... )

        Creating a model instance with client_kwargs for internal client creation:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client_kwargs={'region_name': 'us-west-2', 'endpoint_url': 'https://custom-endpoint.com'}
        ... )

        Creating a model instance with inference and guardrail configurations:
        >>> additional_api_config = {
        ...     "inferenceConfig": {
        ...         "maxTokens": 3000
        ...     },
        ...     "guardrailConfig": {
        ...         "guardrailIdentifier": "identify1",
        ...         "guardrailVersion": 'v1'
        ...     },
        ... }
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='anthropic.claude-3-haiku-20240307-v1:0',
        ...     **additional_api_config
        ... )
    """

    def __init__(
        self,
        model_id: str,
        client=None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        self.client_kwargs = client_kwargs or {}

        # Bedrock only supports `assistant` and `user` roles.
        # Many Bedrock models do not allow conversations to start with the `assistant` role, so the default is set to `user/user`.
        # This parameter is retained for future model implementations and extended support.
        custom_role_conversions = custom_role_conversions or {
            MessageRole.SYSTEM: MessageRole.USER,
            MessageRole.ASSISTANT: MessageRole.USER,
            MessageRole.TOOL_CALL: MessageRole.USER,
            MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API doesn't support flatten messages, must be a list of messages
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> dict:
        """
        Overrides the base method to handle Bedrock-specific configurations.

        This implementation adapts the completion keyword arguments to align with
        Bedrock's requirements, ensuring compatibility with its unique setup and
        constraints.
        """
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Bedrock support stop_sequence using Inference Config
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            **kwargs,
        )

        # Not all models in Bedrock support `toolConfig`. Also, smolagents already include the tool call in the prompt,
        # so adding `toolConfig` could cause conflicts. We remove it to avoid issues.
        completion_kwargs.pop("toolConfig", None)

        # The Bedrock API does not support the `type` key in requests.
        # This block of code modifies the object to meet Bedrock's requirements.
        for message in completion_kwargs.get("messages", []):
            for content in message.get("content", []):
                if "type" in content:
                    del content["type"]

        return {
            "modelId": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        try:
            import boto3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'bedrock' extra to use AmazonBedrockServerModel: `pip install 'smolagents[bedrock]'`"
            ) from e

        return boto3.client("bedrock-runtime", **self.client_kwargs)

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # self.client is created in ApiModel class
        response = self.client.converse(**completion_kwargs)

        # Get first message
        response["output"]["message"]["content"] = response["output"]["message"]["content"][0]["text"]

        self._last_input_token_count = response["usage"]["inputTokens"]
        self._last_output_token_count = response["usage"]["outputTokens"]
        return ChatMessage.from_dict(
            response["output"]["message"],
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response["usage"]["inputTokens"],
                output_tokens=response["usage"]["outputTokens"],
            ),
        )


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXModel",
    "TransformersModel",
    "ApiModel",
    "InferenceClientModel",
    "HfApiModel",
    "LiteLLMModel",
    "LiteLLMRouterModel",
    "OpenAIServerModel",
    "VLLMModel",
    "AzureOpenAIServerModel",
    "AmazonBedrockServerModel",
    "ChatMessage",
]
