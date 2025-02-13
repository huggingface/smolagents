#!/usr/bin/env python
# coding=utf-8

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
import random
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from huggingface_hub import InferenceClient
from huggingface_hub.utils import is_torch_available
from PIL import Image

from .tools import Tool
from .utils import _is_package_available, encode_image_base64, make_image_url


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList

logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
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
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls]
        return cls(role=message.role, content=message.content, tool_calls=tool_calls, raw=raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    for tool_call in message.tool_calls:
        tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


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


def get_tool_json_schema(tool: Tool) -> Dict:
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


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
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
                output_message_list[-1]["content"] += message["content"][0]["text"]
            else:
                output_message_list[-1]["content"] += message["content"]
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


class Model:
    def __init__(self, **kwargs):
        self.last_input_token_count = None
        self.last_output_token_count = None
        self.kwargs = kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, grammar, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
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
            completion_kwargs["stop"] = stop_sequences
        if grammar is not None:
            completion_kwargs["grammar"] = grammar

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "required",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!


class HfApiModel(Model):
    """A class to interact with Hugging Face's Inference API for language model interaction.

    This model allows you to communicate with Hugging Face's models using the Inference API. It can be used in both serverless mode or with a dedicated endpoint, supporting features like stop sequences and grammar customization.

    Parameters:
        model_id (`str`, *optional*, defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        provider (`str`, *optional*):
            Name of the provider to use for inference. Can be `"replicate"`, `"together"`, `"fal-ai"`, `"sambanova"` or `"hf-inference"`.
            defaults to hf-inference (HF Inference API).
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference API'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Hugging Face API.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = HfApiModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
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
        provider: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = 120,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.provider = provider
        if token is None:
            token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(self.model_id, provider=provider, token=token, timeout=timeout)
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        response = self.client.chat_completion(**completion_kwargs)

        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        message = ChatMessage.from_hf_api(response.choices[0].message, raw=response)
        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message


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
        trust_remote_code (bool):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
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
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'smolagents[mlx-lm]'`"
            )
        import mlx_lm

        self.model_id = model_id
        self.model, self.tokenizer = mlx_lm.load(model_id, tokenizer_config={"trust_remote_code": trust_remote_code})
        self.stream_generate = mlx_lm.stream_generate
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key

    def _to_message(self, text, tools_to_call_from):
        if tools_to_call_from:
            # tmp solution for extracting tool JSON without assuming a specific model output format
            maybe_json = "{" + text.split("{", 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
            parsed_text = json.loads(maybe_json)
            tool_name = parsed_text.get(self.tool_name_key, None)
            tool_arguments = parsed_text.get(self.tool_arguments_key, None)
            if tool_name:
                return ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatMessageToolCall(
                            id=uuid.uuid4(),
                            type="function",
                            function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                        )
                    ],
                )
        return ChatMessage(role="assistant", content=text)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            flatten_messages_as_text=True,  # mlx-lm doesn't support vision models
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
        )

        self.last_input_token_count = len(prompt_ids)
        self.last_output_token_count = 0
        text = ""

        for _ in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            self.last_output_token_count += 1
            text += _.text
            for stop_sequence in prepared_stop_sequences:
                if text.strip().endswith(stop_sequence):
                    text = text[: -len(stop_sequence)]
                    return self._to_message(text, tools_to_call_from)

        return self._to_message(text, tools_to_call_from)


class TransformersModel(Model):
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

    Parameters:
        model_id (`str`, *optional*, defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
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
        model_id: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not is_torch_available() or not _is_package_available("transformers"):
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        default_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        if model_id is None:
            model_id = default_model_id
            logger.warning(f"`model_id`not provided, using this default tokenizer for token counts: '{model_id}'")
        self.model_id = model_id
        self.kwargs = kwargs
        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, device_map=device_map)
                self.processor = AutoProcessor.from_pretrained(model_id)
                self._is_vlm = True
            else:
                raise e
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer and model for {model_id=}: {e}. Loading default tokenizer and model instead from {default_model_id=}."
            )
            self.model_id = default_model_id
            self.tokenizer = AutoTokenizer.from_pretrained(default_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch_dtype)

    def make_stopping_criteria(self, stop_sequences: List[str], tokenizer) -> "StoppingCriteriaList":
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: List[str], tokenizer):
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

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            flatten_messages_as_text=(not self._is_vlm),
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
        )

        if max_new_tokens:
            completion_kwargs["max_new_tokens"] = max_new_tokens

        if hasattr(self, "processor"):
            images = [Image.open(image) for image in images] if images else None
            prompt_tensor = self.processor.apply_chat_template(
                messages,
                tools=[get_tool_json_schema(tool) for tool in tools_to_call_from] if tools_to_call_from else None,
                return_tensors="pt",
                tokenize=True,
                return_dict=True,
                images=images,
                add_generation_prompt=True if tools_to_call_from else False,
            )
        else:
            prompt_tensor = self.tokenizer.apply_chat_template(
                messages,
                tools=[get_tool_json_schema(tool) for tool in tools_to_call_from] if tools_to_call_from else None,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True if tools_to_call_from else False,
            )

        prompt_tensor = prompt_tensor.to(self.model.device)
        count_prompt_tokens = prompt_tensor["input_ids"].shape[1]

        if stop_sequences:
            stopping_criteria = self.make_stopping_criteria(
                stop_sequences, tokenizer=self.processor if hasattr(self, "processor") else self.tokenizer
            )
        else:
            stopping_criteria = None

        out = self.model.generate(
            **prompt_tensor,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        self.last_input_token_count = count_prompt_tokens
        self.last_output_token_count = len(generated_tokens)

        if stop_sequences is not None:
            output = remove_stop_sequences(output, stop_sequences)

        if tools_to_call_from is None:
            return ChatMessage(
                role="assistant",
                content=output,
                raw={"out": out, "completion_kwargs": completion_kwargs},
            )
        else:
            if "Action:" in output:
                output = output.split("Action:", 1)[1].strip()
            try:
                start_index = output.index("{")
                end_index = output.rindex("}")
                output = output[start_index : end_index + 1]
            except Exception as e:
                raise Exception("No json blob found in output!") from e

            try:
                parsed_output = json.loads(output)
            except json.JSONDecodeError as e:
                raise ValueError(f"Tool call '{output}' has an invalid JSON structure: {e}")
            tool_name = parsed_output.get("name")
            tool_arguments = parsed_output.get("arguments")
            return ChatMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="".join(random.choices("0123456789", k=5)),
                        type="function",
                        function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                    )
                ],
                raw={"out": out, "completion_kwargs": completion_kwargs},
            )


class LiteLLMModel(Model):
    """This model connects to [LiteLLM](https://www.litellm.ai/) as a gateway to hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str = "anthropic/claude-3-5-sonnet-20240620",
        api_base=None,
        api_key=None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import litellm
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            )

        super().__init__(**kwargs)
        self.model_id = model_id
        # IMPORTANT - Set this to TRUE to add the function to the prompt for Non OpenAI LLMs
        litellm.add_function_to_prompt = True
        self.api_base = api_base
        self.api_key = api_key
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        import litellm

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            flatten_messages_as_text=self.model_id.startswith("ollama"),
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        response = litellm.completion(**completion_kwargs)

        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        )
        message.raw = response

        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message


class OpenAIServerModel(Model):
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
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
        )
        self.custom_role_conversions = custom_role_conversions

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens

        message = ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"})
        )
        message.raw = response
        if tools_to_call_from is not None:
            return parse_tool_args_if_needed(message)
        return message


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
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Azure OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        # read the api key manually, to avoid super().__init__() trying to use the wrong api_key (OPENAI_API_KEY)
        if api_key is None:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        super().__init__(model_id=model_id, api_key=api_key, custom_role_conversions=custom_role_conversions, **kwargs)
        # if we've reached this point, it means the openai package is available (checked in baseclass) so go ahead and import it
        import openai

        self.client = openai.AzureOpenAI(api_key=api_key, api_version=api_version, azure_endpoint=azure_endpoint)


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXModel",
    "TransformersModel",
    "HfApiModel",
    "LiteLLMModel",
    "OpenAIServerModel",
    "AzureOpenAIServerModel",
    "ChatMessage",
]
