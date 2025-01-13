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
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional

from huggingface_hub import (
    InferenceClient,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputFunctionDefinition,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    is_torch_available,
)
import openai

from .tools import Tool

logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_code>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_code>",
}

try:
    import litellm

    is_litellm_available = True
except ImportError:
    is_litellm_available = False


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


def get_json_schema(tool: Tool) -> Dict:
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
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`List[Dict[str, str]]`): List of chat messages.
    """
    final_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        # if not set(message.keys()) == {"role", "content"}:
        #     raise ValueError("Message should contain only 'role' and 'content' keys!")

        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(
                f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
            )

        if role in role_conversions:
            message["role"] = role_conversions[role]

        if (
            len(final_message_list) > 0
            and message["role"] == final_message_list[-1]["role"]
        ):
            final_message_list[-1]["content"] += "\n=======\n" + message["content"]
        else:
            final_message_list.append(message)
    return final_message_list


class Model:
    def __init__(self):
        self.last_input_token_count = None
        self.last_output_token_count = None

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
        max_tokens: int = 1500,
    ) -> str:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.
            max_tokens (`int`, *optional*):
                The maximum count of tokens to generate.
        Returns:
            `str`: The text content of the model's response.
        """
        if not isinstance(messages, List):
            raise ValueError(
                "Messages should be a list of dictionaries with 'role' and 'content' keys."
            )
        if stop_sequences is None:
            stop_sequences = []
        response = self.generate(messages, stop_sequences, grammar, max_tokens)

        return remove_stop_sequences(response, stop_sequences)


class HfApiModel(Model):
    """A class to interact with Hugging Face's Inference API for language model interaction.

    This engine allows you to communicate with Hugging Face's models using the Inference API. It can be used in both serverless mode or with a dedicated endpoint, supporting features like stop sequences and grammar customization.

    Parameters:
        model_id (`str`, *optional*, defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference API'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = HfApiModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     token="your_hf_token_here",
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"], max_tokens=1500)
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        token: Optional[str] = None,
        timeout: Optional[int] = 120,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.model_id = model_id
        if token is None:
            token = os.getenv("HF_TOKEN")
        self.client = InferenceClient(self.model_id, token=token, timeout=timeout)
        self.temperature = temperature

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
        tools_to_call_from: Optional[List[Tool]] = None,
    ) -> str:
        """
        Gets an LLM output message for the given list of input messages.
        If argument `tools_to_call_from` is passed, the model's tool calling options will be used to return a tool call.
        """
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        if tools_to_call_from:
            response = self.client.chat.completions.create(
                messages=messages,
                tools=[get_json_schema(tool) for tool in tools_to_call_from],
                tool_choice="auto",
                stop=stop_sequences,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stop=stop_sequences,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return response.choices[0].message


class TransformersModel(Model):
    """This engine initializes a model and tokenizer from the given `model_id`.

    Parameters:
        model_id (`str`, *optional*, defaults to `"HuggingFaceTB/SmolLM2-1.7B-Instruct"`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        device (`str`, optional, defaults to `"cuda"` if available, else `"cpu"`.):
            The device to load the model on (`"cpu"` or `"cuda"`).
    """

    def __init__(self, model_id: Optional[str] = None, device: Optional[str] = None):
        super().__init__()
        if not is_torch_available():
            raise ImportError("Please install torch in order to use TransformersModel.")
        import torch

        default_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        if model_id is None:
            model_id = default_model_id
            logger.warning(
                f"`model_id`not provided, using this default tokenizer for token counts: '{model_id}'"
            )
        self.model_id = model_id
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        logger.info(f"Using device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer and model for {model_id=}: {e}. Loading default tokenizer and model instead from {model_id=}."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(default_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(default_model_id).to(
                self.device
            )

    def make_stopping_criteria(self, stop_sequences: List[str]) -> StoppingCriteriaList:
        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: List[str], tokenizer):
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(
                    input_ids[0][-1], skip_special_tokens=True
                )
                self.stream += generated
                if any(
                    [
                        self.stream.endswith(stop_string)
                        for stop_string in self.stop_strings
                    ]
                ):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, self.tokenizer)])

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
        tools_to_call_from: Optional[List[Tool]] = None,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )

        if tools_to_call_from is not None:
            prompt_tensor = self.tokenizer.apply_chat_template(
                messages,
                tools=[get_json_schema(tool) for tool in tools_to_call_from],
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        else:
            prompt_tensor = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
            )
        prompt_tensor = prompt_tensor.to(self.model.device)
        count_prompt_tokens = prompt_tensor["input_ids"].shape[1]

        out = self.model.generate(
            **prompt_tensor,
            max_new_tokens=max_tokens,
            stopping_criteria=(
                self.make_stopping_criteria(stop_sequences) if stop_sequences else None
            ),
        )
        generated_tokens = out[0, count_prompt_tokens:]
        output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.last_input_token_count = count_prompt_tokens
        self.last_output_token_count = len(generated_tokens)

        if stop_sequences is not None:
            output = remove_stop_sequences(output, stop_sequences)

        if tools_to_call_from is None:
            return ChatCompletionOutputMessage(role="assistant", content=output)
        else:
            tool_name, tool_arguments = json.load(output)
            return ChatCompletionOutputMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionOutputToolCall(
                        id="".join(random.choices("0123456789", k=5)),
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            name=tool_name, arguments=tool_arguments
                        ),
                    )
                ],
            )


class LiteLLMModel(Model):
    def __init__(
        self,
        model_id="anthropic/claude-3-5-sonnet-20240620",
        api_base=None,
        api_key=None,
        **kwargs,
    ):
        if not is_litellm_available:
            raise ImportError(
                "litellm not found. Install it with `pip install litellm`"
            )
        super().__init__()
        self.model_id = model_id
        # IMPORTANT - Set this to TRUE to add the function to the prompt for Non OpenAI LLMs
        litellm.add_function_to_prompt = True
        self.api_base = api_base
        self.api_key = api_key
        self.kwargs = kwargs

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
        tools_to_call_from: Optional[List[Tool]] = None,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        if tools_to_call_from:
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                tools=[get_json_schema(tool) for tool in tools_to_call_from],
                tool_choice="required",
                stop=stop_sequences,
                max_tokens=max_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
                **self.kwargs,
            )
        else:
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                stop=stop_sequences,
                max_tokens=max_tokens,
                api_base=self.api_base,
                api_key=self.api_key,
                **self.kwargs,
            )
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return response.choices[0].message


class OpenAIServerModel(Model):
    """This engine connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`):
            The API key to use for authentication.
        temperature (`float`, *optional*, defaults to 0.7):
            Controls randomness in the model's responses. Values between 0 and 2.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: str,
        api_key: str,
        temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
        )
        self.temperature = temperature
        self.kwargs = kwargs

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
        tools_to_call_from: Optional[List[Tool]] = None,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        if tools_to_call_from:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                tools=[get_json_schema(tool) for tool in tools_to_call_from],
                tool_choice="auto",
                stop=stop_sequences,
                max_tokens=max_tokens,
                temperature=self.temperature,
                **self.kwargs,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                stop=stop_sequences,
                max_tokens=max_tokens,
                temperature=self.temperature,
                **self.kwargs,
            )
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return response.choices[0].message


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "TransformersModel",
    "HfApiModel",
    "LiteLLMModel",
    "OpenAIServerModel",
]
