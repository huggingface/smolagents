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
from copy import deepcopy
from enum import Enum
import json
from typing import Dict, List, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)

import litellm
import logging
import os
import random

from huggingface_hub import InferenceClient

from .tools import Tool
from .utils import parse_json_tool_call

logger = logging.getLogger(__name__)

DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_action>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_action>",
}


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

    def get_token_counts(self):
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
    ):
        raise NotImplementedError

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
        model_id (`str`, *optional*):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference API'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        base_url (`str`, *optional*):
            Base URL for a custom inference endpoint (TGI).
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.

     Raises:
        ValueError:
            - If neither model_id nor base_url is provided
            - If timeout is not a positive integer
            - If both model_id and base_url are provided simultaneously

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
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = 120,
        base_url: Optional[str] = None
    ):
        super().__init__()
        
        # Handle token
        self.token = token or os.getenv("HF_TOKEN")

         # Validate timeout
        if not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("Timeout must be a positive integer")

         # Validate model_id or base_url is provided
        if not model_id and not base_url:
            raise ValueError("Either model_id or base_url must be provided")
        if model_id and base_url:
            raise ValueError("Cannot provide both model_id and base_url simultaneously")
        
        # Initialize client with clean configuration
        client_config = {
            'token': self.token,
            'timeout': timeout
        }
        
        # Add either model_id or base_url to the config
        if base_url:
            client_config['base_url'] = base_url
        else:
            client_config['model_id'] = model_id
            
        self.client = InferenceClient(**client_config)

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:
        """Generates a text completion for the given message list"""
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )

        # Send messages to the Hugging Face Inference API
        if grammar is not None:
            output = self.client.chat_completion(
                messages,
                stop=stop_sequences,
                response_format=grammar,
                max_tokens=max_tokens,
            )
        else:
            output = self.client.chat.completions.create(
                messages, stop=stop_sequences, max_tokens=max_tokens
            )

        response = output.choices[0].message.content
        self.last_input_token_count = output.usage.prompt_tokens
        self.last_output_token_count = output.usage.completion_tokens
        return response

    def get_tool_call(
        self,
        messages: List[Dict[str, str]],
        available_tools: List[Tool],
        stop_sequences,
    ):
        """Generates a tool call for the given message list. This method is used only by `ToolCallingAgent`."""
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        response = self.client.chat.completions.create(
            messages=messages,
            tools=[get_json_schema(tool) for tool in available_tools],
            tool_choice="auto",
            stop=stop_sequences,
        )
        tool_call = response.choices[0].message.tool_calls[0]
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return tool_call.function.name, tool_call.function.arguments, tool_call.id


class TransformersModel(Model):
    """This engine initializes a model and tokenizer from the given `model_id`."""

    def __init__(self, model_id: Optional[str] = None):
        super().__init__()
        default_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        if model_id is None:
            model_id = default_model_id
            logger.warning(
                f"`model_id`not provided, using this default tokenizer for token counts: '{model_id}'"
            )
        self.model_id = model_id
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer and model for {model_id=}: {e}. Loading default tokenizer and model instead from {model_id=}."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(default_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(default_model_id)

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

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )

        # Get LLM output
        prompt = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
        )
        prompt = prompt.to(self.model.device)
        count_prompt_tokens = prompt["input_ids"].shape[1]

        out = self.model.generate(
            **prompt,
            max_new_tokens=max_tokens,
            stopping_criteria=(
                self.make_stopping_criteria(stop_sequences) if stop_sequences else None
            ),
        )
        generated_tokens = out[0, count_prompt_tokens:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.last_input_token_count = count_prompt_tokens
        self.last_output_token_count = len(generated_tokens)

        if stop_sequences is not None:
            response = remove_stop_sequences(response, stop_sequences)
        return response

    def get_tool_call(
        self,
        messages: List[Dict[str, str]],
        available_tools: List[Tool],
        stop_sequences: Optional[List[str]] = None,
        max_tokens: int = 500,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=[get_json_schema(tool) for tool in available_tools],
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        prompt = prompt.to(self.model.device)
        count_prompt_tokens = prompt["input_ids"].shape[1]

        out = self.model.generate(
            **prompt,
            max_new_tokens=max_tokens,
            stopping_criteria=(
                self.make_stopping_criteria(stop_sequences) if stop_sequences else None
            ),
        )
        generated_tokens = out[0, count_prompt_tokens:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        self.last_input_token_count = count_prompt_tokens
        self.last_output_token_count = len(generated_tokens)

        if stop_sequences is not None:
            response = remove_stop_sequences(response, stop_sequences)

        tool_name, tool_input = parse_json_tool_call(response)
        call_id = "".join(random.choices("0123456789", k=5))

        return tool_name, tool_input, call_id


class LiteLLMModel(Model):
    def __init__(
        self,
        model_id="anthropic/claude-3-5-sonnet-20240620",
        api_base=None,
        api_key=None,
    ):
        super().__init__()
        self.model_id = model_id
        # IMPORTANT - Set this to TRUE to add the function to the prompt for Non OpenAI LLMs
        litellm.add_function_to_prompt = True
        self.api_base = api_base
        self.api_key = api_key

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )

        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            stop=stop_sequences,
            max_tokens=max_tokens,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        return response.choices[0].message.content

    def get_tool_call(
        self,
        messages: List[Dict[str, str]],
        available_tools: List[Tool],
        stop_sequences: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ):
        messages = get_clean_message_list(
            messages, role_conversions=tool_role_conversions
        )
        response = litellm.completion(
            model=self.model_id,
            messages=messages,
            tools=[get_json_schema(tool) for tool in available_tools],
            tool_choice="required",
            stop=stop_sequences,
            max_tokens=max_tokens,
            api_base=self.api_base,
            api_key=self.api_key,
        )
        tool_calls = response.choices[0].message.tool_calls[0]
        self.last_input_token_count = response.usage.prompt_tokens
        self.last_output_token_count = response.usage.completion_tokens
        arguments = json.loads(tool_calls.function.arguments)
        return tool_calls.function.name, arguments, tool_calls.id


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "TransformersModel",
    "HfApiModel",
    "LiteLLMModel",
]
