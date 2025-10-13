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
import logging
import re
import uuid
from collections.abc import AsyncGenerator
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from smolagents.monitoring import TokenUsage
from smolagents.tools import Tool
from smolagents.utils import encode_image_base64, make_image_url, parse_json_blob, RateLimiter
from smolagents.models import ChatMessageToolCallStreamDelta, ChatMessageToolCallFunction
from smolagents import ChatMessage, ChatMessageToolCall, ChatMessageStreamDelta, parse_json_if_needed, MessageRole
if TYPE_CHECKING:
    pass


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


def agglomerate_stream_deltas(
        stream_deltas: list[ChatMessageStreamDelta], role: MessageRole = MessageRole.ASSISTANT
) -> ChatMessage:
    """
    Agglomerate a list of stream deltas into a single stream delta.
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}

    chunks = []
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            chunks.append(stream_delta.content)
        if stream_delta.tool_calls:
            for tool_call_delta in stream_delta.tool_calls:  # ?ormally there should be only one call at a time
                # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = ChatMessageToolCallStreamDelta(
                            id=tool_call_delta.id,
                            type=tool_call_delta.type,
                            function=ChatMessageToolCallFunction(name="", arguments=""),
                        )
                    # Update the tool call at the specific index
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if tool_call_delta.function.name and len(tool_call_delta.function.name) > 0:
                            tool_call.function.name = tool_call_delta.function.name
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += tool_call_delta.function.arguments
                else:
                    raise ValueError(f"Tool call index is not provided in tool delta: {tool_call_delta}")

    return ChatMessage(
        role=role,
        content="".join(chunks),
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


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
        if content[-len(stop_seq):] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
        message_list: list[ChatMessage | dict],
        role_conversions=None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """
    Creates a list of messages to give as input to the LLM. These messages are dictionaries and chat template compatible with transformers LLM chat template.
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`list[ChatMessage | dict]`): List of chat messages. Mixed types are allowed.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    if role_conversions is None:
        role_conversions = {}
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message.content, list):
            for element in message.content:
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

        if len(output_message_list) > 0 and message.role == output_message_list[-1]["role"]:
            assert isinstance(message.content, list), "Error: wrong content:" + str(message.content)
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message.content[0]["text"]
            else:
                for el in message.content:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
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
        function=ChatMessageToolCallFunction(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and the openai/gpt-5 series (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    # o3, o4-mini, and the gpt-5 series (including versioned variants, o3-2025-04-16) don't support stop parameter
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*|gpt-5(-mini|-nano)?[-\d]*)$"
    return not re.match(pattern, model_name)


class AsyncModel:
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
        self.model_id: str | None = model_id

    def _prepare_completion_kwargs(
            self,
            messages: list[ChatMessage | dict],
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
        messages_as_dicts = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
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

    async def generate(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response in asynchronous format.

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

    async def __call__(self, *args, **kwargs):
        return await self.generate(*args, **kwargs)

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
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "AsyncModel":
        return cls(**{k: v for k, v in model_dictionary.items()})

    async def generate_stream(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta|NotImplementedError]:
        yield NotImplementedError("This method should be implemented in child classes")

class AsyncApiModel(AsyncModel):
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
        requests_per_minute (`float`, **optional**):
            Rate limit in requests per minute.
        **kwargs: Additional keyword arguments to pass to the parent class.
    """

    def __init__(
            self,
            model_id: str,
            custom_role_conversions: dict[str, str] | None = None,
            client: Any | None = None,
            requests_per_minute: float | None = None,
            **kwargs,
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()
        self.rate_limiter = RateLimiter(requests_per_minute)

    def create_client(self):
        """Create the API client for the specific service."""
        raise NotImplementedError("Subclasses must implement this method to create a client")

    async def _apply_rate_limit(self):
        """Apply rate limiting before making API calls."""
        await self.rate_limiter.athrottle()


class AsyncOpenAIServerModel(AsyncApiModel):
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

        return openai.AsyncOpenAI(**self.client_kwargs)

    async def generate_stream(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta]:
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
        await self._apply_rate_limit()

        response = await self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        )

        async for event in response:
            if event.usage:
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")

    async def generate(
            self,
            messages: list[ChatMessage | dict],
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
        await self._apply_rate_limit()
        response = await self.client.chat.completions.create(**completion_kwargs)
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


AsyncOpenAIModel = AsyncOpenAIServerModel


class AsyncAzureOpenAIServerModel(AsyncOpenAIServerModel):
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

        return openai.AsyncAzureOpenAI(**self.client_kwargs)


AzureOpenAIModel = AsyncAzureOpenAIServerModel


class AsyncAnthropicAwsBedrock(AsyncApiModel):
    """
    A model class for interacting with Amazon Bedrock Server models through the Bedrock API.

    This class provides an interface to interact with various Bedrock language models,
    allowing for customized model inference, guardrail configuration, message handling,
    and other parameters allowed by boto3 API.

    Authentication:

    Amazon Bedrock supports multiple authentication methods:
    - Default AWS credentials:
       Use the default AWS credential chain (e.g., IAM roles, IAM users).
    - API Key Authentication (requires `boto3 >= 1.39.0`):
       Set the API key using the `AWS_BEARER_TOKEN_BEDROCK` environment variable.

    > [!TIP]
    > API key support requires `boto3 >= 1.39.0`.
    > For users not relying on API key authentication, the minimum supported version is `boto3 >= 1.36.18`.

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
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            tool_choice: str | dict[Any, Any] | None = None,
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
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if not isinstance(content, dict):
                        continue
                    if "type" not in content:
                        content["type"] = "text"
                    elif "text" not in content and "content" in content:
                        content["text"] = content.pop("content")

        messages = completion_kwargs.get("messages", messages)

        completion_kwargs["messages"] = messages

        return {
            "model": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        try:
            import boto3  # type: ignore
            import anthropic
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'anthropic', 'boto3' to use AsyncAnthropicAwsBedrock: `pip install boto3 anthropic`"
            ) from e

        return anthropic.AsyncAnthropicBedrock(**self.client_kwargs)

    async def generate(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            max_tokens: int = 8192,
            **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            stream=False,
            max_tokens=max_tokens,
            **kwargs,
        )
        await self._apply_rate_limit()
        response = await self.client.messages.create(**completion_kwargs)
        response = response.to_dict()

        return ChatMessage(
            role=response["role"],
            content=response["content"][0]["text"],
            token_usage=TokenUsage(
                input_tokens=response["usage"]["input_tokens"],
                output_tokens=response["usage"]["output_tokens"],
            ),
            raw=response,
        )

    async def generate_stream(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            max_tokens: int = 128000,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta]:
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            max_tokens=max_tokens,
            stream=True,
            **kwargs,
        )
        await self._apply_rate_limit()
        response = await self.client.messages.create(**completion_kwargs)
        async for event in response:
            _type = event.type
            if _type == "content_block_delta":
                if event.delta:
                    yield ChatMessageStreamDelta(
                        content=event.delta.text,
                        tool_calls=None,
                        token_usage=None
                    )
            elif _type == "message_stop":
                dict_value = event.to_dict()
                _metric = dict_value["amazon-bedrock-invocationMetrics"]
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=_metric["inputTokenCount"],
                        output_tokens=_metric["outputTokenCount"],
                    )
                )


class AsyncGoogleGemini(AsyncApiModel):
    def __init__(
            self,
            model_id: str,
            client=None,
            client_kwargs: dict[str, Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            **kwargs,
    ):
        self.client_kwargs = client_kwargs or {}

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API doesn't support flatten messages, must be a list of messages
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            tool_choice: str | dict[Any, Any] | None = None,
            **kwargs,
    ) -> dict:
        """
        Overrides the base method to handle Bedrock-specific configurations.

        This implementation adapts the completion keyword arguments to align with
        Bedrock's requirements, ensuring compatibility with its unique setup and
        constraints.
        """
        from google.genai import types

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
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            elif isinstance(message["content"], list):
                for content in message["content"]:
                    if not isinstance(content, dict):
                        continue
                    if "type" not in content:
                        content["type"] = "text"
                    elif "content" not in content and "text" in content:
                        content["content"] = content.pop("text")

        completion_kwargs["messages"] = [
            types.Content(
                role=inp["role"] if inp["role"] == "user" else "model",
                parts=[types.Part(text=part["text"]) for part in inp["content"]],
            )
            for inp in completion_kwargs["messages"]
        ]

        completion_kwargs["config"] = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=kwargs.get("thinking_budget", 0)),
            cached_content=self.kwargs.get("cache").name if self.kwargs.get("cache") else None,
        )

        return completion_kwargs

    def create_client(self):
        try:
            import google.genai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'google-genai' to use AsyncGoogleGemini: `pip install google-genai`"
            ) from e

        return google.genai.client.Client(**self.client_kwargs)

    async def generate(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            max_tokens: int = 8192,
            **kwargs,
    ) -> ChatMessage:

        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            max_tokens=max_tokens,
            **kwargs,
        )

        chat = self.client._aio.chats.create(model=self.model_id, history=completion_kwargs["messages"][:-1])
        response = await chat.send_message(message=completion_kwargs["messages"][-1].parts, config=completion_kwargs["config"])
        response = response.to_json_dict()

        content = response["candidates"][0]["content"]
        metadata = response["usage_metadata"]
        return ChatMessage(
            role=content["role"],
            content=content["parts"][0]["text"],
            raw=response,
            token_usage=TokenUsage(
                input_tokens=metadata.get("prompt_token_count", 0),
                output_tokens=metadata.get("candidates_token_count", 0)
            ),
            tool_calls=None
        )

    async def generate_stream(
            self,
            messages: list[ChatMessage | dict],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Tool] | None = None,
            max_tokens: int = 128000,
            **kwargs,
    ) -> AsyncGenerator[ChatMessageStreamDelta]:
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            max_tokens=max_tokens,
            **kwargs,
        )

        chat = self.client._aio.chats.create(model=self.model_id, history=completion_kwargs["messages"][:-1])
        response = await chat.send_message_stream(message=completion_kwargs["messages"][-1].parts, config=completion_kwargs["config"])

        input_tokens, output_tokens = 0, 0
        async for chunk in response:
            chunk = chunk.to_json_dict()
            content = chunk["candidates"][0]["content"]
            metadata = chunk["usage_metadata"]
            _input_tokens = metadata.get("prompt_token_count", 0)
            _output_tokens = metadata.get("candidates_token_count", 0)
            yield ChatMessageStreamDelta(
                content = content["parts"][0]["text"],
                token_usage=TokenUsage(
                    input_tokens=_input_tokens - input_tokens,
                    output_tokens=_output_tokens - output_tokens,
                )
            )
            input_tokens = _input_tokens
            output_tokens = _output_tokens


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "AsyncModel",
    "AsyncApiModel",
    "AsyncOpenAIServerModel",
    "AsyncOpenAIModel",
    "AsyncAzureOpenAIServerModel",
    "AzureOpenAIModel",
    "AsyncAnthropicAwsBedrock",
    "AsyncGoogleGemini",
    "ChatMessage",
]
