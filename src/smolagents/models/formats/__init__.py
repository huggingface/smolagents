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

"""Format handling for various LLM API specifications.

This module provides data structures and utilities for working with different
message and tool call formats used by LLM APIs.
"""

from .chat_completion import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    ChatMessageToolCallStreamDelta,
    MessageRole,
    _coerce_tool_call,
    agglomerate_stream_deltas,
    get_clean_message_list,
    get_dict_from_nested_dataclasses,
    get_tool_call_from_text,
    parse_json_if_needed,
    tool_role_conversions,
)


__all__ = [
    "ChatMessage",
    "ChatMessageStreamDelta",
    "ChatMessageToolCall",
    "ChatMessageToolCallFunction",
    "ChatMessageToolCallStreamDelta",
    "MessageRole",
    "_coerce_tool_call",
    "agglomerate_stream_deltas",
    "get_clean_message_list",
    "get_dict_from_nested_dataclasses",
    "get_tool_call_from_text",
    "parse_json_if_needed",
    "tool_role_conversions",
]
