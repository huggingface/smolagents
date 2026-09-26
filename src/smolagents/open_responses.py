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

from typing import Any, Dict, List
from .models import ChatMessage

def messages_to_open_responses_input(messages: List[ChatMessage | dict]) -> List[dict]:
    formatted_messages = []

    for msg in messages:
        # Normalize role
        if isinstance(msg, dict):
            role = msg["role"]
            content = msg.get("content")
        else:
            role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
            content = msg.content

        # Determine correct Open Responses content type
        is_assistant = role == "assistant"
        content_type = "output_text" if is_assistant else "input_text"

        content_blocks = []

        # Plain text
        if isinstance(content, str):
            content_blocks.append({
                "type": content_type,
                "text": content,
            })

        # Structured / multimodal
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "type" in block:
                    # Convert legacy "text"
                    if block["type"] == "text":
                        block = {**block, "type": content_type}
                    content_blocks.append(block)
                else:
                    content_blocks.append({
                        "type": content_type,
                        "text": str(block),
                    })

        # Empty
        elif content is None:
            content_blocks = []

        else:
            content_blocks.append({
                "type": content_type,
                "text": str(content),
            })

        formatted_messages.append({
            "role": role,
            "content": content_blocks,
        })

    return formatted_messages