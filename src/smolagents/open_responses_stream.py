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

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

from .models import (
    ChatMessageStreamDelta,
    ChatMessageToolCallStreamDelta,
    ChatMessageToolCallFunction,
)
from .monitoring import TokenUsage


@dataclass
class ContentPartState:
    type: str
    text_buffer: str = ""
    done: bool = False


@dataclass
class ToolCallState:
    name: Optional[str] = None
    arguments_buffer: str = ""
    done: bool = False


@dataclass
class ItemState:
    id: str
    type: str
    status: str = "in_progress"
    content_parts: Dict[int, ContentPartState] = field(default_factory=dict)
    tool_call: Optional[ToolCallState] = None


class OpenResponsesStreamRuntime:
    """
    Spec-compliant Open Responses streaming processor.
    Consumes SSE text/event-stream lines and yields ChatMessageStreamDelta objects.
    """

    def __init__(self):
        self.items: Dict[str, ItemState] = {}
        self.token_usage: Optional[TokenUsage] = None

    def process_sse_stream(
        self, sse_lines: Generator[str, None, None]
    ) -> Generator[ChatMessageStreamDelta, None, None]:
        """
        Entry point: consumes SSE text/event-stream lines
        and yields ChatMessageStreamDelta objects.
        """

        for line in sse_lines:
            line = line.strip()

            if not line:
                continue

            # Terminal signal
            if line == "[DONE]":
                break

            if not line.startswith("data:"):
                continue

            payload = line[len("data:") :].strip()

            try:
                event = json.loads(payload)
            except Exception:
                continue

            yield from self._handle_event(event)

        if self.token_usage:
            yield ChatMessageStreamDelta(token_usage=self.token_usage)

    def _handle_event(self, event: Dict[str, Any]):
        if not isinstance(event, dict):
            return []

        event_type = event.get("type")
        if not event_type:
            return []

        try:
            if event_type == "response.output_item.added":
                return self._on_item_added(event)

            if event_type == "response.content_part.added":
                return self._on_content_part_added(event)

            if event_type == "response.output_text.delta":
                return self._on_text_delta(event)

            if event_type == "response.output_text.done":
                return self._on_text_done(event)

            if event_type == "response.content_part.done":
                return []

            if event_type == "response.output_item.done":
                return self._on_item_done(event)

            if event_type == "response.function_call.delta":
                return self._on_tool_delta(event)

            if event_type == "response.function_call.done":
                return self._on_tool_done(event)

            if event_type == "response.completed":
                return self._on_response_completed(event)


        except Exception:
            return []

        return []
    
    def process_event(self, event: dict):
        """
        Public event entrypoint for Open Responses runtime.
        Safe wrapper around internal handler.
        Always returns iterable deltas.
        """
        try:
            return self._handle_event(event) or []
        except Exception:
            # Fail safe — streaming must never crash agent loop
            return []

    def _on_item_added(self, event):
        item = event.get("item")
        if not item:
            return []

        item_id = item.get("id")
        if not item_id:
            return []

        self.items[item_id] = ItemState(
            id=item_id,
            type=item.get("type", "unknown"),
            status=item.get("status", "in_progress"),
        )

        return []

    def _on_item_done(self, event):
        item = event["item"]
        state = self.items.get(item["id"])

        if state:
            state.status = "completed"

        return []

    def _on_content_part_added(self, event):
        item_id = event.get("item_id")
        index = event.get("content_index")

        if item_id not in self.items:
            return []

        if index is None:
            return []

        part = event.get("part", {})

        self.items[item_id].content_parts[index] = ContentPartState(
            type=part.get("type", "output_text"),
            text_buffer="",
            done=False,
        )

        return []

    def _on_text_delta(self, event):
        item_id = event.get("item_id")
        index = event.get("content_index", 0)
        delta = event.get("delta", "")

        state = self.items.get(item_id)
        if not state:
            return []

        part = state.content_parts.get(index)
        if not part or part.done:
            return []

        part.text_buffer += delta

        return [ChatMessageStreamDelta(content=delta)]

    def _on_text_done(self, event):
        item_id = event["item_id"]
        index = event["content_index"]

        state = self.items[item_id]
        state.content_parts[index].done = True

        return []

    def _on_content_part_done(self, event):
        return []

    def _on_tool_delta(self, event):
        item_id = event.get("item_id")
        if item_id not in self.items:
            return []

        state = self.items[item_id]

        if state.tool_call is None:
            state.tool_call = ToolCallState(name=event.get("name"))

        delta = event.get("delta", "")
        state.tool_call.arguments_buffer += delta

        return []

    def _on_tool_done(self, event):
        item_id = event.get("item_id")
        state = self.items.get(item_id)

        if not state or not state.tool_call:
            return []

        raw = state.tool_call.arguments_buffer

        try:
            parsed = json.loads(raw)
        except Exception:
            # Spec-tolerant ignoring invalid JSON for now
            parsed = raw

        state.tool_call.arguments_buffer = parsed
        state.tool_call.done = True

        return [
            ChatMessageStreamDelta(
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        function=ChatMessageToolCallFunction(
                            name=state.tool_call.name or "unknown",
                            arguments=parsed,
                        )
                    )
                ]
            )
        ]

    def _on_response_completed(self, event):
        response = event.get("response")
        if not response:
            return []

        usage = getattr(response, "usage", None) or response.get("usage") or {}

        # Handle SDK object OR raw dict
        if hasattr(usage, "input_tokens"):
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
        else:
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        self.token_usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return [
            ChatMessageStreamDelta(token_usage=self.token_usage)
        ]

