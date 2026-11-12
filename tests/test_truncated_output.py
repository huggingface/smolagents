#!/usr/bin/env python
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
"""A code action from a reply the provider marked truncated must not run.

The class below stands in for the API client, not for its payloads: it returns the same
`ChatMessage` the client builds, carrying the finish reason the provider gave.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from smolagents.agents import CodeAgent
from smolagents.models import (
    ChatMessage,
    ChatMessageStreamDelta,
    InferenceClientModel,
    LiteLLMModel,
    MessageRole,
    Model,
    OpenAIModel,
    agglomerate_stream_deltas,
)
from smolagents.tools import Tool


TRUNCATED_CODE = "<code>\nappend_line('only line')\n"
FINAL_ANSWER = '<code>\nfinal_answer("done")\n</code>'


class ScriptedModel(Model):
    """Sends the truncated code block once, then a normal final answer."""

    def __init__(self, finish_reason):
        super().__init__()
        self.finish_reason = finish_reason
        self.calls = 0

    def generate(self, messages, tools_to_call_from=None, stop_sequences=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return self._message(TRUNCATED_CODE, self.finish_reason)
        return self._message(FINAL_ANSWER, "stop")

    @staticmethod
    def _message(content, finish_reason):
        # Before the fix ChatMessage has no such field, so the model simply cannot
        # express an end reason; that is the pre-fix behaviour under test.
        kwargs = {}
        if "finish_reason" in {field.name for field in dataclasses.fields(ChatMessage)}:
            kwargs["finish_reason"] = finish_reason
        return ChatMessage(role=MessageRole.ASSISTANT, content=content, **kwargs)


class AppendLineTool(Tool):
    """Records every line it is asked to append."""

    name = "append_line"
    description = "Append one line to the report."
    inputs = {"line": {"type": "string", "description": "The line to append."}}
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.written = []

    def forward(self, line: str) -> str:
        self.written.append(line)
        return "ok"


def _agent(model):
    tool = AppendLineTool()
    agent = CodeAgent(tools=[tool], model=model)
    return agent, tool.written


def test_truncated_reply_is_not_executed():
    """finish_reason='length' plus a code block: the code must not run."""
    model = ScriptedModel("length")
    agent, written = _agent(model)

    agent.run("write the report")

    assert written == []
    assert model.calls >= 2  # the reply was reported as truncated, so the code never ran


def test_complete_reply_is_executed_once():
    """A normal ending still executes, exactly once."""
    model = ScriptedModel("stop")
    agent, written = _agent(model)

    agent.run("write the report")

    assert written == ["only line"]


def test_missing_finish_reason_is_unchanged():
    """No end reason at all is out of scope for tier 1: behaviour must not change."""
    model = ScriptedModel(None)
    agent, written = _agent(model)

    agent.run("write the report")

    assert written == ["only line"]


def test_agglomeration_keeps_the_finish_reason():
    deltas = [
        ChatMessageStreamDelta(content=TRUNCATED_CODE),
        ChatMessageStreamDelta(content="", finish_reason="length"),
    ]

    message = agglomerate_stream_deltas(deltas)

    assert message.finish_reason == "length"
    assert message.content == TRUNCATED_CODE


def _completion_response(content, finish_reason):
    """The shape the OpenAI-compatible clients return, with only what the adapter reads."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=content, tool_calls=None),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _transport(response, path):
    """A stub client whose one behaviour is answering `path` with `response`."""
    client = Mock()
    *parents, leaf = path.split(".")
    target = client
    for part in parents:
        target = getattr(target, part)
    getattr(target, leaf).return_value = response
    return client


def test_real_adapter_carries_the_truncation_to_the_guard():
    """Only the transport is faked: OpenAIModel.generate() must pass on the end reason.

    The tests above inject `ChatMessage` directly, so they say nothing about whether the
    adapter sets that field. This one goes through the real `OpenAIModel` and the real
    `CodeAgent`, and fails if `generate()` rebuilds the message without the end reason.
    """
    client = Mock()
    client.chat.completions.create.side_effect = [
        _completion_response(TRUNCATED_CODE, "length"),
        _completion_response(FINAL_ANSWER, "stop"),
    ]
    model = OpenAIModel(model_id="local-test", client=client)
    tool = AppendLineTool()
    agent = CodeAgent(tools=[tool], model=model)

    assert agent.run("write the report") == "done"

    assert tool.written == []
    assert client.chat.completions.create.call_count == 2


@pytest.mark.parametrize(
    ("model_class", "call_path"),
    [
        (OpenAIModel, "chat.completions.create"),
        (LiteLLMModel, "completion"),
        (InferenceClientModel, "chat_completion"),
    ],
)
def test_every_non_streaming_adapter_keeps_the_end_reason(model_class, call_path):
    """Every adapter that rebuilds the reply for `generate()` must keep the end reason."""
    client = _transport(_completion_response(TRUNCATED_CODE, "length"), call_path)

    message = model_class(model_id="local-test", client=client).generate([{"role": "user", "content": "hi"}])

    assert message.content == TRUNCATED_CODE
    assert message.finish_reason == "length"
