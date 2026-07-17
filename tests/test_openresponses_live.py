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

import unittest
import pytest

from smolagents import CodeAgent, OpenAIModel
from smolagents.open_responses_stream import OpenResponsesStreamRuntime
from smolagents.open_responses import messages_to_open_responses_input
from smolagents.monitoring import TokenUsage
import openai

def test_messages_to_openresponses_input_text():
    messages = [{"role": "user", "content": "hi"}]
    out = messages_to_open_responses_input(messages)

    assert out[0]["content"][0]["type"] == "input_text"
    assert out[0]["content"][0]["text"] == "hi"

def test_runtime_on_bad_events():
    rt = OpenResponsesStreamRuntime()

    malformed = [
        {},
        {"type": "response.output_text.delta"},
        {"type": "response.function_call.done"},
        {"type": "response.output_item.added", "item": {}},
    ]

    for e in malformed:
        rt.process_event(e)

def test_runtime_text_stream_flow():
    rt = OpenResponsesStreamRuntime()

    events = [
        {
            "type": "response.output_item.added",
            "item": {"id": "msg1", "type": "message"},
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg1",
            "content_index": 0,
            "part": {"type": "output_text"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg1",
            "content_index": 0,
            "delta": "Hi",
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg1",
            "content_index": 0,
            "delta": "!",
        },
    ]

    chunks = []
    for e in events:
        for d in rt.process_event(e):
            if d.content:
                chunks.append(d.content)

    assert "".join(chunks) == "Hi!"

def test_runtime_tool_call_streaming():
    rt = OpenResponsesStreamRuntime()

    events = [
        {"type": "response.output_item.added", "item": {"id": "tool1", "type": "tool_call"}},
        {"type": "response.function_call.delta", "item_id": "tool1", "name": "final_answer", "delta": '{"answer":'},
        {"type": "response.function_call.delta", "item_id": "tool1", "delta": '"ok"}'},
        {"type": "response.function_call.done", "item_id": "tool1"},
    ]

    tool_out = None
    for e in events:
        out = rt.process_event(e)
        if out:
            tool_out = out[-1]

    assert tool_out is not None
    assert tool_out.tool_calls[0].function.name == "final_answer"

def test_runtime_response_completed_usage():
    rt = OpenResponsesStreamRuntime()

    fake_completed = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
            }
        },
    }

    deltas = rt.process_event(fake_completed)

    assert deltas
    assert deltas[0].token_usage.input_tokens == 3
    assert deltas[0].token_usage.output_tokens == 5

def test_openresponses_live_stream_full():
    import os
    model = OpenAIModel(
        model_id="gpt-5.2-codex",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.0,
        max_tokens=20,
        use_open_responses=True,
    )

    agent = CodeAgent(
        tools=[],
        model=model,
        stream_outputs=True,
        max_steps=1,
    )

    output = list(agent.run("Say OK."))

    assert "OK" in "".join([tok or "" for tok in output])
    assert agent.monitor.total_input_token_count > 0
    assert agent.monitor.total_output_token_count > 0

def test_openresponses_flag_disabled_falls_back():
    import os
    model = OpenAIModel(
        model_id="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.0,
        max_tokens=10,
        use_open_responses=False,
    )

    agent = CodeAgent(tools=[], model=model, max_steps=1)

    outputs = list(agent.run("Say OK."))
    assert len(outputs) > 0