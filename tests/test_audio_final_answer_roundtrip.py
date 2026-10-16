# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Ensure final_answer preserves typed audio and original sampling metadata."""

import pytest
from PIL import Image

from smolagents import AgentAudio, AgentImage, AgentText, Model, Tool, ToolCallingAgent
from smolagents.agent_types import handle_agent_output_types
from smolagents.default_tools import FinalAnswerTool
from smolagents.models import ChatMessage


torch = pytest.importorskip("torch")
pytest.importorskip("soundfile")


class Waveform(Tool):
    name = "waveform"
    description = "Generate a deterministic waveform."
    inputs = {}
    output_type = "audio"

    def forward(self):
        return (8000, [0.0, 0.25, -0.25])


class RecordedModel(Model):
    def __init__(self):
        super().__init__()
        self.calls = iter([("waveform", {}), ("final_answer", {"answer": "audio.mp3"})])

    def generate(self, messages, **kwargs):
        name, arguments = next(self.calls)
        return ChatMessage.from_dict(
            {
                "role": "assistant",
                "tool_calls": [{"id": name, "type": "function", "function": {"name": name, "arguments": arguments}}],
            }
        )


def test_original_tool_audio_has_expected_metadata():
    value = Waveform()(sanitize_inputs_outputs=True)
    assert isinstance(value, AgentAudio)
    assert value.samplerate == 8000
    torch.testing.assert_close(value.to_raw(), torch.tensor([0.0, 0.25, -0.25]))


def test_final_output_converter_does_not_downcast_agent_audio_to_text():
    original = Waveform()(sanitize_inputs_outputs=True)
    result = handle_agent_output_types(original)
    assert result is original
    assert result.samplerate == 8000
    assert not isinstance(result, AgentText)


def test_tool_calling_agent_preserves_audio_rate_and_type_through_final_answer():
    agent = ToolCallingAgent(
        tools=[Waveform()], model=RecordedModel(), max_steps=2, verbosity_level=-1
    )
    result = agent.run("Return the waveform.")
    assert isinstance(result, AgentAudio)
    assert result is agent.state["audio.mp3"]
    assert result.samplerate == 8000
    torch.testing.assert_close(result.to_raw(), torch.tensor([0.0, 0.25, -0.25]))
    assert not isinstance(result, AgentText)


def test_generic_plain_string_still_becomes_agent_text():
    result = handle_agent_output_types("finished")
    assert isinstance(result, AgentText)
    assert str(result) == "finished"


class ImageSizeFinalAnswer(FinalAnswerTool):
    output_type = "string"

    def forward(self, answer):
        return f"{answer.width}x{answer.height}"


def test_custom_final_answer_tool_still_receives_sanitized_agent_image():
    agent = ToolCallingAgent(
        tools=[ImageSizeFinalAnswer()], model=Model(), verbosity_level=-1
    )
    agent.state["image.png"] = AgentImage(Image.new("RGB", (8, 6)))

    assert agent.execute_tool_call("final_answer", {"answer": "image.png"}) == "8x6"
