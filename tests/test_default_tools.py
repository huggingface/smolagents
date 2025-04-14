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

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from .test_tools import ToolTesterMixin


class DefaultToolTests(unittest.TestCase):
    def test_visit_webpage(self):
        arguments = {"url": "https://en.wikipedia.org/wiki/United_States_Secretary_of_Homeland_Security"}
        result = VisitWebpageTool()(arguments)
        assert isinstance(result, str)
        assert "* [About Wikipedia](/wiki/Wikipedia:About)" in result  # Proper wikipedia pages have an About

    def test_ddgs_with_kwargs(self):
        result = DuckDuckGoSearchTool(timeout=20)("DeepSeek parent company")
        assert isinstance(result, str)


class TestPythonInterpreterTool(ToolTesterMixin):
    def setup_method(self):
        self.tool = PythonInterpreterTool(authorized_imports=["numpy"])
        self.tool.setup()

    def test_exact_match_arg(self):
        result = self.tool("(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_exact_match_kwarg(self):
        result = self.tool(code="(2 / 2) * 4")
        assert result == "Stdout:\n\nOutput: 4.0"

    def test_agent_type_output(self):
        inputs = ["2 * 2"]
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_agent_types_inputs(self):
        inputs = ["2 * 2"]
        _inputs = []

        for _input, expected_input in zip(inputs, self.tool.inputs.values()):
            input_type = expected_input["type"]
            if isinstance(input_type, list):
                _inputs.append([_AGENT_TYPE_MAPPING[_input_type](_input) for _input_type in input_type])
            else:
                _inputs.append(_AGENT_TYPE_MAPPING[input_type](_input))

        # Should not raise an error
        output = self.tool(*inputs, sanitize_inputs_outputs=True)
        output_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
        assert isinstance(output, output_type)

    def test_imports_work(self):
        result = self.tool("import numpy as np")
        assert "import from numpy is not allowed" not in result.lower()

    def test_unauthorized_imports_fail(self):
        with pytest.raises(Exception) as e:
            self.tool("import sympy as sp")
        assert "sympy" in str(e).lower()


class TestSpeechToTextTool:
    def test_new_instance(self):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        tool = SpeechToTextTool()
        assert tool is not None
        assert tool.pre_processor_class == WhisperProcessor
        assert tool.model_class == WhisperForConditionalGeneration


@pytest.mark.parametrize(
    "language, content_type, extract_format, query",
    [
        ("en", "summary", "HTML", "Python_(programming_language)"),  # English, Summary Mode, HTML format
        ("en", "text", "WIKI", "Python_(programming_language)"),  # English, Full Text Mode, WIKI format
        ("es", "summary", "HTML", "Python_(lenguaje_de_programación)"),  # Spanish, Summary Mode, HTML format
        ("es", "text", "WIKI", "Python_(lenguaje_de_programación)"),  # Spanish, Full Text Mode, WIKI format
    ],
)
def test_wikipedia_search(language, content_type, extract_format, query):
    tool = WikipediaSearchTool(
        user_agent="TestAgent (test@example.com)",
        language=language,
        content_type=content_type,
        extract_format=extract_format,
    )

    result = tool.forward(query)

    assert isinstance(result, str), "Output should be a string"
    assert "✅ **Wikipedia Page:**" in result, "Response should contain Wikipedia page title"
    assert "🔗 **Read more:**" in result, "Response should contain Wikipedia page URL"

    if content_type == "summary":
        assert len(result.split()) < 1000, "Summary mode should return a shorter text"
    if content_type == "text":
        assert len(result.split()) > 1000, "Full text mode should return a longer text"
