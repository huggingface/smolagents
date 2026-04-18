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
    OlostepAnswerTool,
    OlostepScrapeWebpageTool,
    OlostepSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)
from smolagents.local_python_executor import ExecutionTimeoutError

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


class DefaultToolTests(unittest.TestCase):
    def test_visit_webpage(self):
        arguments = {"url": "https://huggingface.co/"}
        result = VisitWebpageTool()(arguments)
        assert isinstance(result, str)
        assert "Hugging Face – The AI community building the future" in result

    @require_run_all
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

    def test_custom_timeout(self):
        """Test that PythonInterpreterTool respects custom timeout."""
        tool = PythonInterpreterTool(authorized_imports=["time"], timeout_seconds=1)
        tool.setup()

        # Code that sleeps for 2 seconds should timeout with 1-second limit
        code = """
import time
time.sleep(2)
"""
        with pytest.raises(ExecutionTimeoutError, match="Code execution exceeded the maximum execution time"):
            tool(code)

    def test_disabled_timeout(self):
        """Test that PythonInterpreterTool can disable timeout."""
        tool = PythonInterpreterTool(authorized_imports=["time"], timeout_seconds=None)
        tool.setup()

        # Code should complete even without timeout
        code = """
import time
time.sleep(0.5)
result = "completed"
"""
        result = tool(code)
        assert "completed" in result


class TestSpeechToTextTool:
    def test_new_instance(self):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        tool = SpeechToTextTool()
        assert tool is not None
        assert tool.pre_processor_class == WhisperProcessor
        assert tool.model_class == WhisperForConditionalGeneration

    def test_initialization(self):
        from transformers.models.whisper import WhisperForConditionalGeneration, WhisperProcessor

        tool = SpeechToTextTool(model="dummy_model_id")
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


class TestOlostepSearchTool(ToolTesterMixin):
    """Tests for OlostepSearchTool."""

    def setup_method(self):
        self.tool = OlostepSearchTool(max_results=5, country="US")

    def test_missing_api_key(self, monkeypatch):
        """Test that EnvironmentError is raised when OLOSTEP_API_KEY is not set."""
        monkeypatch.delenv("OLOSTEP_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OLOSTEP_API_KEY environment variable is not set"):
            self.tool.forward("test query")

    def test_initialization_with_params(self):
        """Test that OlostepSearchTool initializes with custom parameters."""
        tool = OlostepSearchTool(max_results=20, country="GB")
        assert tool.max_results == 20
        assert tool.country == "GB"

    def test_tool_attributes(self):
        """Test that the tool has required attributes."""
        assert self.tool.name == "olostep_search"
        assert self.tool.output_type == "string"
        assert "query" in self.tool.inputs
        assert self.tool.inputs["query"]["type"] == "string"

    def test_search_with_results(self, monkeypatch):
        """Test successful search with mocked API."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeScrapes:
            def create(self, **kwargs):
                class FakeResult:
                    json_content = {
                        "organic": [
                            {"title": "Result 1", "link": "https://example.com/1", "snippet": "Snippet 1"}
                        ]
                    }

                return FakeResult()

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("AI agents")
        assert isinstance(result, str)
        assert "Search Results" in result

    def test_search_no_results(self, monkeypatch):
        """Test search when no results are found."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeScrapes:
            def create(self, **kwargs):
                class FakeResult:
                    json_content = {"organic": []}

                return FakeResult()

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("query_with_no_results")
        assert "No results found" in result

    def test_search_api_error(self, monkeypatch):
        """Test search when API raises an error."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        from olostep import Olostep_BaseError

        class FakeScrapes:
            def create(self, **kwargs):
                raise Olostep_BaseError("API Error")

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("test query")
        assert "Olostep search error" in result


class TestOlostepScrapeWebpageTool(ToolTesterMixin):
    """Tests for OlostepScrapeWebpageTool."""

    def setup_method(self):
        self.tool = OlostepScrapeWebpageTool(wait_before_scraping=0, country="US")

    def test_missing_api_key(self, monkeypatch):
        """Test that EnvironmentError is raised when OLOSTEP_API_KEY is not set."""
        monkeypatch.delenv("OLOSTEP_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OLOSTEP_API_KEY environment variable is not set"):
            self.tool.forward("https://example.com")

    def test_initialization_with_params(self):
        """Test that OlostepScrapeWebpageTool initializes with custom parameters."""
        tool = OlostepScrapeWebpageTool(wait_before_scraping=5000, country="GB")
        assert tool.wait_before_scraping == 5000
        assert tool.country == "GB"

    def test_tool_attributes(self):
        """Test that the tool has required attributes."""
        assert self.tool.name == "olostep_scrape_webpage"
        assert self.tool.output_type == "string"
        assert "url" in self.tool.inputs
        assert self.tool.inputs["url"]["type"] == "string"

    def test_scrape_with_content(self, monkeypatch):
        """Test successful scraping with mocked API."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeScrapes:
            def create(self, **kwargs):
                class FakeResult:
                    markdown_content = "# Webpage Title\n\nContent of the page"

                return FakeResult()

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("https://huggingface.co/docs/smolagents")
        assert isinstance(result, str)
        assert "Webpage Title" in result

    def test_scrape_no_content(self, monkeypatch):
        """Test scraping when no content is extracted."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeScrapes:
            def create(self, **kwargs):
                class FakeResult:
                    markdown_content = "   "

                return FakeResult()

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("https://example.com")
        assert "No content could be extracted" in result

    def test_scrape_api_error(self, monkeypatch):
        """Test scraping when API raises an error."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        from olostep import Olostep_BaseError

        class FakeScrapes:
            def create(self, **kwargs):
                raise Olostep_BaseError("API Error")

        class FakeClient:
            scrapes = FakeScrapes()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("https://example.com")
        assert "Olostep scrape error" in result


class TestOlostepAnswerTool(ToolTesterMixin):
    """Tests for OlostepAnswerTool."""

    def setup_method(self):
        self.tool = OlostepAnswerTool()

    def test_missing_api_key(self, monkeypatch):
        """Test that EnvironmentError is raised when OLOSTEP_API_KEY is not set."""
        monkeypatch.delenv("OLOSTEP_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OLOSTEP_API_KEY environment variable is not set"):
            self.tool.forward("What is Python?")

    def test_tool_attributes(self):
        """Test that the tool has required attributes."""
        assert self.tool.name == "olostep_answer"
        assert self.tool.output_type == "string"
        assert "question" in self.tool.inputs
        assert self.tool.inputs["question"]["type"] == "string"

    def test_answer_with_response(self, monkeypatch):
        """Test successful answer generation with mocked API."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeAnswers:
            def create(self, **kwargs):
                class FakeResult:
                    answer = "Python is a high-level programming language."

                return FakeResult()

        class FakeClient:
            answers = FakeAnswers()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("What is Python?")
        assert isinstance(result, str)
        assert "Python is a high-level programming language" in result

    def test_answer_empty_response(self, monkeypatch):
        """Test when no answer could be generated."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        class FakeAnswers:
            def create(self, **kwargs):
                class FakeResult:
                    answer = "   "

                return FakeResult()

        class FakeClient:
            answers = FakeAnswers()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("unanswerable question xyz")
        assert "No answer could be generated" in result

    def test_answer_api_error(self, monkeypatch):
        """Test when API raises an error."""
        monkeypatch.setenv("OLOSTEP_API_KEY", "test-key")

        from olostep import Olostep_BaseError

        class FakeAnswers:
            def create(self, **kwargs):
                raise Olostep_BaseError("API Error")

        class FakeClient:
            answers = FakeAnswers()

        monkeypatch.setattr("olostep.Olostep", lambda api_key: FakeClient())

        result = self.tool.forward("What is X?")
        assert "Olostep answer error" in result
