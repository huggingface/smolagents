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


class TestWebSearchTimeouts:
    """Tests that web search tools pass timeout to requests.get (issue #1713)."""

    def test_google_search_tool_default_timeout(self):
        """GoogleSearchTool should default to 20s timeout."""
        import os

        os.environ["SERPAPI_API_KEY"] = "test-key"
        try:
            from smolagents.default_tools import GoogleSearchTool

            tool = GoogleSearchTool(provider="serpapi")
            assert tool.timeout == 20
        finally:
            del os.environ["SERPAPI_API_KEY"]

    def test_google_search_tool_custom_timeout(self):
        """GoogleSearchTool should accept a custom timeout."""
        import os

        os.environ["SERPAPI_API_KEY"] = "test-key"
        try:
            from smolagents.default_tools import GoogleSearchTool

            tool = GoogleSearchTool(provider="serpapi", timeout=60)
            assert tool.timeout == 60
        finally:
            del os.environ["SERPAPI_API_KEY"]

    def test_google_search_tool_passes_timeout_to_requests(self):
        """GoogleSearchTool.forward() should pass timeout to requests.get."""
        import os
        from unittest.mock import MagicMock, patch

        os.environ["SERPAPI_API_KEY"] = "test-key"
        try:
            from smolagents.default_tools import GoogleSearchTool

            tool = GoogleSearchTool(provider="serpapi", timeout=30)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"organic_results": []}
            with patch("requests.get", return_value=mock_response) as mock_get:
                tool.forward("test query")
                mock_get.assert_called_once()
                assert mock_get.call_args.kwargs.get("timeout") == 30
        finally:
            del os.environ["SERPAPI_API_KEY"]

    def test_api_web_search_tool_default_timeout(self):
        """ApiWebSearchTool should default to 20s timeout."""
        from smolagents.default_tools import ApiWebSearchTool

        tool = ApiWebSearchTool(api_key="test-key")
        assert tool.timeout == 20

    def test_api_web_search_tool_custom_timeout(self):
        """ApiWebSearchTool should accept a custom timeout."""
        from smolagents.default_tools import ApiWebSearchTool

        tool = ApiWebSearchTool(api_key="test-key", timeout=45)
        assert tool.timeout == 45

    def test_api_web_search_tool_passes_timeout_to_requests(self):
        """ApiWebSearchTool.forward() should pass timeout to requests.get."""
        from unittest.mock import MagicMock, patch

        from smolagents.default_tools import ApiWebSearchTool

        tool = ApiWebSearchTool(api_key="test-key", timeout=30, rate_limit=None)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.forward("test query")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs.get("timeout") == 30

    def test_web_search_tool_default_timeout(self):
        """WebSearchTool should default to 20s timeout."""
        from smolagents.default_tools import WebSearchTool

        tool = WebSearchTool()
        assert tool.timeout == 20

    def test_web_search_tool_custom_timeout(self):
        """WebSearchTool should accept a custom timeout."""
        from smolagents.default_tools import WebSearchTool

        tool = WebSearchTool(timeout=10)
        assert tool.timeout == 10

    def test_web_search_tool_duckduckgo_passes_timeout(self):
        """WebSearchTool.search_duckduckgo() should pass timeout to requests.get."""
        from unittest.mock import MagicMock, patch

        from smolagents.default_tools import WebSearchTool

        tool = WebSearchTool(engine="duckduckgo", timeout=15)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.search_duckduckgo("test query")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs.get("timeout") == 15

    def test_web_search_tool_bing_passes_timeout(self):
        """WebSearchTool.search_bing() should pass timeout to requests.get."""
        from unittest.mock import MagicMock, patch

        from smolagents.default_tools import WebSearchTool

        tool = WebSearchTool(engine="bing", timeout=25)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<rss><channel></channel></rss>"
        mock_response.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.search_bing("test query")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs.get("timeout") == 25

    def test_visit_webpage_tool_default_timeout(self):
        """VisitWebpageTool should default to 20s timeout."""
        tool = VisitWebpageTool()
        assert tool.timeout == 20

    def test_visit_webpage_tool_custom_timeout(self):
        """VisitWebpageTool should accept a custom timeout."""
        tool = VisitWebpageTool(timeout=60)
        assert tool.timeout == 60

    def test_visit_webpage_tool_passes_timeout_to_requests(self):
        """VisitWebpageTool.forward() should pass timeout to requests.get."""
        from unittest.mock import MagicMock, patch

        tool = VisitWebpageTool(timeout=35)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Hello</body></html>"
        mock_response.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.forward("https://example.com")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs.get("timeout") == 35
