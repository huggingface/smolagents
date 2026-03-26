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
from unittest.mock import MagicMock, patch

import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import (
    CrwCrawlTool,
    CrwScrapeTool,
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


class TestCrwScrapeTool:
    def test_initialization_defaults(self):
        tool = CrwScrapeTool()
        assert tool.api_url == "http://localhost:3000"
        assert tool.formats == ["markdown"]
        assert tool.only_main_content is True

    def test_initialization_custom(self):
        tool = CrwScrapeTool(
            api_url="http://myserver:4000/",
            api_key="test-key",
            only_main_content=False,
            formats=["markdown", "html"],
        )
        assert tool.api_url == "http://myserver:4000"
        assert tool.api_key == "test-key"
        assert tool.only_main_content is False
        assert tool.formats == ["markdown", "html"]

    @patch("requests.post")
    def test_scrape_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "# Hello World\n\nThis is content.",
                "metadata": {
                    "title": "Hello World",
                    "sourceURL": "https://example.com",
                },
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwScrapeTool()
        result = tool("https://example.com")
        assert "Hello World" in result
        assert "https://example.com" in result

    @patch("requests.post")
    def test_scrape_with_css_selector(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {"markdown": "Selected content", "metadata": {}},
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwScrapeTool()
        tool("https://example.com", css_selector="article.main")

        call_payload = mock_post.call_args[1]["json"]
        assert call_payload["cssSelector"] == "article.main"

    @patch("requests.post")
    def test_scrape_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "invalid_url",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwScrapeTool()
        result = tool("not-a-url")
        assert "failed" in result.lower()

    @patch("requests.post")
    def test_scrape_content_truncation(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "markdown": "x" * 50000,
                "metadata": {"title": "Big Page"},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwScrapeTool(max_output_length=1000)
        result = tool("https://example.com")
        assert "truncated" in result.lower()
        assert len(result) <= 1000

    @patch("requests.post")
    def test_scrape_links_format(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {
                "links": ["https://example.com/a", "https://example.com/b"],
                "metadata": {"title": "Links Page"},
            },
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwScrapeTool(formats=["links"])
        result = tool("https://example.com")
        assert "https://example.com/a" in result
        assert "https://example.com/b" in result


class TestCrwCrawlTool:
    def test_initialization_defaults(self):
        tool = CrwCrawlTool()
        assert tool.api_url == "http://localhost:3000"
        assert tool.max_depth == 2
        assert tool.max_pages == 10

    @patch("requests.get")
    @patch("requests.post")
    def test_crawl_success(self, mock_post, mock_get):
        # Mock crawl start
        mock_start = MagicMock()
        mock_start.json.return_value = {"success": True, "id": "test-crawl-id"}
        mock_start.raise_for_status = MagicMock()
        mock_post.return_value = mock_start

        # Mock status check — completed immediately
        mock_status = MagicMock()
        mock_status.json.return_value = {
            "status": "completed",
            "total": 1,
            "data": [
                {
                    "markdown": "# Page 1 content",
                    "metadata": {
                        "title": "Page 1",
                        "sourceURL": "https://example.com",
                    },
                }
            ],
        }
        mock_status.raise_for_status = MagicMock()
        mock_get.return_value = mock_status

        tool = CrwCrawlTool(poll_interval=0.01)
        result = tool("https://example.com")
        assert "Page 1" in result
        assert "https://example.com" in result

    @patch("requests.post")
    def test_crawl_start_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "invalid_url",
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = CrwCrawlTool()
        result = tool("bad-url")
        assert "failed" in result.lower()

    @patch("requests.post")
    def test_scrape_timeout(self, mock_post):
        import requests

        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

        tool = CrwScrapeTool()
        result = tool("https://example.com")
        assert "timed out" in result.lower()

    @patch("requests.get")
    @patch("requests.post")
    def test_crawl_failed_status(self, mock_post, mock_get):
        mock_start = MagicMock()
        mock_start.json.return_value = {"success": True, "id": "crawl-fail-id"}
        mock_start.raise_for_status = MagicMock()
        mock_post.return_value = mock_start

        mock_status = MagicMock()
        mock_status.json.return_value = {"status": "failed"}
        mock_status.raise_for_status = MagicMock()
        mock_get.return_value = mock_status

        tool = CrwCrawlTool(poll_interval=0.01)
        result = tool("https://example.com")
        assert "failed" in result.lower()

    @patch("requests.get")
    @patch("requests.post")
    def test_crawl_intermediate_then_completed(self, mock_post, mock_get):
        mock_start = MagicMock()
        mock_start.json.return_value = {"success": True, "id": "crawl-poll-id"}
        mock_start.raise_for_status = MagicMock()
        mock_post.return_value = mock_start

        # First poll returns "scraping", second returns "completed"
        mock_scraping = MagicMock()
        mock_scraping.json.return_value = {"status": "scraping"}
        mock_scraping.raise_for_status = MagicMock()

        mock_completed = MagicMock()
        mock_completed.json.return_value = {
            "status": "completed",
            "total": 1,
            "data": [
                {
                    "markdown": "# Done",
                    "metadata": {"title": "Done", "sourceURL": "https://example.com"},
                }
            ],
        }
        mock_completed.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_scraping, mock_completed]

        tool = CrwCrawlTool(poll_interval=0.01)
        result = tool("https://example.com")
        assert "Done" in result
