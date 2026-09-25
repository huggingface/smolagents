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
from unittest.mock import Mock, patch

import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import (
    AgentLookupTool,
    AgentSearchTool,
    AgentVerifyTool,
    DuckDuckGoSearchTool,
    MarketplaceSearchTool,
    PythonInterpreterTool,
    SpeechToTextTool,
    TrustGateTool,
    VisitWebpageTool,
    WebSearchTool,
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


class TestAgentFolioTools:
    def test_lookup_formats_profile_from_agentfolio(self):
        tool = AgentLookupTool(base_url="https://agentfolio.test/api")
        mock_profile = {
            "id": "agent_braintest",
            "name": "brainTEST",
            "bio": "testing agent",
            "skills": ["testing"],
            "trustScore": 800,
            "verificationLevelName": "Verified",
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_profile
            mock_get.return_value.raise_for_status = lambda: None
            result = tool("agent_braintest")

        mock_get.assert_called_once_with("https://agentfolio.test/api/profile/agent_braintest", timeout=10)
        assert "brainTEST" in result
        assert "testing" in result
        assert "800" in result

    def test_search_uses_agents_endpoint_and_filters_by_trust(self):
        tool = AgentSearchTool(base_url="https://agentfolio.test/api")
        mock_agents = {
            "agents": [
                {"id": "agent_low", "name": "Low", "trustScore": 10, "skills": ["solana"]},
                {"id": "agent_high", "name": "High", "trustScore": 800, "skills": ["solana"]},
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_agents
            mock_get.return_value.raise_for_status = lambda: None
            result = tool(query="solana", min_trust=500)

        mock_get.assert_called_once_with("https://agentfolio.test/api/agents", params={"search": "solana"}, timeout=10)
        assert "agent_high" in result
        assert "agent_low" not in result

    def test_search_handles_top_level_agent_list_response(self):
        tool = AgentSearchTool(base_url="https://agentfolio.test/api")
        mock_agents = [
            {"id": "agent_low", "name": "Low", "trustScore": 10, "skills": ["solana"]},
            {"id": "agent_high", "name": "High", "trust_score": {"reputationScore": 800}, "skills": ["solana"]},
        ]

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_agents
            mock_get.return_value.raise_for_status = lambda: None
            result = tool(query="solana", min_trust=500)

        assert "agent_high" in result
        assert "agent_low" not in result

    def test_verify_fetches_profile_and_endorsements(self):
        tool = AgentVerifyTool(base_url="https://agentfolio.test/api")
        mock_profile = {
            "id": "agent_braintest",
            "name": "brainTEST",
            "trustScore": 80,
            "verification": '{"tier":"verified","score":0}',
            "verification_data": {"solana": {"verified": True}, "github": {"verified": False}},
        }

        with patch("requests.get") as mock_get:
            profile_response = mock_get.return_value
            endorsements_response = Mock()
            profile_response.status_code = 200
            profile_response.json.return_value = mock_profile
            profile_response.raise_for_status = lambda: None
            endorsements_response.status_code = 200
            endorsements_response.json.return_value = {"endorsements": [{"id": "endorsement-1"}], "total": 1}
            mock_get.side_effect = [profile_response, endorsements_response]
            result = tool("agent_braintest")

        assert mock_get.call_args_list[0].args[0] == "https://agentfolio.test/api/profile/agent_braintest"
        assert mock_get.call_args_list[1].args[0] == "https://agentfolio.test/api/profile/agent_braintest/endorsements"
        assert '"trust_score": 80' in result
        assert '"tier": "verified"' in result
        assert '"endorsement_count": 1' in result
        assert "solana" in result
        assert "github" not in result

    def test_trust_gate_uses_profile_trust_score(self):
        tool = TrustGateTool(base_url="https://agentfolio.test/api")
        mock_profile = {
            "id": "agent_braintest",
            "trust_score": {"reputationScore": 800},
            "verification": '{"score":0}',
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = mock_profile
            mock_get.return_value.raise_for_status = lambda: None
            result = tool(agent_id="agent_braintest", min_trust=500)

        assert '"passed": true' in result
        assert '"trust_score": 800' in result
        assert '"required": 500' in result

    def test_marketplace_search_filters_jobs_by_category(self):
        tool = MarketplaceSearchTool(base_url="https://agentfolio.test/api")
        mock_jobs = {
            "jobs": [
                {
                    "id": "job_1",
                    "title": "QA task",
                    "category": "testing",
                    "budgetAmount": 5,
                    "budgetCurrency": "USDC",
                },
                {
                    "id": "job_2",
                    "title": "Design task",
                    "category": "design",
                    "budgetAmount": 9,
                    "budgetCurrency": "USDC",
                },
            ]
        }

        with patch("requests.get") as mock_get:
            mock_get.return_value.json.return_value = mock_jobs
            mock_get.return_value.raise_for_status = lambda: None
            result = tool(category="testing")

        mock_get.assert_called_once_with("https://agentfolio.test/api/marketplace/jobs", timeout=10)
        assert "QA task" in result
        assert "Design task" not in result

    def test_agentfolio_tools_are_not_enabled_by_default(self):
        from smolagents.default_tools import TOOL_MAPPING

        assert "agent_lookup" not in TOOL_MAPPING
        assert "agent_search" not in TOOL_MAPPING
        assert "agent_verify" not in TOOL_MAPPING
        assert "trust_gate" not in TOOL_MAPPING
        assert "marketplace_search" not in TOOL_MAPPING


class TestWebSearchToolExa:
    """Tests for the Exa engine in WebSearchTool."""

    def test_exa_missing_api_key(self):
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="EXA_API_KEY"):
                tool("test query")

    def test_exa_search_results(self):
        mock_response_data = {
            "results": [
                {
                    "title": "Exa AI",
                    "url": "https://exa.ai",
                    "highlights": ["Exa is a search engine for AI."],
                },
                {
                    "title": "Hugging Face",
                    "url": "https://huggingface.co",
                    "highlights": ["The AI community", "building the future."],
                },
            ]
        }
        tool = WebSearchTool(engine="exa", max_results=2)
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test query")

        assert "## Search Results" in result
        assert "[Exa AI](https://exa.ai)" in result
        assert "[Hugging Face](https://huggingface.co)" in result
        assert "Exa is a search engine for AI." in result
        assert "The AI community building the future." in result

        # Verify the API was called with correct headers and payload
        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["headers"]["x-exa-integration"] == "smolagents"
        assert call_kwargs.kwargs["json"]["numResults"] == 2
        assert call_kwargs.kwargs["json"]["contents"] == {"highlights": True}

    def test_exa_no_results(self):
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = {"results": []}
                mock_post.return_value.raise_for_status = lambda: None
                with pytest.raises(Exception, match="No results found"):
                    tool("obscure query")

    def test_exa_empty_highlights(self):
        mock_response_data = {
            "results": [
                {
                    "title": "No Highlights Page",
                    "url": "https://example.com",
                }
            ]
        }
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test")

        assert "[No Highlights Page](https://example.com)" in result

    def test_exa_null_highlights(self):
        mock_response_data = {
            "results": [
                {
                    "title": "Null Highlights Page",
                    "url": "https://example.com",
                    "highlights": None,
                }
            ]
        }
        tool = WebSearchTool(engine="exa")
        with patch.dict("os.environ", {"EXA_API_KEY": "test-key"}):
            with patch("requests.post") as mock_post:
                mock_post.return_value.json.return_value = mock_response_data
                mock_post.return_value.raise_for_status = lambda: None
                result = tool("test")

        assert "[Null Highlights Page](https://example.com)" in result


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
