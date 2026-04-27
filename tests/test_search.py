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


from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from smolagents import DuckDuckGoSearchTool, ExaSearchTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_run_all


class TestDuckDuckGoSearchTool(ToolTesterMixin):
    def setup_method(self):
        self.tool = DuckDuckGoSearchTool()
        self.tool.setup()

    @require_run_all
    def test_exact_match_arg(self):
        result = self.tool("Agents")
        assert isinstance(result, str)

    @require_run_all
    def test_agent_type_output(self):
        super().test_agent_type_output()


def _make_exa_result(
    title="Example Title",
    url="https://example.com/",
    published_date=None,
    author=None,
    text=None,
    summary=None,
    highlights=None,
):
    return SimpleNamespace(
        title=title,
        url=url,
        published_date=published_date,
        author=author,
        text=text,
        summary=summary,
        highlights=highlights,
    )


def _make_exa_client(results):
    """Build a fake Exa client whose `.search()` returns a SimpleNamespace with `results`."""
    client = MagicMock()
    client.headers = {}
    client.search.return_value = SimpleNamespace(results=results)
    return client


class TestExaSearchTool:
    def _build_tool(self, results, **tool_kwargs):
        fake_client = _make_exa_client(results)
        # `from exa_py import Exa` runs inside ExaSearchTool.__init__; patch the source
        # module so the constructor receives our fake.
        with patch("exa_py.Exa", return_value=fake_client):
            tool = ExaSearchTool(api_key="test-key", **tool_kwargs)
        return tool, fake_client

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="EXA_API_KEY"):
            with patch("exa_py.Exa"):
                ExaSearchTool()

    def test_integration_header_set(self):
        _tool, client = self._build_tool([])
        assert client.headers["x-exa-integration"] == "smolagents"

    def test_invalid_contents_shortcut_raises(self):
        with patch("exa_py.Exa", return_value=_make_exa_client([])):
            with pytest.raises(ValueError, match="Invalid `contents` shortcut"):
                ExaSearchTool(api_key="test-key", contents="bogus")

    def test_no_results_returns_empty_message(self):
        tool, _ = self._build_tool([])
        out = tool("anything")
        assert isinstance(out, str)
        assert "No results found" in out

    def test_formats_highlights_when_present(self):
        results = [
            _make_exa_result(
                title="Hugging Face",
                url="https://huggingface.co/",
                published_date="2024-01-15",
                author="HF Team",
                highlights=["Open-source AI platform.", "Hub for models and datasets."],
                text="Full page text",
                summary="A summary",
            )
        ]
        tool, _ = self._build_tool(results)
        out = tool("hugging face")
        assert "[Hugging Face](https://huggingface.co/)" in out
        assert "Published: 2024-01-15" in out
        assert "Author: HF Team" in out
        # Highlights take precedence over summary/text.
        assert "Open-source AI platform." in out
        assert "Hub for models and datasets." in out
        assert "A summary" not in out
        assert "Full page text" not in out

    def test_falls_back_to_summary_when_no_highlights(self):
        results = [_make_exa_result(summary="Short summary text", text="Full text")]
        tool, _ = self._build_tool(results)
        out = tool("query")
        assert "Short summary text" in out
        assert "Full text" not in out

    def test_falls_back_to_text_when_no_highlights_or_summary(self):
        long_text = "x" * 600
        results = [_make_exa_result(text=long_text)]
        tool, _ = self._build_tool(results)
        out = tool("query")
        # First 500 chars of text plus an ellipsis appear in the output.
        assert "x" * 500 in out
        assert "…" in out

    def test_handles_result_with_no_content(self):
        results = [_make_exa_result(title="Bare", url="https://bare.example/")]
        tool, _ = self._build_tool(results)
        out = tool("query")
        assert "[Bare](https://bare.example/)" in out

    def test_forward_passes_filters_to_client(self):
        results = [_make_exa_result()]
        tool, client = self._build_tool(
            results,
            max_results=3,
            search_type="neural",
            category="research paper",
            include_domains=["arxiv.org"],
            exclude_domains=["spam.example"],
            include_text=["transformer"],
            exclude_text=["draft"],
            start_published_date="2024-01-01",
            end_published_date="2024-12-31",
            contents="text",
        )
        tool("transformer architectures")
        client.search.assert_called_once()
        args, kwargs = client.search.call_args
        assert args == ("transformer architectures",)
        assert kwargs["num_results"] == 3
        assert kwargs["type"] == "neural"
        assert kwargs["category"] == "research paper"
        assert kwargs["include_domains"] == ["arxiv.org"]
        assert kwargs["exclude_domains"] == ["spam.example"]
        assert kwargs["include_text"] == ["transformer"]
        assert kwargs["exclude_text"] == ["draft"]
        assert kwargs["start_published_date"] == "2024-01-01"
        assert kwargs["end_published_date"] == "2024-12-31"
        assert kwargs["contents"] == {"text": True}

    def test_custom_contents_dict_passed_through(self):
        results = [_make_exa_result()]
        custom = {"text": {"maxCharacters": 200}, "highlights": True}
        tool, client = self._build_tool(results, contents=custom)
        tool("query")
        _, kwargs = client.search.call_args
        assert kwargs["contents"] == custom
