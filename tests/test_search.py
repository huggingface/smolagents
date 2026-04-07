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


from unittest.mock import MagicMock, patch

from smolagents import DuckDuckGoSearchTool
from smolagents.default_tools import WebSearchTool

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


class TestWebSearchToolTimeout:
    """Tests for the configurable timeout parameter on WebSearchTool."""

    def test_default_timeout_is_30_seconds(self):
        tool = WebSearchTool()
        assert tool.timeout == 30

    def test_custom_timeout_is_stored(self):
        tool = WebSearchTool(timeout=60)
        assert tool.timeout == 60

    def test_timeout_passed_to_duckduckgo_request(self):
        tool = WebSearchTool(engine="duckduckgo", timeout=45)
        mock_response = MagicMock()
        mock_response.text = "<html></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.search_duckduckgo("test query")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs["timeout"] == 45

    def test_timeout_passed_to_bing_request(self):
        tool = WebSearchTool(engine="bing", timeout=15)
        mock_response = MagicMock()
        mock_response.text = "<rss><channel></channel></rss>"
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            tool.search_bing("test query")
            mock_get.assert_called_once()
            assert mock_get.call_args.kwargs["timeout"] == 15
