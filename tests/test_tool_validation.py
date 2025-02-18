import pytest

from smolagents.default_tools import DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool
from smolagents.tool_validation import validate_tool_attributes


@pytest.mark.parametrize("tool_class", [DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool])
def test_validate_tool_attributes(tool_class):
    assert validate_tool_attributes(tool_class) is None, f"failed for {tool_class.name} tool"
