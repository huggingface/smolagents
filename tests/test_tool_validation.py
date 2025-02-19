import pytest

from smolagents.default_tools import DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool
from smolagents.tool_validation import validate_tool_attributes
from smolagents.tools import Tool


@pytest.mark.parametrize("tool_class", [DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool])
def test_validate_tool_attributes(tool_class):
    assert validate_tool_attributes(tool_class) is None, f"failed for {tool_class.name} tool"


class DummyTool(Tool):
    name = "DummyTool"
    description = "A dummy tool for testing"
    inputs = {"output_string": {"type": "string", "description": "The string to output."}}
    output_type = "string"

    def __init__(self, tool_class=Tool):
        super().__init__()
        self.tool_class = tool_class

    def forward(self, output_string: str) -> str:
        return "dummy output: " + output_string


def test_validate_tool_attributes_exceptions():
    with pytest.raises(ValueError):
        validate_tool_attributes(DummyTool)
