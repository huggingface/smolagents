import pytest

from smolagents.tools import Tool


@pytest.fixture
def boolean_default_tool_class():
    class BooleanDefaultTool(Tool):
        name = "boolean_default_tool"
        description = "A tool with a boolean default parameter"
        inputs = {
            "text": {"type": "string", "description": "Input text"},
            "flag": {"type": "boolean", "description": "Boolean flag with default value", "nullable": True},
        }
        output_type = "string"

        def forward(self, text: str, flag: bool = False) -> str:
            return f"Text: {text}, Flag: {flag}"

    return BooleanDefaultTool()
