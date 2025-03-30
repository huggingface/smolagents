from typing import Optional

import pytest

from smolagents.tools import Tool, tool


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


@pytest.fixture
def boolean_default_tool_function():
    @tool
    def boolean_default_tool(text: str, flag: bool = False) -> str:
        """
        A tool with a boolean default parameter.

        Args:
            text: Input text
            flag: Boolean flag with default value
        """
        return f"Text: {text}, Flag: {flag}"

    return boolean_default_tool


@pytest.fixture
def optional_input_tool_class():
    class OptionalInputTool(Tool):
        name = "optional_input_tool"
        description = "A tool with an optional input parameter"
        inputs = {
            "required_text": {"type": "string", "description": "Required input text"},
            "optional_text": {"type": "string", "description": "Optional input text", "nullable": True},
        }
        output_type = "string"

        def forward(self, required_text: str, optional_text: Optional[str] = None) -> str:
            if optional_text:
                return f"{required_text} + {optional_text}"
            return required_text

    return OptionalInputTool()


@pytest.fixture
def optional_input_tool_function():
    @tool
    def optional_input_tool(required_text: str, optional_text: Optional[str] = None) -> str:
        """
        A tool with an optional input parameter.

        Args:
            required_text: Required input text
            optional_text: Optional input text
        """
        if optional_text:
            return f"{required_text} + {optional_text}"
        return required_text

    return optional_input_tool
