import ast
from textwrap import dedent

import pytest

from smolagents.default_tools import DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool
from smolagents.tool_validation import MethodChecker, validate_tool_attributes
from smolagents.tools import Tool, tool


UNDEFINED_VARIABLE = "undefined_variable"


@pytest.mark.parametrize("tool_class", [DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool])
def test_validate_tool_attributes_with_default_tools(tool_class):
    assert validate_tool_attributes(tool_class) is None, f"failed for {tool_class.name} tool"


class ValidTool(Tool):
    name = "valid_tool"
    description = "A valid tool"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"
    simple_attr = "string"
    dict_attr = {"key": "value"}

    def __init__(self, optional_param="default"):
        super().__init__()
        self.param = optional_param

    def forward(self, input: str) -> str:
        return input.upper()


@tool
def valid_tool_function(input: str) -> str:
    """A valid tool function.

    Args:
        input (str): Input string.
    """
    return input.upper()


@pytest.mark.parametrize("tool_class", [ValidTool, valid_tool_function.__class__])
def test_validate_tool_attributes_valid(tool_class):
    assert validate_tool_attributes(tool_class) is None


class InvalidToolName(Tool):
    name = "invalid tool name"
    description = "Tool with invalid name"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, input: str) -> str:
        return input


class InvalidToolComplexAttrs(Tool):
    name = "invalid_tool"
    description = "Tool with complex class attributes"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"
    complex_attr = [x for x in range(3)]  # Complex class attribute

    def __init__(self):
        super().__init__()

    def forward(self, input: str) -> str:
        return input


class InvalidToolRequiredParams(Tool):
    name = "invalid_tool"
    description = "Tool with required params"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self, required_param, kwarg1=1):  # No default value
        super().__init__()
        self.param = required_param

    def forward(self, input: str) -> str:
        return input


class InvalidToolNonLiteralDefaultParam(Tool):
    name = "invalid_tool"
    description = "Tool with non-literal default parameter value"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self, default_param=UNDEFINED_VARIABLE):  # UNDEFINED_VARIABLE as default is non-literal
        super().__init__()
        self.default_param = default_param

    def forward(self, input: str) -> str:
        return input


class InvalidToolUndefinedNames(Tool):
    name = "invalid_tool"
    description = "Tool with undefined names"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return UNDEFINED_VARIABLE  # Undefined name


@pytest.mark.parametrize(
    "tool_class, expected_error",
    [
        (
            InvalidToolName,
            "Class attribute 'name' must be a valid Python identifier and not a reserved keyword, found 'invalid tool name'",
        ),
        (InvalidToolComplexAttrs, "Complex attributes should be defined in __init__, not as class attributes"),
        (InvalidToolRequiredParams, "Parameters in __init__ must have default values, found required parameters"),
        (
            InvalidToolNonLiteralDefaultParam,
            "Parameters in __init__ must have literal default values, found non-literal defaults",
        ),
        (InvalidToolUndefinedNames, "Name 'UNDEFINED_VARIABLE' is undefined"),
    ],
)
def test_validate_tool_attributes_exceptions(tool_class, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        validate_tool_attributes(tool_class)


class MultipleAssignmentsTool(Tool):
    name = "multiple_assignments_tool"
    description = "Tool with multiple assignments"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def __init__(self):
        super().__init__()

    def forward(self, input: str) -> str:
        a, b = "1", "2"
        return a + b


def test_validate_tool_attributes_multiple_assignments():
    validate_tool_attributes(MultipleAssignmentsTool)


@tool
def tool_function_with_multiple_assignments(input: str) -> str:
    """A valid tool function.

    Args:
        input (str): Input string.
    """
    a, b = "1", "2"
    return input.upper() + a + b


@pytest.mark.parametrize("tool_instance", [MultipleAssignmentsTool(), tool_function_with_multiple_assignments])
def test_tool_to_dict_validation_with_multiple_assignments(tool_instance):
    tool_instance.to_dict()


class TestMethodChecker:
    def test_multiple_assignments(self):
        source_code = dedent(
            """
            def forward(self) -> str:
                a, b = "1", "2"
                return a + b
            """
        )
        method_checker = MethodChecker(set())
        method_checker.visit(ast.parse(source_code))
        assert method_checker.errors == []
