import ast
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from typing import Optional

import pytest

from smolagents.default_tools import (
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    SpeechToTextTool,
    VisitWebpageTool,
    WebSearchTool,
)
from smolagents.tool_validation import MethodChecker, validate_tool_attributes
from smolagents.tools import Tool, tool


UNDEFINED_VARIABLE = "undefined_variable"


@pytest.mark.parametrize(
    "tool_class", [DuckDuckGoSearchTool, GoogleSearchTool, SpeechToTextTool, VisitWebpageTool, WebSearchTool]
)
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


# ---------------------------------------------------------------------------
# Issue #2736 Regression Test Suite: Deterministic Validation & Optimization Invariance
# ---------------------------------------------------------------------------


def run_in_python_subprocess(code: str, optimize_flags: list[str]) -> subprocess.CompletedProcess:
    cmd = [sys.executable] + optimize_flags + ["-c", code]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


class NonDictInputTool(Tool):
    name = "non_dict_input_tool"
    description = "Tool with non-dict input"
    inputs = {"input": "not a dict"}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


class BrokenTool(Tool):
    name = "broken_tool"
    description = "Tool with missing description in inputs"
    inputs = {"input": {"type": "string"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


class MissingTypeTool(Tool):
    name = "missing_type_tool"
    description = "Tool with missing type in inputs"
    inputs = {"input": {"description": "input without type"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


class InvalidOutputTypeTool(Tool):
    name = "invalid_output_type_tool"
    description = "Tool with invalid output_type"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "invalid_output_type"

    def forward(self, input: str) -> str:
        return input


class NullableInInputsOnlyTool(Tool):
    name = "nullable_in_inputs_only_tool"
    description = "Tool with nullable in inputs but not in forward signature"
    inputs = {"input": {"type": "string", "description": "input", "nullable": True}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


class NullableInSigOnlyTool(Tool):
    name = "nullable_in_sig_only_tool"
    description = "Tool with nullable in signature but not in inputs"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def forward(self, input: Optional[str] = None) -> str:
        return input or ""


class SignatureMismatchTool(Tool):
    name = "signature_mismatch_tool"
    description = "Tool with parameter mismatch between inputs and forward signature"
    inputs = {"input": {"type": "string", "description": "input"}}
    output_type = "string"

    def forward(self, wrong_param: str) -> str:
        return wrong_param


class RegressionValidTool(Tool):
    name = "regression_valid_tool"
    description = "Valid tool for regression testing"
    inputs = {
        "text": {"type": "string", "description": "Text input"},
        "flag": {"type": "boolean", "description": "Optional boolean flag", "nullable": True},
    }
    output_type = "string"

    def forward(self, text: str, flag: Optional[bool] = None) -> str:
        return text if flag else text.lower()


CODE_NON_DICT_INPUT = dedent(
    """
    from smolagents.tools import Tool

    class NonDictInputTool(Tool):
        name = "non_dict_input_tool"
        description = "Tool with non-dict input"
        inputs = {"input": "not a dict"}
        output_type = "string"

        def forward(self, input: str) -> str:
            return input

    NonDictInputTool()
    """
)

CODE_MISSING_DESCRIPTION = dedent(
    """
    from smolagents.tools import Tool

    class BrokenTool(Tool):
        name = "broken_tool"
        description = "Tool with missing description in inputs"
        inputs = {"input": {"type": "string"}}
        output_type = "string"

        def forward(self, input: str) -> str:
            return input

    BrokenTool()
    """
)

CODE_MISSING_TYPE = dedent(
    """
    from smolagents.tools import Tool

    class MissingTypeTool(Tool):
        name = "missing_type_tool"
        description = "Tool with missing type in inputs"
        inputs = {"input": {"description": "input without type"}}
        output_type = "string"

        def forward(self, input: str) -> str:
            return input

    MissingTypeTool()
    """
)

CODE_INVALID_OUTPUT_TYPE = dedent(
    """
    from smolagents.tools import Tool

    class InvalidOutputTypeTool(Tool):
        name = "invalid_output_type_tool"
        description = "Tool with invalid output_type"
        inputs = {"input": {"type": "string", "description": "input"}}
        output_type = "invalid_output_type"

        def forward(self, input: str) -> str:
            return input

    InvalidOutputTypeTool()
    """
)

CODE_NULLABLE_IN_INPUTS_ONLY = dedent(
    """
    from smolagents.tools import Tool

    class NullableInInputsOnlyTool(Tool):
        name = "nullable_in_inputs_only_tool"
        description = "Tool with nullable in inputs but not in forward signature"
        inputs = {"input": {"type": "string", "description": "input", "nullable": True}}
        output_type = "string"

        def forward(self, input: str) -> str:
            return input

    NullableInInputsOnlyTool()
    """
)

CODE_NULLABLE_IN_SIG_ONLY = dedent(
    """
    from typing import Optional
    from smolagents.tools import Tool

    class NullableInSigOnlyTool(Tool):
        name = "nullable_in_sig_only_tool"
        description = "Tool with nullable in signature but not in inputs"
        inputs = {"input": {"type": "string", "description": "input"}}
        output_type = "string"

        def forward(self, input: Optional[str] = None) -> str:
            return input or ""

    NullableInSigOnlyTool()
    """
)

CODE_SIGNATURE_MISMATCH = dedent(
    """
    from smolagents.tools import Tool

    class SignatureMismatchTool(Tool):
        name = "signature_mismatch_tool"
        description = "Tool with parameter mismatch between inputs and forward signature"
        inputs = {"input": {"type": "string", "description": "input"}}
        output_type = "string"

        def forward(self, wrong_param: str) -> str:
            return wrong_param

    SignatureMismatchTool()
    """
)

CODE_VALID_TOOL = dedent(
    """
    from typing import Optional
    from smolagents.tools import Tool

    class RegressionValidTool(Tool):
        name = "regression_valid_tool"
        description = "Valid tool for regression testing"
        inputs = {
            "text": {"type": "string", "description": "Text input"},
            "flag": {"type": "boolean", "description": "Optional boolean flag", "nullable": True},
        }
        output_type = "string"

        def forward(self, text: str, flag: Optional[bool] = None) -> str:
            return text if flag else text.lower()

    tool = RegressionValidTool()
    assert tool.name == "regression_valid_tool"
    assert tool("HELLO", flag=False) == "hello"
    """
)


# Feature 1: Non-dict input validation (R1)
def test_non_dict_input_normal():
    with pytest.raises(TypeError, match="Input 'input' should be a dictionary."):
        NonDictInputTool()


def test_non_dict_input_opt():
    res = run_in_python_subprocess(CODE_NON_DICT_INPUT, ["-O"])
    assert res.returncode != 0
    assert "TypeError" in res.stderr
    assert "Input 'input' should be a dictionary." in res.stderr


def test_non_dict_input_opt_oo():
    res = run_in_python_subprocess(CODE_NON_DICT_INPUT, ["-OO"])
    assert res.returncode != 0
    assert "TypeError" in res.stderr
    assert "Input 'input' should be a dictionary." in res.stderr


# Feature 2: Missing description in input (R1, R2)
def test_missing_description_normal():
    with pytest.raises(
        ValueError,
        match=r"Input 'input' should have keys 'type' and 'description', has only \['type'\]\.",
    ):
        BrokenTool()


def test_missing_description_opt():
    res = run_in_python_subprocess(CODE_MISSING_DESCRIPTION, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "Input 'input' should have keys 'type' and 'description'" in res.stderr


def test_missing_description_opt_oo():
    res = run_in_python_subprocess(CODE_MISSING_DESCRIPTION, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "Input 'input' should have keys 'type' and 'description'" in res.stderr


# Feature 3: Missing type in input (R1, R2)
def test_missing_type_normal():
    with pytest.raises(
        ValueError,
        match=r"Input 'input' should have keys 'type' and 'description', has only \['description'\]\.",
    ):
        MissingTypeTool()


def test_missing_type_opt():
    res = run_in_python_subprocess(CODE_MISSING_TYPE, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "Input 'input' should have keys 'type' and 'description'" in res.stderr


def test_missing_type_opt_oo():
    res = run_in_python_subprocess(CODE_MISSING_TYPE, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "Input 'input' should have keys 'type' and 'description'" in res.stderr


# Feature 4: Invalid output_type (R1, R2)
def test_invalid_output_type_normal():
    with pytest.raises(
        ValueError,
        match="Tool 'invalid_output_type_tool': output_type 'invalid_output_type' must be one of",
    ):
        InvalidOutputTypeTool()


def test_invalid_output_type_opt():
    res = run_in_python_subprocess(CODE_INVALID_OUTPUT_TYPE, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "output_type 'invalid_output_type' must be one of" in res.stderr


def test_invalid_output_type_opt_oo():
    res = run_in_python_subprocess(CODE_INVALID_OUTPUT_TYPE, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "output_type 'invalid_output_type' must be one of" in res.stderr


# Feature 5: Nullable in inputs only (R1, R2)
def test_nullable_in_inputs_normal():
    with pytest.raises(
        ValueError,
        match="Nullable argument 'input' in inputs should have key 'nullable' set to True in function signature.",
    ):
        NullableInInputsOnlyTool()


def test_nullable_in_inputs_opt():
    res = run_in_python_subprocess(CODE_NULLABLE_IN_INPUTS_ONLY, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "Nullable argument 'input' in inputs should have key 'nullable' set to True in function signature."
        in res.stderr
    )


def test_nullable_in_inputs_opt_oo():
    res = run_in_python_subprocess(CODE_NULLABLE_IN_INPUTS_ONLY, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "Nullable argument 'input' in inputs should have key 'nullable' set to True in function signature."
        in res.stderr
    )


# Feature 6: Nullable in signature only (R1, R2)
def test_nullable_in_sig_normal():
    with pytest.raises(
        ValueError,
        match="Nullable argument 'input' in function signature should have key 'nullable' set to True in inputs.",
    ):
        NullableInSigOnlyTool()


def test_nullable_in_sig_opt():
    res = run_in_python_subprocess(CODE_NULLABLE_IN_SIG_ONLY, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "Nullable argument 'input' in function signature should have key 'nullable' set to True in inputs."
        in res.stderr
    )


def test_nullable_in_sig_opt_oo():
    res = run_in_python_subprocess(CODE_NULLABLE_IN_SIG_ONLY, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "Nullable argument 'input' in function signature should have key 'nullable' set to True in inputs."
        in res.stderr
    )


# Feature 7: Signature parameter mismatch (R1, R2)
def test_signature_mismatch_normal():
    with pytest.raises(
        ValueError,
        match=r"In tool 'signature_mismatch_tool', 'forward' method parameters were \{'wrong_param'\}, but expected \{'input'\}\.",
    ):
        SignatureMismatchTool()


def test_signature_mismatch_opt():
    res = run_in_python_subprocess(CODE_SIGNATURE_MISMATCH, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "In tool 'signature_mismatch_tool', 'forward' method parameters were {'wrong_param'}, but expected {'input'}."
        in res.stderr
    )


def test_signature_mismatch_opt_oo():
    res = run_in_python_subprocess(CODE_SIGNATURE_MISMATCH, ["-OO"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert (
        "In tool 'signature_mismatch_tool', 'forward' method parameters were {'wrong_param'}, but expected {'input'}."
        in res.stderr
    )


# Feature 8: Valid tool instantiation (R3)
def test_valid_tool_normal():
    tool_inst = RegressionValidTool()
    assert tool_inst.name == "regression_valid_tool"
    assert tool_inst("HELLO", flag=False) == "hello"
    assert tool_inst("WORLD", flag=True) == "WORLD"


def test_valid_tool_opt():
    res = run_in_python_subprocess(CODE_VALID_TOOL, ["-O"])
    assert res.returncode == 0, f"Failed with stderr: {res.stderr}"


def test_valid_tool_opt_oo():
    res = run_in_python_subprocess(CODE_VALID_TOOL, ["-OO"])
    assert res.returncode == 0, f"Failed with stderr: {res.stderr}"


# Additional Edge Cases & Adversarial Verification


class UnauthorizedInputTypeTool(Tool):
    name = "unauthorized_input_type_tool"
    description = "Tool with unauthorized input type"
    inputs = {"input": {"type": "unknown_type", "description": "input with unknown type"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


def test_unauthorized_input_type_normal():
    with pytest.raises(ValueError, match="must be one of"):
        UnauthorizedInputTypeTool()


def test_unauthorized_input_type_opt():
    code = dedent(
        """
        from smolagents.tools import Tool

        class UnauthorizedInputTypeTool(Tool):
            name = "unauthorized_input_type_tool"
            description = "Tool with unauthorized input type"
            inputs = {"input": {"type": "unknown_type", "description": "input with unknown type"}}
            output_type = "string"

            def forward(self, input: str) -> str:
                return input

        UnauthorizedInputTypeTool()
        """
    )
    res = run_in_python_subprocess(code, ["-O"])
    assert res.returncode != 0
    assert "ValueError" in res.stderr
    assert "must be one of" in res.stderr


class NonStringListInputTypeTool(Tool):
    name = "non_string_list_input_type_tool"
    description = "Tool with non-string list in type"
    inputs = {"input": {"type": [123], "description": "input with integer in type list"}}
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


def test_non_string_list_input_type_normal():
    with pytest.raises(
        TypeError,
        match="when type is a list, all elements must be strings",
    ):
        NonStringListInputTypeTool()


def test_non_string_list_input_type_opt():
    code = dedent(
        """
        from smolagents.tools import Tool

        class NonStringListInputTypeTool(Tool):
            name = "non_string_list_input_type_tool"
            description = "Tool with non-string list in type"
            inputs = {"input": {"type": [123], "description": "input with integer in type list"}}
            output_type = "string"

            def forward(self, input: str) -> str:
                return input

        NonStringListInputTypeTool()
        """
    )
    res = run_in_python_subprocess(code, ["-O"])
    assert res.returncode != 0
    assert "TypeError" in res.stderr
    assert "when type is a list, all elements must be strings" in res.stderr
