import pytest

from smolagents.tools import tool, validate_tool_arguments


@pytest.mark.parametrize(
    "tool_input_type, value",
    [
        (str | int, "a"),
        (str | int, 1),
        (str | None, None),
        (float, 1),
    ],
)
def test_validate_tool_arguments_single_value_accepts_valid(tool_input_type, value):
    @tool
    def test_tool(argument_a: tool_input_type) -> str:
        """Fake tool

        Args:
            argument_a: The input
        """
        return str(argument_a)

    validate_tool_arguments(test_tool, value)  # should not raise


@pytest.mark.parametrize(
    "tool_input_type, value",
    [
        (str, 1),
        (int, "a"),
        (str, None),
    ],
)
def test_validate_tool_arguments_single_value_rejects_invalid(tool_input_type, value):
    @tool
    def test_tool(argument_a: tool_input_type) -> str:
        """Fake tool

        Args:
            argument_a: The input
        """
        return str(argument_a)

    with pytest.raises(TypeError):
        validate_tool_arguments(test_tool, value)
