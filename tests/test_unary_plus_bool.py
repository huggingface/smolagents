"""Unary plus must apply Python's bool-to-int conversion."""

from smolagents.local_python_executor import BASE_PYTHON_TOOLS, evaluate_python_code


def test_unary_plus_converts_bool_to_int():
    values = [True, False, -7, 2.5]
    result, _ = evaluate_python_code(
        "normalized = [+value for value in values]\nnormalized",
        static_tools=dict(BASE_PYTHON_TOOLS),
        state={"values": values},
    )

    assert result == [1, 0, -7, 2.5]
    assert [type(value) for value in result] == [int, int, int, float]
