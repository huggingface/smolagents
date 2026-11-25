"""A generator expression binds its leftmost iterable at creation time."""

import pytest

from smolagents.local_python_executor import (
    BASE_PYTHON_TOOLS,
    InterpreterError,
    evaluate_python_code,
)


def _result(code):
    return evaluate_python_code(code, static_tools=BASE_PYTHON_TOOLS, timeout_seconds=None)[0]


def test_leftmost_iterable_is_bound_at_creation():
    code = """
items = [1, 2]
values = (x for x in items)
items = [9]
result = list(values)
result
"""
    assert _result(code) == [1, 2]


def test_invalid_leftmost_iterable_fails_at_creation():
    code = "values = (x for x in 1)\nresult = 'ok'"
    with pytest.raises(InterpreterError, match="TypeError"):
        _result(code)
