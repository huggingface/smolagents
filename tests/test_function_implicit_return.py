"""Functions without return should yield None, not their last statement value."""

from smolagents.local_python_executor import LocalPythonExecutor


def _run(code):
    executor = LocalPythonExecutor([])
    executor.send_tools({})
    executor(code)
    return executor.state


def test_function_without_return_yields_none():
    state = _run(
        """
def find_positive(values):
    for value in values:
        candidate = value
        if candidate > 0:
            return candidate
result = find_positive([-3, -2])
"""
    )
    assert state["result"] is None


def test_function_with_return_still_works():
    state = _run(
        """
def find_positive(values):
    for value in values:
        if value > 0:
            return value
result = find_positive([-3, 2])
"""
    )
    assert state["result"] == 2
