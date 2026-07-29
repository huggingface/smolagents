import pytest

from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor


def test_issue_2395():
    executor = LocalPythonExecutor(additional_authorized_imports=[])

    malicious_code = """
class TimeBomb:
    def __del__(self):
        pass

bomb = TimeBomb()
"""

    with pytest.raises(InterpreterError, match=r"dunder.*__del__|__del__.*dunder"):
        executor(malicious_code)
