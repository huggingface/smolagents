from textwrap import dedent

import pytest

from smolagents.local_python_executor import InterpreterError, LocalPythonExecutor


def test_issue_2395():
    executor = LocalPythonExecutor(additional_authorized_imports=[])

    malicious_code = dedent(
        """
        class TimeBomb:
            def __init__(self):
                self.armed = True

            def __del__(self):
                self.armed = False

        bomb = TimeBomb()
        """
    )

    with pytest.raises(InterpreterError, match="__del__"):
        executor(malicious_code)
