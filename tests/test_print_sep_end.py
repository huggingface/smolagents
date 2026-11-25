"""LocalPythonExecutor should honour print's sep, end and file arguments."""

import io

from smolagents.local_python_executor import LocalPythonExecutor


def test_print_sep_and_end():
    code = '''print("name", "score", sep=",")
print("Ada", "42", sep=",", end="")'''
    executor = LocalPythonExecutor([])
    executor.send_tools({})

    result = executor(code)

    assert result.logs == "name,score\nAda,42"


def test_print_file_goes_to_destination():
    executor = LocalPythonExecutor(["io"])
    executor.send_tools({})
    code = '''import io
buffer = io.StringIO()
print("a", "b", sep="-", file=buffer)
print("buffer:", buffer.getvalue())
'''
    result = executor(code)

    assert "buffer: a-b\n" in result.logs
    assert "a-b\n" not in result.logs.replace("buffer: a-b\n", "")
