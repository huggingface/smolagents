# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
import pathlib
import tempfile
import textwrap
import unittest

import pytest
from IPython.core.interactiveshell import InteractiveShell

from smolagents import Tool
from smolagents.tools import tool
from smolagents.utils import get_source, parse_code_blobs


class AgentTextTests(unittest.TestCase):
    def test_parse_code_blobs(self):
        with pytest.raises(ValueError):
            parse_code_blobs("Wrong blob!")

        # Parsing mardkwon with code blobs should work
        output = parse_code_blobs("""
Here is how to solve the problem:
Code:
```py
import numpy as np
```<end_code>
""")
        assert output == "import numpy as np"

        # Parsing code blobs should work
        code_blob = "import numpy as np"
        output = parse_code_blobs(code_blob)
        assert output == code_blob

    def test_multiple_code_blobs(self):
        test_input = """Here's a function that adds numbers:
```python
def add(a, b):
    return a + b
```
And here's a function that multiplies them:
```py
def multiply(a, b):
    return a * b
```"""

        expected_output = """def add(a, b):
    return a + b

def multiply(a, b):
    return a * b"""
        result = parse_code_blobs(test_input)
        assert result == expected_output


@pytest.fixture(scope="function")
def ipython_shell():
    """Reset IPython shell before and after each test."""
    shell = InteractiveShell.instance()
    shell.reset()  # Clean before test
    yield shell
    shell.reset()  # Clean after test


@pytest.mark.parametrize(
    "obj_name, code_blob",
    [
        ("test_func", "def test_func():\n    return 42"),
        ("TestClass", "class TestClass:\n    ..."),
    ],
)
def test_get_source_ipython(ipython_shell, obj_name, code_blob):
    ipython_shell.run_cell(code_blob, store_history=True)
    obj = ipython_shell.user_ns[obj_name]
    assert get_source(obj) == code_blob


def test_get_source_standard_class():
    class TestClass: ...

    source = get_source(TestClass)
    assert source == "class TestClass: ..."
    assert source == textwrap.dedent(inspect.getsource(TestClass)).strip()


def test_get_source_standard_function():
    def test_func(): ...

    source = get_source(test_func)
    assert source == "def test_func(): ..."
    assert source == textwrap.dedent(inspect.getsource(test_func)).strip()


def test_get_source_ipython_errors_empty_cells(ipython_shell):
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    ipython_shell.user_ns["In"] = [""]
    ipython_shell.run_cell(test_code, store_history=True)
    with pytest.raises(ValueError, match="No code cells found in IPython session"):
        get_source(ipython_shell.user_ns["TestClass"])


def test_get_source_ipython_errors_definition_not_found(ipython_shell):
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    ipython_shell.user_ns["In"] = ["", "print('No class definition here')"]
    ipython_shell.run_cell(test_code, store_history=True)
    with pytest.raises(ValueError, match="Could not find source code for TestClass in IPython history"):
        get_source(ipython_shell.user_ns["TestClass"])


def test_get_source_ipython_errors_type_error():
    with pytest.raises(TypeError, match="Expected class or callable"):
        get_source(None)


def test_e2e_class_tool_save():
    class TestTool(Tool):
        name = "test_tool"
        description = "Test tool description"
        inputs = {
            "task": {
                "type": "string",
                "description": "tool input",
            }
        }
        output_type = "string"

        def forward(self, task: str):
            import IPython  # noqa: F401

            return task

    test_tool = TestTool()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_tool.save(tmp_dir, make_gradio_app=True)
        assert set(os.listdir(tmp_dir)) == {"requirements.txt", "app.py", "tool.py"}
        assert (
            pathlib.Path(tmp_dir, "tool.py").read_text()
            == """from typing import Any, Optional
from smolagents.tools import Tool
import IPython

class TestTool(Tool):
    name = "test_tool"
    description = "Test tool description"
    inputs = {'task': {'type': 'string', 'description': 'tool input'}}
    output_type = "string"

    def forward(self, task: str):
        import IPython  # noqa: F401

        return task

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
"""
        )
        requirements = set(pathlib.Path(tmp_dir, "requirements.txt").read_text().split())
        assert requirements == {"IPython", "smolagents"}
        assert (
            pathlib.Path(tmp_dir, "app.py").read_text()
            == """from smolagents import launch_gradio_demo
from tool import TestTool

tool = TestTool()

launch_gradio_demo(tool)
"""
        )


def test_e2e_ipython_class_tool_save():
    shell = InteractiveShell.instance()
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_blob = textwrap.dedent(f"""
        from smolagents.tools import Tool
        class TestTool(Tool):
            name = "test_tool"
            description = "Test tool description"
            inputs = {{"task": {{"type": "string",
                    "description": "tool input",
                }}
            }}
            output_type = "string"

            def forward(self, task: str):
                import IPython  # noqa: F401

                return task
        TestTool().save("{tmp_dir}", make_gradio_app=True)
    """)
        assert shell.run_cell(code_blob, store_history=True).success
        assert set(os.listdir(tmp_dir)) == {"requirements.txt", "app.py", "tool.py"}
        assert (
            pathlib.Path(tmp_dir, "tool.py").read_text()
            == """from typing import Any, Optional
from smolagents.tools import Tool
import IPython

class TestTool(Tool):
    name = "test_tool"
    description = "Test tool description"
    inputs = {'task': {'type': 'string', 'description': 'tool input'}}
    output_type = "string"

    def forward(self, task: str):
        import IPython  # noqa: F401

        return task

    def __init__(self, *args, **kwargs):
        self.is_initialized = False
"""
        )
        requirements = set(pathlib.Path(tmp_dir, "requirements.txt").read_text().split())
        assert requirements == {"IPython", "smolagents"}
        assert (
            pathlib.Path(tmp_dir, "app.py").read_text()
            == """from smolagents import launch_gradio_demo
from tool import TestTool

tool = TestTool()

launch_gradio_demo(tool)
"""
        )


def test_e2e_function_tool_save():
    @tool
    def test_tool(task: str) -> str:
        """
        Test tool description

        Args:
            task: tool input
        """
        import IPython  # noqa: F401

        return task

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_tool.save(tmp_dir, make_gradio_app=True)
        assert set(os.listdir(tmp_dir)) == {"requirements.txt", "app.py", "tool.py"}
        assert (
            pathlib.Path(tmp_dir, "tool.py").read_text()
            == """from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "test_tool"
    description = "Test tool description"
    inputs = {"task":{"type":"string","description":"tool input"}}
    output_type = "string"

    def forward(self, task: str) -> str:
        \"""
        Test tool description

        Args:
            task: tool input
        \"""
        import IPython  # noqa: F401

        return task"""
        )
        requirements = set(pathlib.Path(tmp_dir, "requirements.txt").read_text().split())
        assert requirements == {"smolagents"}  # FIXME: IPython should be in the requirements
        assert (
            pathlib.Path(tmp_dir, "app.py").read_text()
            == """from smolagents import launch_gradio_demo
from tool import SimpleTool

tool = SimpleTool()

launch_gradio_demo(tool)
"""
        )


def test_e2e_ipython_function_tool_save():
    shell = InteractiveShell.instance()
    with tempfile.TemporaryDirectory() as tmp_dir:
        code_blob = textwrap.dedent(f"""
        from smolagents import tool

        @tool
        def test_tool(task: str) -> str:
            \"""
            Test tool description

            Args:
                task: tool input
            \"""
            import IPython  # noqa: F401

            return task

        test_tool.save("{tmp_dir}", make_gradio_app=True)
        """)
        assert shell.run_cell(code_blob, store_history=True).success
        assert set(os.listdir(tmp_dir)) == {"requirements.txt", "app.py", "tool.py"}
        assert (
            pathlib.Path(tmp_dir, "tool.py").read_text()
            == """from smolagents import Tool
from typing import Any, Optional

class SimpleTool(Tool):
    name = "test_tool"
    description = "Test tool description"
    inputs = {"task":{"type":"string","description":"tool input"}}
    output_type = "string"

    def forward(self, task: str) -> str:
        \"""
        Test tool description

        Args:
            task: tool input
        \"""
        import IPython  # noqa: F401

        return task"""
        )
        requirements = set(pathlib.Path(tmp_dir, "requirements.txt").read_text().split())
        assert requirements == {"smolagents"}  # FIXME: IPython should be in the requirements
        assert (
            pathlib.Path(tmp_dir, "app.py").read_text()
            == """from smolagents import launch_gradio_demo
from tool import SimpleTool

tool = SimpleTool()

launch_gradio_demo(tool)
"""
        )
