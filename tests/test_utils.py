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
import textwrap
import unittest

import pytest
from IPython.core.interactiveshell import InteractiveShell

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


@pytest.mark.parametrize(
    "obj_name, code_blob",
    [
        ("test_func", "def test_func():\n    return 42"),
        ("TestClass", "class TestClass:\n    ..."),
    ],
)
def test_get_source_ipython(obj_name, code_blob):
    shell = InteractiveShell.instance()
    test_code = textwrap.dedent(code_blob).strip()
    shell.user_ns["In"] = ["", test_code]
    exec(test_code)
    assert get_source(locals()[obj_name]) == code_blob


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


def test_get_source_ipython_errors_empty_cells():
    shell = InteractiveShell.instance()
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    shell.user_ns["In"] = [""]
    exec(test_code)
    with pytest.raises(ValueError, match="No code cells found in IPython session"):
        get_source(locals()["TestClass"])


def test_get_source_ipython_errors_definition_not_found():
    shell = InteractiveShell.instance()
    test_code = textwrap.dedent("""class TestClass:\n    ...""").strip()
    shell.user_ns["In"] = ["", "print('No class definition here')"]
    exec(test_code)
    with pytest.raises(ValueError, match="Could not find source code for TestClass in IPython history"):
        get_source(locals()["TestClass"])


def test_get_source_ipython_errors_type_error():
    with pytest.raises(TypeError, match="Expected class or callable"):
        get_source(None)
