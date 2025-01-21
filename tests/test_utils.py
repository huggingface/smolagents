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
import unittest

import pytest

from smolagents.utils import parse_code_blobs


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
