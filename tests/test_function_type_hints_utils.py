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
from typing import Optional, Tuple

from smolagents._function_type_hints_utils import get_json_schema


class AgentTextTests(unittest.TestCase):
    def test_return_none(self):
        def fn(x: int, y: Optional[Tuple[str, str, float]] = None) -> None:
            """
            Test function
            Args:
                x: The first input
                y: The second input
            """
            pass

        schema = get_json_schema(fn)
        expected_schema = {
            "name": "fn",
            "description": "Test function",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The first input"},
                    "y": {
                        "type": "array",
                        "description": "The second input",
                        "nullable": True,
                        "prefixItems": [{"type": "string"}, {"type": "string"}, {"type": "number"}],
                    },
                },
                "required": ["x"],
            },
            "return": {"type": "null"},
        }
        self.assertEqual(
            schema["function"]["parameters"]["properties"]["y"], expected_schema["parameters"]["properties"]["y"]
        )
        self.assertEqual(schema["function"], expected_schema)
