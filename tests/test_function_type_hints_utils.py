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
from typing import Any

import pytest

from smolagents._function_type_hints_utils import DocstringParsingException, get_imports, get_json_schema


@pytest.fixture
def valid_func():
    """A well-formed function with docstring, type hints, and return block."""

    def multiply(x: int, y: float) -> float:
        """
        Multiplies two numbers.

        Args:
            x: The first number.
            y: The second number.
        Returns:
            Product of x and y.
        """
        return x * y

    return multiply


@pytest.fixture
def no_docstring_func():
    """Function with no docstring."""

    def sample(x: int):
        return x

    return sample


@pytest.fixture
def missing_arg_doc_func():
    """Function with docstring but missing an argument description."""

    def add(x: int, y: int):
        """
        Adds two numbers.

        Args:
            x: The first number.
        """
        return x + y

    return add


@pytest.fixture
def bad_return_func():
    """Function docstring with missing return description (allowed)."""

    def do_nothing(x: str | None = None):
        """
        Does nothing.

        Args:
            x: Some optional string.
        """
        pass

    return do_nothing


@pytest.fixture
def complex_types_func():
    def process_data(items: list[str], config: dict[str, float], point: tuple[int, int]) -> dict:
        """
        Process some data.

        Args:
            items: List of items to process.
            config: Configuration parameters.
            point: A position as (x,y).

        Returns:
            Processed data result.
        """
        return {"result": True}

    return process_data


@pytest.fixture
def optional_types_func():
    def process_with_optional(required_arg: str, optional_arg: int | None = None) -> str:
        """
        Process with optional argument.

        Args:
            required_arg: A required string argument.
            optional_arg: An optional integer argument.

        Returns:
            Processing result.
        """
        return "processed"

    return process_with_optional


@pytest.fixture
def enum_choices_func():
    def select_color(color: str) -> str:
        """
        Select a color.

        Args:
            color: The color to select (choices: ["red", "green", "blue"])

        Returns:
            Selected color.
        """
        return color

    return select_color


@pytest.fixture
def union_types_func():
    def process_union(value: int | str) -> bool | str:
        """
        Process a value that can be either int or string.

        Args:
            value: An integer or string value.

        Returns:
            Processing result.
        """
        return True if isinstance(value, int) else "string result"

    return process_union


@pytest.fixture
def nested_types_func():
    def process_nested_data(data: list[dict[str, Any]]) -> list[str]:
        """
        Process nested data structure.

        Args:
            data: List of dictionaries to process.

        Returns:
            List of processed results.
        """
        return ["result"]

    return process_nested_data


@pytest.fixture
def typed_docstring_func():
    def calculate(x: int, y: float) -> float:
        """
        Calculate something.

        Args:
            x (int): An integer parameter with type in docstring.
            y (float): A float parameter with type in docstring.

        Returns:
            float: The calculated result.
        """
        return x * y

    return calculate


@pytest.fixture
def mismatched_types_func():
    def convert(value: int) -> str:
        """
        Convert a value.

        Args:
            value (str): A string value (type mismatch with hint).

        Returns:
            int: Converted value (type mismatch with hint).
        """
        return str(value)

    return convert


@pytest.fixture
def complex_docstring_types_func():
    def process(data: dict[str, list[int]]) -> list[dict[str, Any]]:
        """
        Process complex data.

        Args:
            data (Dict[str, List[int]]): Nested structure with types.

        Returns:
            List[Dict[str, Any]]: Processed results with types.
        """
        return [{"result": sum(v) for k, v in data.items()}]

    return process


@pytest.fixture
def keywords_in_description_func():
    def process(value: str) -> str:
        """
        Function with Args: or Returns: keywords in its description.

        Args:
            value: A string value.

        Returns:
            str: Processed value.
        """
        return value.upper()

    return process


class TestGetJsonSchema:
    def test_get_json_schema_example(self):
        def fn(x: int, y: tuple[str, str, float] | None = None) -> None:
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
        assert schema["function"]["parameters"]["properties"]["y"] == expected_schema["parameters"]["properties"]["y"]
        assert schema["function"] == expected_schema

    @pytest.mark.parametrize(
        "fixture_name,should_fail",
        [
            ("valid_func", False),
            # ('no_docstring_func', True),
            # ('missing_arg_doc_func', True),
            ("bad_return_func", False),
        ],
    )
    def test_get_json_schema(self, request, fixture_name, should_fail):
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)
        assert schema["type"] == "function"
        assert "function" in schema
        assert "parameters" in schema["function"]

    @pytest.mark.parametrize(
        "fixture_name,should_fail",
        [
            # ('valid_func', False),
            ("no_docstring_func", True),
            ("missing_arg_doc_func", True),
            # ('bad_return_func', False),
        ],
    )
    def test_get_json_schema_raises(self, request, fixture_name, should_fail):
        func = request.getfixturevalue(fixture_name)
        with pytest.raises(DocstringParsingException):
            get_json_schema(func)

    @pytest.mark.parametrize(
        "fixture_name,expected_properties",
        [
            ("valid_func", {"x": "integer", "y": "number"}),
            ("bad_return_func", {"x": "string"}),
        ],
    )
    def test_property_types(self, request, fixture_name, expected_properties):
        """Test that property types are correctly mapped."""
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)

        properties = schema["function"]["parameters"]["properties"]
        for prop_name, expected_type in expected_properties.items():
            assert properties[prop_name]["type"] == expected_type

    def test_schema_basic_structure(self, valid_func):
        """Test that basic schema structure is correct."""
        schema = get_json_schema(valid_func)
        # Check schema type
        assert schema["type"] == "function"
        assert "function" in schema
        # Check function schema
        function_schema = schema["function"]
        assert function_schema["name"] == "multiply"
        assert "description" in function_schema
        assert function_schema["description"] == "Multiplies two numbers."
        # Check parameters schema
        assert "parameters" in function_schema
        params = function_schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert set(params["required"]) == {"x", "y"}
        properties = params["properties"]
        assert properties["x"]["type"] == "integer"
        assert properties["y"]["type"] == "number"
        # Check return schema
        assert "return" in function_schema
        return_schema = function_schema["return"]
        assert return_schema["type"] == "number"
        assert return_schema["description"] == "Product of x and y."

    def test_complex_types(self, complex_types_func):
        """Test schema generation for complex types."""
        schema = get_json_schema(complex_types_func)
        properties = schema["function"]["parameters"]["properties"]
        # Check list type
        assert properties["items"]["type"] == "array"
        # Check dict type
        assert properties["config"]["type"] == "object"
        # Check tuple type
        assert properties["point"]["type"] == "array"
        assert len(properties["point"]["prefixItems"]) == 2
        assert properties["point"]["prefixItems"][0]["type"] == "integer"
        assert properties["point"]["prefixItems"][1]["type"] == "integer"

    def test_optional_types(self, optional_types_func):
        """Test schema generation for optional arguments."""
        schema = get_json_schema(optional_types_func)
        params = schema["function"]["parameters"]
        # Required argument should be in required list
        assert "required_arg" in params["required"]
        # Optional argument should not be in required list
        assert "optional_arg" not in params["required"]
        # Optional argument should be nullable
        assert params["properties"]["optional_arg"]["nullable"] is True
        assert params["properties"]["optional_arg"]["type"] == "integer"

    def test_enum_choices(self, enum_choices_func):
        """Test schema generation for enum choices in docstring."""
        schema = get_json_schema(enum_choices_func)
        color_prop = schema["function"]["parameters"]["properties"]["color"]
        assert "enum" in color_prop
        assert color_prop["enum"] == ["red", "green", "blue"]

    def test_union_types(self, union_types_func):
        """Test schema generation for union types."""
        schema = get_json_schema(union_types_func)
        value_prop = schema["function"]["parameters"]["properties"]["value"]
        return_prop = schema["function"]["return"]
        # Check union in parameter
        assert len(value_prop["type"]) == 2
        # Check union in return type: should be converted to "any"
        assert return_prop["type"] == "any"

    def test_nested_types(self, nested_types_func):
        """Test schema generation for nested complex types."""
        schema = get_json_schema(nested_types_func)
        data_prop = schema["function"]["parameters"]["properties"]["data"]
        assert data_prop["type"] == "array"

    def test_typed_docstring_parsing(self, typed_docstring_func):
        """Test parsing of docstrings with type annotations."""
        schema = get_json_schema(typed_docstring_func)
        # Type hints should take precedence over docstring types
        assert schema["function"]["parameters"]["properties"]["x"]["type"] == "integer"
        assert schema["function"]["parameters"]["properties"]["y"]["type"] == "number"
        # Description should be extracted correctly
        assert (
            schema["function"]["parameters"]["properties"]["x"]["description"]
            == "An integer parameter with type in docstring."
        )
        assert (
            schema["function"]["parameters"]["properties"]["y"]["description"]
            == "A float parameter with type in docstring."
        )
        # Return type and description should be correct
        assert schema["function"]["return"]["type"] == "number"
        assert schema["function"]["return"]["description"] == "The calculated result."

    def test_mismatched_docstring_types(self, mismatched_types_func):
        """Test that type hints take precedence over docstring types when they conflict."""
        schema = get_json_schema(mismatched_types_func)
        # Type hints should take precedence over docstring types
        assert schema["function"]["parameters"]["properties"]["value"]["type"] == "integer"
        # Return type from type hint should be used, not docstring
        assert schema["function"]["return"]["type"] == "string"

    def test_complex_docstring_types(self, complex_docstring_types_func):
        """Test parsing of complex type annotations in docstrings."""
        schema = get_json_schema(complex_docstring_types_func)
        # Check that complex nested type is parsed correctly from type hints
        data_prop = schema["function"]["parameters"]["properties"]["data"]
        assert data_prop["type"] == "object"
        # Check return type
        return_prop = schema["function"]["return"]
        assert return_prop["type"] == "array"
        # Description should include the type information from docstring
        assert data_prop["description"] == "Nested structure with types."
        assert return_prop["description"] == "Processed results with types."

    @pytest.mark.parametrize(
        "fixture_name,expected_description",
        [
            ("typed_docstring_func", "An integer parameter with type in docstring."),
            ("complex_docstring_types_func", "Nested structure with types."),
        ],
    )
    def test_type_in_description_handling(self, request, fixture_name, expected_description):
        """Test that type information in docstrings is preserved in description."""
        func = request.getfixturevalue(fixture_name)
        schema = get_json_schema(func)
        # First parameter description should contain the expected text
        first_param_name = list(schema["function"]["parameters"]["properties"].keys())[0]
        assert schema["function"]["parameters"]["properties"][first_param_name]["description"] == expected_description

    def test_with_special_words_in_description_func(self, keywords_in_description_func):
        schema = get_json_schema(keywords_in_description_func)
        assert schema["function"]["description"] == "Function with Args: or Returns: keywords in its description."


class TestGetCode:
    @pytest.mark.parametrize(
        "code, expected",
        [
            (
                """
        import numpy
        import pandas
        """,
                ["numpy", "pandas"],
            ),
            # From imports
            (
                """
        from torch import nn
        from transformers import AutoModel
        """,
                ["torch", "transformers"],
            ),
            # Mixed case with nested imports
            (
                """
        import numpy as np
        from torch.nn import Linear
        import os.path
        """,
                ["numpy", "torch", "os"],
            ),
            # Try/except block (should be filtered)
            (
                """
        try:
            import torch
        except ImportError:
            pass
        import numpy
        """,
                ["numpy"],
            ),
            # Flash attention block (should be filtered)
            (
                """
        if is_flash_attn_2_available():
            from flash_attn import flash_attn_func
        import transformers
        """,
                ["transformers"],
            ),
            # Relative imports (should be excluded)
            (
                """
        from .utils import helper
        from ..models import transformer
        """,
                [],
            ),
        ],
    )
    def test_get_imports(self, code: str, expected: list[str]):
        assert sorted(get_imports(code)) == sorted(expected)
