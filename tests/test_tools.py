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
import warnings
from enum import Enum
from textwrap import dedent
from typing import Any, Literal
from unittest.mock import MagicMock, patch

import mcp
import numpy as np
import PIL.Image
import pytest

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.tools import AUTHORIZED_TYPES, Tool, ToolCollection, launch_gradio_demo, tool, validate_tool_arguments

from .utils.markers import require_run_all


class ToolTesterMixin:
    def test_inputs_output(self):
        assert hasattr(self.tool, "inputs")
        assert hasattr(self.tool, "output_type")

        inputs = self.tool.inputs
        assert isinstance(inputs, dict)

        for _, input_spec in inputs.items():
            assert "type" in input_spec
            assert "description" in input_spec
            assert input_spec["type"] in AUTHORIZED_TYPES
            assert isinstance(input_spec["description"], str)

        output_type = self.tool.output_type
        assert output_type in AUTHORIZED_TYPES

    def test_common_attributes(self):
        assert hasattr(self.tool, "description")
        assert hasattr(self.tool, "name")
        assert hasattr(self.tool, "inputs")
        assert hasattr(self.tool, "output_type")

    def test_agent_type_output(self, create_inputs):
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(**inputs, sanitize_inputs_outputs=True)
        if self.tool.output_type != "any":
            agent_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def create_inputs(self, shared_datadir):
        def _create_inputs(tool_inputs: dict[str, dict[str | type, str]]) -> dict[str, Any]:
            inputs = {}

            for input_name, input_desc in tool_inputs.items():
                input_type = input_desc["type"]

                if input_type == "string":
                    inputs[input_name] = "Text input"
                elif input_type == "image":
                    inputs[input_name] = PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))
                elif input_type == "audio":
                    inputs[input_name] = np.ones(3000)
                else:
                    raise ValueError(f"Invalid type requested: {input_type}")

            return inputs

        return _create_inputs


class TestTool:
    @pytest.mark.parametrize(
        "type_value, should_raise_error, error_contains",
        [
            # Valid cases
            ("string", False, None),
            (["string", "number"], False, None),
            # Invalid cases
            ("invalid_type", ValueError, "must be one of"),
            (["string", "invalid_type"], ValueError, "must be one of"),
            ([123, "string"], TypeError, "when type is a list, all elements must be strings"),
            (123, TypeError, "must be a string or list of strings"),
        ],
    )
    def test_tool_input_type_validation(self, type_value, should_raise_error, error_contains):
        """Test the validation of the type property in tool inputs."""

        # Define a tool class with the test type value
        def create_tool():
            class TestTool(Tool):
                name = "test_tool"
                description = "A tool for testing type validation"
                inputs = {"text": {"type": type_value, "description": "Some input"}}
                output_type = "string"

                def forward(self, text) -> str:
                    return text

            return TestTool()

        # Check if we expect this to raise an exception
        if should_raise_error:
            with pytest.raises(should_raise_error) as exc_info:
                create_tool()
            # Verify the error message contains expected text
            assert error_contains in str(exc_info.value)
        else:
            # Should not raise an exception
            tool = create_tool()
            assert isinstance(tool, Tool)

    @pytest.mark.parametrize(
        "tool_fixture, expected_output",
        [
            ("no_input_tool", 'def no_input_tool() -> string:\n    """Tool with no inputs\n    """'),
            (
                "single_input_tool",
                'def single_input_tool(text: string) -> string:\n    """Tool with one input\n\n    Args:\n        text: Input text\n    """',
            ),
            (
                "multi_input_tool",
                'def multi_input_tool(text: string, count: integer) -> object:\n    """Tool with multiple inputs\n\n    Args:\n        text: Text input\n        count: Number count\n    """',
            ),
            (
                "multiline_description_tool",
                'def multiline_description_tool(input: string) -> string:\n    """This is a tool with\n    multiple lines\n    in the description\n\n    Args:\n        input: Some input\n    """',
            ),
        ],
    )
    def test_tool_to_code_prompt_output_format(self, tool_fixture, expected_output, request):
        """Test that to_code_prompt generates properly formatted and indented output."""
        tool = request.getfixturevalue(tool_fixture)
        code_prompt = tool.to_code_prompt()
        assert code_prompt == expected_output

    @pytest.mark.parametrize(
        "tool_fixture, expected_output",
        [
            (
                "no_input_tool",
                "no_input_tool: Tool with no inputs\n    Takes inputs: {}\n    Returns an output of type: string",
            ),
            (
                "single_input_tool",
                "single_input_tool: Tool with one input\n    Takes inputs: {'text': {'type': 'string', 'description': 'Input text'}}\n    Returns an output of type: string",
            ),
            (
                "multi_input_tool",
                "multi_input_tool: Tool with multiple inputs\n    Takes inputs: {'text': {'type': 'string', 'description': 'Text input'}, 'count': {'type': 'integer', 'description': 'Number count'}}\n    Returns an output of type: object",
            ),
            (
                "multiline_description_tool",
                "multiline_description_tool: This is a tool with\nmultiple lines\nin the description\n    Takes inputs: {'input': {'type': 'string', 'description': 'Some input'}}\n    Returns an output of type: string",
            ),
        ],
    )
    def test_tool_to_tool_calling_prompt_output_format(self, tool_fixture, expected_output, request):
        """Test that to_tool_calling_prompt generates properly formatted output."""
        tool = request.getfixturevalue(tool_fixture)
        tool_calling_prompt = tool.to_tool_calling_prompt()
        assert tool_calling_prompt == expected_output

    def test_tool_init_with_decorator(self):
        @tool
        def coolfunc(a: str, b: int) -> float:
            """Cool function

            Args:
                a: The first argument
                b: The second one
            """
            return b + 2, a

        assert coolfunc.output_type == "number"

    def test_tool_init_vanilla(self):
        class HFModelDownloadsTool(Tool):
            name = "model_download_counter"
            description = """
            This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
            It returns the name of the checkpoint."""

            inputs = {
                "task": {
                    "type": "string",
                    "description": "the task category (such as text-classification, depth-estimation, etc)",
                }
            }
            output_type = "string"

            def forward(self, task: str) -> str:
                return "best model"

        tool = HFModelDownloadsTool()
        assert list(tool.inputs.keys())[0] == "task"

    def test_tool_init_decorator_raises_issues(self):
        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int):
                """Cool function

                Args:
                    a: The first argument
                    b: The second one
                """
                return a + b

            assert coolfunc.output_type == "number"
        assert "Tool return type not found" in str(e)

        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int) -> int:
                """Cool function

                Args:
                    a: The first argument
                """
                return b + a

            assert coolfunc.output_type == "number"
        assert "docstring has no description for the argument" in str(e)

    def test_saving_tool_raises_error_imports_outside_function(self, tmp_path):
        with pytest.raises(Exception) as e:
            import numpy as np

            @tool
            def get_current_time() -> str:
                """
                Gets the current time.
                """
                return str(np.random.random())

            get_current_time.save(tmp_path)

        assert "np" in str(e)

        # Also test with classic definition
        with pytest.raises(Exception) as e:

            class GetCurrentTimeTool(Tool):
                name = "get_current_time_tool"
                description = "Gets the current time"
                inputs = {}
                output_type = "string"

                def forward(self):
                    return str(np.random.random())

            get_current_time = GetCurrentTimeTool()
            get_current_time.save(tmp_path)

        assert "np" in str(e)

    def test_tool_definition_raises_no_error_imports_in_function(self):
        @tool
        def get_current_time() -> str:
            """
            Gets the current time.
            """
            from datetime import datetime

            return str(datetime.now())

        class GetCurrentTimeTool(Tool):
            name = "get_current_time_tool"
            description = "Gets the current time"
            inputs = {}
            output_type = "string"

            def forward(self):
                from datetime import datetime

                return str(datetime.now())

    def test_tool_to_dict_allows_no_arg_in_init(self):
        """Test that a tool cannot be saved with required args in init"""

        class FailTool(Tool):
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def __init__(self, url):
                super().__init__(self)
                self.url = url

            def forward(self, string_input: str) -> str:
                return self.url + string_input

        fail_tool = FailTool("dummy_url")
        with pytest.raises(Exception) as e:
            fail_tool.to_dict()
        assert "Parameters in __init__ must have default values, found required parameters" in str(e)

        class PassTool(Tool):
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def __init__(self, url: str | None = "none"):
                super().__init__(self)
                self.url = url

            def forward(self, string_input: str) -> str:
                return self.url + string_input

        fail_tool = PassTool()
        fail_tool.to_dict()

    def test_saving_tool_allows_no_imports_from_outside_methods(self, tmp_path):
        # Test that using imports from outside functions fails
        import numpy as np

        class FailTool(Tool):
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def useless_method(self):
                self.client = np.random.random()
                return ""

            def forward(self, string_input):
                return self.useless_method() + string_input

        fail_tool = FailTool()
        with pytest.raises(Exception) as e:
            fail_tool.save(tmp_path)
        assert "'np' is undefined" in str(e)

        # Test that putting these imports inside functions works
        class SuccessTool(Tool):
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def useless_method(self):
                import numpy as np

                self.client = np.random.random()
                return ""

            def forward(self, string_input):
                return self.useless_method() + string_input

        success_tool = SuccessTool()
        success_tool.save(tmp_path)

    def test_tool_missing_class_attributes_raises_error(self):
        with pytest.raises(Exception) as e:

            class GetWeatherTool(Tool):
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }

                def forward(self, location: str, celsius: bool | None = False) -> str:
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool()
        assert "You must set an attribute output_type" in str(e)

    def test_tool_from_decorator_optional_args(self):
        @tool
        def get_weather(location: str, celsius: bool | None = False) -> str:
            """
            Get weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert "nullable" in get_weather.inputs["celsius"]
        assert get_weather.inputs["celsius"]["nullable"]
        assert "nullable" not in get_weather.inputs["location"]

    def test_tool_mismatching_nullable_args_raises_error(self):
        with pytest.raises(Exception) as e:

            class GetWeatherTool(Tool):
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }
                output_type = "string"

                def forward(self, location: str, celsius: bool | None = False) -> str:
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool()
        assert "Nullable" in str(e)

        with pytest.raises(Exception) as e:

            class GetWeatherTool2(Tool):
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                    },
                }
                output_type = "string"

                def forward(self, location: str, celsius: bool = False) -> str:
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool2()
        assert "Nullable" in str(e)

        with pytest.raises(Exception) as e:

            class GetWeatherTool3(Tool):
                name = "get_weather"
                description = "Get weather in the next days at given location."
                inputs = {
                    "location": {"type": "string", "description": "the location"},
                    "celsius": {
                        "type": "string",
                        "description": "the temperature type",
                        "nullable": True,
                    },
                }
                output_type = "string"

                def forward(self, location, celsius: str) -> str:
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool3()
        assert "Nullable" in str(e)

    def test_tool_default_parameters_is_nullable(self):
        @tool
        def get_weather(location: str, celsius: bool = False) -> str:
            """
            Get weather in the next days at given location.

            Args:
                location: The location to get the weather for.
                celsius: is the temperature given in celsius?
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        assert get_weather.inputs["celsius"]["nullable"]

    def test_tool_supports_any_none(self, tmp_path):
        @tool
        def get_weather(location: Any) -> None:
            """
            Get weather in the next days at given location.

            Args:
                location: The location to get the weather for.
            """
            return

        get_weather.save(tmp_path)
        assert get_weather.inputs["location"]["type"] == "any"
        assert get_weather.output_type == "null"

    def test_tool_supports_array(self):
        @tool
        def get_weather(locations: list[str], months: tuple[str, str] | None = None) -> dict[str, float]:
            """
            Get weather in the next days at given locations.

            Args:
                locations: The locations to get the weather for.
                months: The months to get the weather for
            """
            return

        assert get_weather.inputs["locations"]["type"] == "array"
        assert get_weather.inputs["months"]["type"] == "array"

    def test_tool_supports_string_literal(self):
        @tool
        def get_weather(unit: Literal["celsius", "fahrenheit"] = "celsius") -> None:
            """
            Get weather in the next days at given location.

            Args:
                unit: The unit of temperature
            """
            return

        assert get_weather.inputs["unit"]["type"] == "string"
        assert get_weather.inputs["unit"]["enum"] == ["celsius", "fahrenheit"]

    def test_tool_supports_numeric_literal(self):
        @tool
        def get_choice(choice: Literal[1, 2, 3]) -> None:
            """
            Get choice based on the provided numeric literal.

            Args:
                choice: The numeric choice to be made.
            """
            return

        assert get_choice.inputs["choice"]["type"] == "integer"
        assert get_choice.inputs["choice"]["enum"] == [1, 2, 3]

    def test_tool_supports_nullable_literal(self):
        @tool
        def get_choice(choice: Literal[1, 2, 3, None]) -> None:
            """
            Get choice based on the provided value.

            Args:
                choice: The numeric choice to be made.
            """
            return

        assert get_choice.inputs["choice"]["type"] == "integer"
        assert get_choice.inputs["choice"]["nullable"] is True
        assert get_choice.inputs["choice"]["enum"] == [1, 2, 3]

    def test_saving_tool_produces_valid_pyhon_code_with_multiline_description(self, tmp_path):
        @tool
        def get_weather(location: Any) -> None:
            """
            Get weather in the next days at given location.
            And works pretty well.

            Args:
                location: The location to get the weather for.
            """
            return

        get_weather.save(tmp_path)
        with open(os.path.join(tmp_path, "tool.py"), "r", encoding="utf-8") as f:
            source_code = f.read()
            compile(source_code, f.name, "exec")

    @pytest.mark.parametrize("fixture_name", ["boolean_default_tool_class", "boolean_default_tool_function"])
    def test_to_dict_boolean_default_input(self, fixture_name, request):
        """Test that boolean input parameter with default value is correctly represented in to_dict output"""
        tool = request.getfixturevalue(fixture_name)
        result = tool.to_dict()
        # Check that the boolean default annotation is preserved
        assert "flag: bool = False" in result["code"]
        # Check nullable attribute is set for the parameter with default value
        assert "'nullable': True" in result["code"]

    @pytest.mark.parametrize("fixture_name", ["optional_input_tool_class", "optional_input_tool_function"])
    def test_to_dict_optional_input(self, fixture_name, request):
        """Test that Optional/nullable input parameter is correctly represented in to_dict output"""
        tool = request.getfixturevalue(fixture_name)
        result = tool.to_dict()
        # Check the Optional type annotation is preserved
        assert "optional_text: str | None = None" in result["code"]
        # Check that the input is marked as nullable in the code
        assert "'nullable': True" in result["code"]

    def test_from_dict_roundtrip(self, example_tool):
        # Convert to dict
        tool_dict = example_tool.to_dict()
        # Create from dict
        recreated_tool = Tool.from_dict(tool_dict)
        # Verify properties
        assert recreated_tool.name == example_tool.name
        assert recreated_tool.description == example_tool.description
        assert recreated_tool.inputs == example_tool.inputs
        assert recreated_tool.output_type == example_tool.output_type
        # Verify functionality
        test_input = "Hello, world!"
        assert recreated_tool(test_input) == test_input.upper()

    def test_tool_from_dict_invalid(self):
        # Missing code key
        with pytest.raises(ValueError) as e:
            Tool.from_dict({"name": "invalid_tool"})
        assert "must contain 'code' key" in str(e)

    def test_tool_decorator_preserves_original_function(self):
        # Define a test function with type hints and docstring
        def test_function(items: list[str]) -> str:
            """Join a list of strings.
            Args:
                items: A list of strings to join
            Returns:
                The joined string
            """
            return ", ".join(items)

        # Store original function signature, name, and source
        original_signature = inspect.signature(test_function)
        original_name = test_function.__name__
        original_docstring = test_function.__doc__

        # Create a tool from the function
        test_tool = tool(test_function)

        # Check that the original function is unchanged
        assert original_signature == inspect.signature(test_function)
        assert original_name == test_function.__name__
        assert original_docstring == test_function.__doc__

        # Verify that the tool's forward method has a different signature (it has 'self')
        tool_forward_sig = inspect.signature(test_tool.forward)
        assert list(tool_forward_sig.parameters.keys())[0] == "self"

        # Original function should not have 'self' parameter
        assert "self" not in original_signature.parameters

    def test_tool_with_union_type_return(self):
        @tool
        def union_type_return_tool_function(param: int) -> str | bool:
            """
            Tool with output union type.

            Args:
                param: Input parameter.
            """
            return str(param) if param > 0 else False

        assert isinstance(union_type_return_tool_function, Tool)
        assert union_type_return_tool_function.output_type == "any"


class TestToolDecorator:
    def test_tool_decorator_source_extraction_with_multiple_decorators(self):
        """Test that @tool correctly extracts source code with multiple decorators."""

        def dummy_decorator(func):
            return func

        with pytest.warns(UserWarning, match="has decorators other than @tool"):

            @tool
            @dummy_decorator
            def multi_decorator_tool(text: str) -> str:
                """Tool with multiple decorators.

                Args:
                    text: Input text
                """
                return text.upper()

        # Verify the tool works
        assert isinstance(multi_decorator_tool, Tool)
        assert multi_decorator_tool.name == "multi_decorator_tool"
        assert multi_decorator_tool("hello") == "HELLO"

        # Verify the source code extraction is correct
        forward_source = multi_decorator_tool.forward.__source__
        assert "def forward(self, text: str) -> str:" in forward_source
        assert "return text.upper()" in forward_source
        # Should not contain decorator lines
        assert "@tool" not in forward_source
        assert "@dummy_decorator" not in forward_source
        # Should not contain definition line
        assert "def multi_decorator_tool" not in forward_source

    def test_tool_decorator_source_extraction_with_multiline_signature(self):
        """Test that @tool correctly extracts source code with multiline function signatures."""

        with warnings.catch_warnings():
            warnings.simplefilter("error")

            @tool
            def multiline_signature_tool(
                text: str,
                count: int = 1,
                uppercase: bool = False,
                multiline_parameter_1: int = 1_000,
                multiline_parameter_2: int = 2_000,
            ) -> str:
                """Tool with multiline signature.

                Args:
                    text: Input text
                    count: Number of repetitions
                    uppercase: Whether to convert to uppercase
                    multiline_parameter_1: Dummy parameter
                    multiline_parameter_2: Dummy parameter
                """
                result = text * count
                return result.upper() if uppercase else result

        # Verify the tool works
        assert isinstance(multiline_signature_tool, Tool)
        assert multiline_signature_tool.name == "multiline_signature_tool"
        assert multiline_signature_tool("hello", 2, True) == "HELLOHELLO"

        # Verify the source code extraction is correct
        forward_source = multiline_signature_tool.forward.__source__
        assert (
            "def forward(self, text: str, count: int=1, uppercase: bool=False, multiline_parameter_1: int=1000, multiline_parameter_2: int=2000) -> str:"
            in forward_source
            or "def forward(self, text: str, count: int = 1, uppercase: bool = False, multiline_parameter_1: int = 1000, multiline_parameter_2: int = 2000) -> str:"
            in forward_source
        )
        assert "result = text * count" in forward_source
        assert "return result.upper() if uppercase else result" in forward_source
        # Should not contain the original multiline function definition
        assert "def multiline_signature_tool(" not in forward_source
        # Should not contain leftover lines from the original multiline function definition
        assert "            count: int = 1," not in forward_source
        assert "            count: int=1," not in forward_source

    def test_tool_decorator_source_extraction_with_multiple_decorators_and_multiline(self):
        """Test that @tool works with both multiple decorators and multiline signatures."""

        def dummy_decorator_1(func):
            return func

        def dummy_decorator_2(func):
            return func

        with pytest.warns(UserWarning, match="has decorators other than @tool"):

            @tool
            @dummy_decorator_1
            @dummy_decorator_2
            def complex_tool(
                text: str,
                multiplier: int = 2,
                separator: str = " ",
                multiline_parameter_1: int = 1_000,
                multiline_parameter_2: int = 2_000,
            ) -> str:
                """Complex tool with multiple decorators and multiline signature.

                Args:
                    text: Input text
                    multiplier: How many times to repeat
                    separator: What to use between repetitions
                    multiline_parameter_1: Dummy parameter
                    multiline_parameter_2: Dummy parameter
                """
                parts = [text] * multiplier
                return separator.join(parts)

        # Verify the tool works
        assert isinstance(complex_tool, Tool)
        assert complex_tool.name == "complex_tool"
        assert complex_tool("hello", 3, "-") == "hello-hello-hello"

        # Verify the source code extraction is correct
        forward_source = complex_tool.forward.__source__
        assert (
            "def forward(self, text: str, multiplier: int=2, separator: str=' ', multiline_parameter_1: int=1000, multiline_parameter_2: int=2000) -> str:"
            in forward_source
            or "def forward(self, text: str, multiplier: int = 2, separator: str = ' ', multiline_parameter_1: int = 1000, multiline_parameter_2: int = 2000) -> str:"
            in forward_source
        )
        assert "parts = [text] * multiplier" in forward_source
        assert "return separator.join(parts)" in forward_source
        # Should not contain any decorator lines
        assert "@tool" not in forward_source
        assert "@dummy_decorator_1" not in forward_source
        assert "@dummy_decorator_2" not in forward_source
        # Should not contain leftover lines from the original multiline function definition
        assert "            multiplier: int = 2," not in forward_source
        assert "            multiplier: int=2," not in forward_source


@pytest.fixture
def mock_server_parameters():
    return MagicMock()


@pytest.fixture
def mock_mcp_adapt():
    with patch("mcpadapt.core.MCPAdapt") as mock:
        mock.return_value.__enter__.return_value = ["tool1", "tool2"]
        mock.return_value.__exit__.return_value = None
        yield mock


@pytest.fixture
def mock_smolagents_adapter():
    with patch("mcpadapt.smolagents_adapter.SmolAgentsAdapter") as mock:
        yield mock


class TestToolCollection:
    def test_from_mcp(self, mock_server_parameters, mock_mcp_adapt, mock_smolagents_adapter):
        with ToolCollection.from_mcp(mock_server_parameters, trust_remote_code=True) as tool_collection:
            assert isinstance(tool_collection, ToolCollection)
            assert len(tool_collection.tools) == 2
            assert "tool1" in tool_collection.tools
            assert "tool2" in tool_collection.tools

    @require_run_all
    def test_integration_from_mcp(self):
        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server")

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run()
        """).strip()

        mcp_server_params = mcp.StdioServerParameters(
            command="python",
            args=["-c", mcp_server_script],
        )

        with ToolCollection.from_mcp(mcp_server_params, trust_remote_code=True) as tool_collection:
            assert len(tool_collection.tools) == 1, "Expected 1 tool"
            assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
            assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"

    def test_integration_from_mcp_with_streamable_http(self):
        import subprocess
        import time

        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run(transport="streamable-http")
        """).strip()

        # start the SSE mcp server in a subprocess
        server_process = subprocess.Popen(
            ["python", "-c", mcp_server_script],
        )

        # wait for the server to start
        time.sleep(1)

        try:
            with ToolCollection.from_mcp(
                {"url": "http://127.0.0.1:8000/mcp", "transport": "streamable-http"}, trust_remote_code=True
            ) as tool_collection:
                assert len(tool_collection.tools) == 1, "Expected 1 tool"
                assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
                assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"
        finally:
            # clean up the process when test is done
            server_process.kill()
            server_process.wait()

    def test_integration_from_mcp_with_sse(self):
        import subprocess
        import time

        # define the most simple mcp server with one tool that echoes the input text
        mcp_server_script = dedent("""\
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("Echo Server", host="127.0.0.1", port=8000)

            @mcp.tool()
            def echo_tool(text: str) -> str:
                return text

            mcp.run("sse")
        """).strip()

        # start the SSE mcp server in a subprocess
        server_process = subprocess.Popen(
            ["python", "-c", mcp_server_script],
        )

        # wait for the server to start
        time.sleep(1)

        try:
            with ToolCollection.from_mcp(
                {"url": "http://127.0.0.1:8000/sse", "transport": "sse"}, trust_remote_code=True
            ) as tool_collection:
                assert len(tool_collection.tools) == 1, "Expected 1 tool"
                assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
                assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"
        finally:
            # clean up the process when test is done
            server_process.kill()
            server_process.wait()


@pytest.mark.parametrize("tool_fixture_name", ["boolean_default_tool_class"])
def test_launch_gradio_demo_does_not_raise(tool_fixture_name, request):
    tool = request.getfixturevalue(tool_fixture_name)
    with patch("gradio.Interface.launch") as mock_launch:
        launch_gradio_demo(tool)
    assert mock_launch.call_count == 1


@pytest.mark.parametrize(
    "tool_input_type, expected_input, expects_error",
    [
        (bool, True, False),
        (str, "b", False),
        (int, 1, False),
        (float, 1, False),
        (list, ["a", "b"], False),
        (list[str], ["a", "b"], False),
        (dict[str, str], {"a": "b"}, False),
        (dict[str, str], "b", True),
        (bool, "b", True),
        (str | int, "a", False),
        (str | int, 1, False),
        (str | int, None, True),
        (str | int, True, True),
    ],
)
def test_validate_tool_arguments(tool_input_type, expected_input, expects_error):
    @tool
    def test_tool(argument_a: tool_input_type) -> str:
        """Fake tool

        Args:
            argument_a: The input
        """
        return argument_a

    if expects_error:
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(test_tool, {"argument_a": expected_input})

    else:
        # Should not raise any exception
        validate_tool_arguments(test_tool, {"argument_a": expected_input})


@pytest.mark.parametrize(
    "scenario, type_hint, default, input_value, expected_error_message",
    [
        # Required parameters (no default)
        # - Valid input
        ("required_unsupported_none", str, ..., "text", None),
        # - None not allowed
        ("required_unsupported_none", str, ..., None, "Argument param has type 'null' but should be 'string'"),
        # - Missing required parameter is not allowed
        ("required_unsupported_none", str, ..., ..., "Argument param is required"),
        #
        # Required parameters but supports None
        # - Valid input
        ("required_supported_none", str | None, ..., "text", None),
        # - None allowed
        ("required_supported_none", str | None, ..., None, None),
        # - Missing required parameter is not allowed
        # TODO: Fix this test case: property is marked as nullable because it can be None, but it can't be missing because it is required
        # ("required_supported_none", str | None, ..., ..., "Argument param is required"),
        pytest.param(
            "required_supported_none",
            str | None,
            ...,
            ...,
            "Argument param is required",
            marks=pytest.mark.skip(reason="TODO: Fix this test case"),
        ),
        #
        # Optional parameters (has default, doesn't support None)
        # - Valid input
        ("optional_unsupported_none", str, "default", "text", None),
        # - None not allowed
        # TODO: Fix this test case: property is marked as nullable because it has a default value, but it can't be None
        # ("optional_unsupported_none", str, "default", None, "Argument param has type 'null' but should be 'string'"),
        pytest.param(
            "optional_unsupported_none",
            str,
            "default",
            None,
            "Argument 'param' cannot be 'null'",
            marks=pytest.mark.skip(reason="TODO: Fix this test case"),
        ),
        # - Missing optional parameter is allowed
        ("optional_unsupported_none", str, "default", ..., None),
        #
        # Optional and supports None parameters with string default
        # - Valid input
        ("optional_supported_none_str_default", str | None, "default", "text", None),
        # - None allowed
        ("optional_supported_none_str_default", str | None, "default", None, None),
        # - Missing optional parameter is allowed
        ("optional_supported_none_str_default", str | None, "default", ..., None),
        #
        # Optional and supports None parameters with None default
        # - Valid input
        ("optional_supported_none_none_default", str | None, None, "text", None),
        # - None allowed
        ("optional_supported_none_none_default", str | None, None, None, None),
        # - Missing optional parameter is allowed
        ("optional_supported_none_none_default", str | None, None, ..., None),
    ],
)
def test_validate_tool_arguments_nullable(scenario, type_hint, default, input_value, expected_error_message):
    """Test validation of tool arguments with focus on nullable properties: optional (with default value) and supporting None value.

    Args:
        scenario: The scenario to test
        type_hint: The type hint for the parameter
        default: The default value for the parameter
        input_value: The input value for the parameter
        expected_error_message: The expected error message
    """

    # Create a tool with the appropriate signature
    if default is ...:  # Using Ellipsis to indicate no default value

        @tool
        def test_tool(param: type_hint) -> str:
            """Test tool.

            Args:
                param: Input param
            """
            return str(param) if param is not None else "NULL"
    else:

        @tool
        def test_tool(param: type_hint = default) -> str:
            """Test tool.

            Args:
                param: Input param.
            """
            return str(param) if param is not None else "NULL"

    # Test with the input dictionary
    input_dict = {"param": input_value} if input_value is not ... else {}

    print("INPUT", input_dict, type_hint, default)

    if expected_error_message:
        with pytest.raises((ValueError, TypeError), match=expected_error_message):
            validate_tool_arguments(test_tool, input_dict)
    else:
        # Should not raise any exception
        validate_tool_arguments(test_tool, input_dict)


class TestPydanticToolIntegration:
    """Test Pydantic BaseModel integration with Tool creation and validation."""

    @pytest.fixture
    def pydantic_available(self):
        """Check if Pydantic is available for testing."""
        try:
            import importlib.util

            if importlib.util.find_spec("pydantic") is None:
                pytest.skip("Pydantic not available")
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_tool_with_pydantic_model(self, pydantic_available):
        """Test creating a tool that uses complex nested Pydantic models with constraints."""
        import pydantic
        from pydantic import Field

        class Address(pydantic.BaseModel):
            """Address information."""

            street: str = Field(..., min_length=1, max_length=200, description="Street address")
            city: str = Field(..., min_length=1, max_length=100, description="City name")
            postal_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$", description="US postal code")
            country: str = Field(default="US", description="Country code")

        class PersonInfo(pydantic.BaseModel):
            """Information about a person."""

            name: str = Field(..., min_length=1, max_length=100, description="Person's full name")
            age: int = Field(..., ge=0, le=150, description="Age in years")
            email: str | None = Field(None, pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Email address")
            address: Address = Field(..., description="Primary address")
            secondary_addresses: list[Address] = Field(
                default_factory=list, max_length=3, description="Additional addresses"
            )
            is_active: bool = Field(default=True, description="Whether the person is active")

        @tool
        def process_person(person: PersonInfo) -> str:
            """
            Process comprehensive information about a person including nested address data.

            Args:
                person: Complete person information with address details

            Returns:
                A formatted string with the person's information including addresses
            """
            addr = person.address
            addr_str = f"{addr.street}, {addr.city}, {addr.postal_code}, {addr.country}"
            email_part = f" (email: {person.email})" if person.email else ""
            secondary_count = len(person.secondary_addresses)
            secondary_part = f" with {secondary_count} additional addresses" if secondary_count > 0 else ""
            status = "active" if person.is_active else "inactive"
            return f"Person: {person.name}, age {person.age}{email_part}, {status}, at {addr_str}{secondary_part}"

        # Test tool creation
        assert process_person.name == "process_person"
        assert isinstance(process_person.inputs, dict)
        assert "person" in process_person.inputs

        # Check that the complex nested Pydantic schema was properly converted
        person_schema = process_person.inputs["person"]
        assert isinstance(person_schema, dict)
        assert person_schema["type"] == "object"

        # Verify nested address schema is properly included
        properties = person_schema["properties"]
        assert "address" in properties
        address_schema = properties["address"]
        assert address_schema["type"] == "object"
        assert "properties" in address_schema

        # Verify address properties have constraints
        address_props = address_schema["properties"]
        assert "street" in address_props
        street_schema = address_props["street"]
        assert street_schema["type"] == "string"
        assert street_schema["minLength"] == 1
        assert street_schema["maxLength"] == 200

        # Verify postal code has pattern constraint
        postal_schema = address_props["postal_code"]
        assert postal_schema["type"] == "string"
        assert "pattern" in postal_schema
        assert postal_schema["pattern"] == r"^\d{5}(-\d{4})?$"

        # Verify age has numerical constraints
        age_schema = properties["age"]
        assert age_schema["type"] == "integer"
        assert age_schema["minimum"] == 0
        assert age_schema["maximum"] == 150

        # Verify array handling for secondary addresses
        secondary_schema = properties["secondary_addresses"]
        assert secondary_schema["type"] == "array"
        assert secondary_schema["maxItems"] == 3
        assert "items" in secondary_schema
        assert secondary_schema["items"]["type"] == "object"

        class ProcessPersonTool(Tool):
            name = "process_person_class"
            description = (
                "Process comprehensive information about a person including nested address data using Tool subclass"
            )
            inputs = {"person": PersonInfo}
            output_type = "string"

            def forward(self, person: PersonInfo) -> str:
                addr = person.address
                addr_str = f"{addr.street}, {addr.city}, {addr.postal_code}, {addr.country}"
                email_part = f" (email: {person.email})" if person.email else ""
                secondary_count = len(person.secondary_addresses)
                secondary_part = f" with {secondary_count} additional addresses" if secondary_count > 0 else ""
                status = "active" if person.is_active else "inactive"
                return f"Person: {person.name}, age {person.age}{email_part}, {status}, at {addr_str}{secondary_part}"

        # Instantiate the class-based tool
        process_person_class = ProcessPersonTool()

        # Verify Tool subclass creates the same schema structure
        assert process_person_class.name == "process_person_class"
        assert isinstance(process_person_class.inputs, dict)
        assert "person" in process_person_class.inputs

        # Check that the Tool subclass generated the same complex nested Pydantic schema
        person_schema_class = process_person_class.inputs["person"]
        assert isinstance(person_schema_class, dict)
        assert person_schema_class["type"] == "object"

        # Verify the schemas are equivalent between decorator and class approaches
        properties_class = person_schema_class["properties"]

        # Compare key properties to ensure both approaches generate the same schema
        assert set(properties.keys()) == set(properties_class.keys())

        # Verify nested address schema is identical
        address_schema_class = properties_class["address"]
        assert address_schema_class["type"] == "object"
        assert "properties" in address_schema_class

        # Compare address properties
        address_props_class = address_schema_class["properties"]
        assert set(address_props.keys()) == set(address_props_class.keys())

        # Verify constraints are preserved in class approach
        street_schema_class = address_props_class["street"]
        assert street_schema_class["type"] == "string"
        assert street_schema_class["minLength"] == 1
        assert street_schema_class["maxLength"] == 200

        postal_schema_class = address_props_class["postal_code"]
        assert postal_schema_class["type"] == "string"
        assert "pattern" in postal_schema_class
        assert postal_schema_class["pattern"] == r"^\d{5}(-\d{4})?$"

        age_schema_class = properties_class["age"]
        assert age_schema_class["type"] == "integer"
        assert age_schema_class["minimum"] == 0
        assert age_schema_class["maximum"] == 150

    def test_pydantic_tool_validation_success(self, pydantic_available):
        """Test successful validation of Pydantic tool arguments."""
        import pydantic

        class UserData(pydantic.BaseModel):
            username: str
            active: bool = True

        @tool
        def process_user(user: UserData) -> str:
            """
            Process user data.

            Args:
                user: User data to process

            Returns:
                User summary
            """
            return f"User {user.username} is {'active' if user.active else 'inactive'}"

        # Test with valid input
        valid_input = {"user": {"username": "alice", "active": True}}

        # Should not raise any exception
        validate_tool_arguments(process_user, valid_input)

    def test_pydantic_tool_validation_with_optional_fields(self, pydantic_available):
        """Test validation with optional Pydantic fields."""
        import pydantic

        class ProfileData(pydantic.BaseModel):
            """Profile data for a user."""
            name: str
            bio: str | None = None
            age: int | None = None

        @tool
        def create_profile(profile: ProfileData) -> str:
            """
            Create a user profile.

            Args:
                profile: Profile data

            Returns:
                Profile summary
            """
            return f"Profile for {profile.name}"

        # Test with minimal required fields
        minimal_input = {"profile": {"name": "Bob"}}

        # Should not raise any exception
        validate_tool_arguments(create_profile, minimal_input)

        # Test with all fields
        complete_input = {"profile": {"name": "Alice", "bio": "Software engineer", "age": 30}}

        # Should not raise any exception
        validate_tool_arguments(create_profile, complete_input)

        # Test that optional fields are properly marked as nullable in schema
        schema = create_profile.inputs["profile"]
        properties = schema["properties"]
        required_fields = schema.get("required", [])

        # Check required field
        assert "name" in required_fields
        assert "nullable" not in properties["name"]

        # Check optional fields are marked as nullable
        assert "bio" not in required_fields
        assert properties["bio"].get("nullable") is True

        assert "age" not in required_fields
        assert properties["age"].get("nullable") is True

        # Test with Tool subclass to ensure same behavior
        class ProfileTool(Tool):
            name = "profile_tool"
            description = "Tool with optional fields"
            inputs = {"profile": ProfileData}
            output_type = "string"

            def forward(self, profile: ProfileData) -> str:
                return f"Profile for {profile.name}"

        tool_instance = ProfileTool()
        schema_class = tool_instance.inputs["profile"]
        properties_class = schema_class["properties"]
        required_fields_class = schema_class.get("required", [])

        # Check the Tool subclass generates the same nullable behavior
        assert "name" in required_fields_class
        assert "nullable" not in properties_class["name"]

        assert "bio" not in required_fields_class
        assert properties_class["bio"].get("nullable") is True

        assert "age" not in required_fields_class
        assert properties_class["age"].get("nullable") is True

    def test_pydantic_tool_validation_failure(self, pydantic_available):
        """Test validation failures with Pydantic tool arguments."""
        import pydantic

        class StrictData(pydantic.BaseModel):
            count: int
            message: str

        @tool
        def process_strict(data: StrictData) -> str:
            """
            Process strict data.

            Args:
                data: Strict data requirements

            Returns:
                Processing result
            """
            return f"Processed {data.count}: {data.message}"

        # Test with missing required field
        invalid_input = {
            "data": {
                "count": 5
                # missing "message"
            }
        }

        # Should raise validation error for missing required field
        with pytest.raises(ValueError, match="Required property.*missing"):
            validate_tool_arguments(process_strict, invalid_input)

        # Test Tool subclass with Pydantic model missing docstring
        class ModelWithoutDocstring(pydantic.BaseModel):
            field: str

        class InvalidTool(Tool):
            name = "invalid_tool"
            description = "Tool with invalid Pydantic model"
            inputs = {"data": ModelWithoutDocstring}
            output_type = "string"

            def forward(self, data):
                return str(data.field)

        # Should raise ValueError for missing docstring
        with pytest.raises(ValueError, match="must have a docstring to provide a description"):
            InvalidTool()

    def test_pydantic_tool_validation_error(self, pydantic_available):
        """Test that incorrect Pydantic objects and incompatible dicts both raise validation errors."""
        import pydantic
        from pydantic import Field, ValidationError

        class ContactInfo(pydantic.BaseModel):
            email: str = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$", description="Valid email address")
            phone: str = Field(..., min_length=10, max_length=15, description="Phone number")

        class UserProfile(pydantic.BaseModel):
            name: str = Field(..., min_length=1, max_length=50, description="User name")
            age: int = Field(..., ge=18, le=120, description="User age")
            contact: ContactInfo = Field(..., description="Contact information")
            active: bool = Field(default=True, description="Is user active")

        @tool
        def create_user_profile(profile: UserProfile) -> str:
            """
            Create a user profile with validation.

            Args:
                profile: User profile data with contact info

            Returns:
                Profile creation result
            """
            return f"Created profile for {profile.name}, age {profile.age}"

        # Test 1: Invalid dictionary - constraint violations
        invalid_dict_constraints = {
            "profile": {
                "name": "",  # violates min_length=1
                "age": 15,  # violates ge=18
                "contact": {
                    "email": "invalid-email",  # violates email pattern
                    "phone": "123",  # violates min_length=10
                },
                "active": True,
            }
        }

        # Should raise validation error for constraint violations
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(create_user_profile, invalid_dict_constraints)

        # Test 2: Invalid dictionary - missing required fields
        invalid_dict_missing = {
            "profile": {
                "name": "John Doe"
                # missing required 'age' and 'contact' fields
            }
        }

        # Should raise validation error for missing required fields
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(create_user_profile, invalid_dict_missing)

        # Test 3: Invalid dictionary - wrong types
        invalid_dict_types = {
            "profile": {
                "name": "John Doe",
                "age": "not-a-number",  # wrong type
                "contact": {"email": "john@example.com", "phone": "1234567890"},
            }
        }

        # Should raise validation error for wrong types
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(create_user_profile, invalid_dict_types)

        # Test 4: Invalid nested dictionary structure
        invalid_dict_nested = {
            "profile": {
                "name": "John Doe",
                "age": 25,
                "contact": "not-an-object",  # should be an object
            }
        }

        # Should raise validation error for invalid nested structure
        with pytest.raises((ValueError, TypeError)):
            validate_tool_arguments(create_user_profile, invalid_dict_nested)

        # Test 5: Direct Pydantic validation error during conversion
        # This tests the _convert_dict_args_to_pydantic_models method
        try:
            # This should trigger a ValidationError during Pydantic model creation
            invalid_contact_data = {
                "email": "bad-email-format",
                "phone": "123",  # too short
            }
            invalid_profile_data = {"name": "John Doe", "age": 25, "contact": invalid_contact_data}

            # Try to create the tool with invalid data - this should fail
            create_user_profile(profile=invalid_profile_data)
            assert False, "Expected ValidationError but none was raised"

        except (ValidationError, ValueError, TypeError):
            # This is expected - the validation should catch the errors
            assert True

    def test_pydantic_tool_with_enum_constraints(self, pydantic_available):
        """Test Pydantic tool with enum field constraints."""
        import pydantic

        class Priority(str, Enum):
            LOW = "low"
            MEDIUM = "medium"
            HIGH = "high"

        class TaskData(pydantic.BaseModel):
            title: str
            priority: Priority = Priority.MEDIUM

        @tool
        def create_task(task: TaskData) -> str:
            """
            Create a task.

            Args:
                task: Task data

            Returns:
                Task summary
            """
            return f"Task '{task.title}' with priority {task.priority.value}"

        # Test with valid enum value
        valid_input = {"task": {"title": "Fix bug", "priority": "high"}}

        # Should not raise any exception
        validate_tool_arguments(create_task, valid_input)

        # Test with invalid enum value
        invalid_input = {
            "task": {
                "title": "Fix bug",
                "priority": "urgent",  # Not in enum
            }
        }

        # Should raise validation error for invalid enum value
        with pytest.raises(ValueError, match="not in allowed values"):
            validate_tool_arguments(create_task, invalid_input)

    def test_pydantic_tool_with_nested_models(self, pydantic_available):
        """Test Pydantic tool with nested model structures."""
        import pydantic

        class Address(pydantic.BaseModel):
            street: str
            city: str

        class Contact(pydantic.BaseModel):
            name: str
            address: Address

        @tool
        def process_contact(contact: Contact) -> str:
            """
            Process contact information.

            Args:
                contact: Contact data

            Returns:
                Contact summary
            """
            return f"{contact.name} lives at {contact.address.street}, {contact.address.city}"

        # Test with valid nested structure
        valid_input = {"contact": {"name": "John Doe", "address": {"street": "123 Main St", "city": "Anytown"}}}

        # Should not raise any exception
        validate_tool_arguments(process_contact, valid_input)

        # Test with missing nested field
        invalid_input = {
            "contact": {
                "name": "John Doe",
                "address": {
                    "street": "123 Main St"
                    # missing "city"
                },
            }
        }

        # Should raise validation error for missing nested field
        with pytest.raises(ValueError, match="Required property.*missing"):
            validate_tool_arguments(process_contact, invalid_input)
