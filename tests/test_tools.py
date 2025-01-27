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
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import mcp
import numpy as np
import pytest
import torch
from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import get_tests_dir

from smolagents.tools import AUTHORIZED_TYPES, Tool, ToolCollection, tool
from smolagents.types import _AGENT_TYPE_MAPPING, AgentAudio, AgentImage, AgentText


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def create_inputs(tool_inputs: Dict[str, Dict[Union[str, type], str]]):
    inputs = {}

    for input_name, input_desc in tool_inputs.items():
        input_type = input_desc["type"]

        if input_type == "string":
            inputs[input_name] = "Text input"
        elif input_type == "image":
            inputs[input_name] = Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png").resize((512, 512))
        elif input_type == "audio":
            inputs[input_name] = np.ones(3000)
        else:
            raise ValueError(f"Invalid type requested: {input_type}")

    return inputs


def output_type(output):
    if isinstance(output, (str, AgentText)):
        return "string"
    elif isinstance(output, (Image.Image, AgentImage)):
        return "image"
    elif isinstance(output, (torch.Tensor, AgentAudio)):
        return "audio"
    else:
        raise TypeError(f"Invalid output: {output}")


class ToolTesterMixin:
    def test_inputs_output(self):
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

        inputs = self.tool.inputs
        self.assertTrue(isinstance(inputs, dict))

        for _, input_spec in inputs.items():
            self.assertTrue("type" in input_spec)
            self.assertTrue("description" in input_spec)
            self.assertTrue(input_spec["type"] in AUTHORIZED_TYPES)
            self.assertTrue(isinstance(input_spec["description"], str))

        output_type = self.tool.output_type
        self.assertTrue(output_type in AUTHORIZED_TYPES)

    def test_common_attributes(self):
        self.assertTrue(hasattr(self.tool, "description"))
        self.assertTrue(hasattr(self.tool, "name"))
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

    def test_agent_type_output(self):
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(**inputs, sanitize_inputs_outputs=True)
        if self.tool.output_type != "any":
            agent_type = _AGENT_TYPE_MAPPING[self.tool.output_type]
            self.assertTrue(isinstance(output, agent_type))


class ToolTests(unittest.TestCase):
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

    def test_saving_tool_raises_error_imports_outside_function(self):
        with pytest.raises(Exception) as e:
            import numpy as np

            @tool
            def get_current_time() -> str:
                """
                Gets the current time.
                """
                return str(np.random.random())

            get_current_time.save("output")

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
            get_current_time.save("output")

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

    def test_saving_tool_allows_no_arg_in_init(self):
        # Test one cannot save tool with additional args in init
        class FailTool(Tool):
            name = "specific"
            description = "test description"
            inputs = {"string_input": {"type": "string", "description": "input description"}}
            output_type = "string"

            def __init__(self, url):
                super().__init__(self)
                self.url = "none"

            def forward(self, string_input: str) -> str:
                return self.url + string_input

        fail_tool = FailTool("dummy_url")
        with pytest.raises(Exception) as e:
            fail_tool.save("output")
        assert "__init__" in str(e)

    def test_saving_tool_allows_no_imports_from_outside_methods(self):
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
            fail_tool.save("output")
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
        success_tool.save("output")

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

                def forward(self, location: str, celsius: Optional[bool] = False) -> str:
                    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

            GetWeatherTool()
        assert "You must set an attribute output_type" in str(e)

    def test_tool_from_decorator_optional_args(self):
        @tool
        def get_weather(location: str, celsius: Optional[bool] = False) -> str:
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

                def forward(self, location: str, celsius: Optional[bool] = False) -> str:
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

    def test_tool_supports_any_none(self):
        @tool
        def get_weather(location: Any) -> None:
            """
            Get weather in the next days at given location.

            Args:
                location: The location to get the weather for.
            """
            return

        assert get_weather.inputs["location"]["type"] == "any"

    def test_tool_supports_array(self):
        @tool
        def get_weather(locations: List[str], months: Optional[Tuple[str, str]] = None) -> Dict[str, float]:
            """
            Get weather in the next days at given locations.

            Args:
                locations: The locations to get the weather for.
                months: The months to get the weather for
            """
            return

        assert get_weather.inputs["locations"]["type"] == "array"
        assert get_weather.inputs["months"]["type"] == "array"


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
        with ToolCollection.from_mcp(mock_server_parameters) as tool_collection:
            assert isinstance(tool_collection, ToolCollection)
            assert len(tool_collection.tools) == 2
            assert "tool1" in tool_collection.tools
            assert "tool2" in tool_collection.tools

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

        with ToolCollection.from_mcp(mcp_server_params) as tool_collection:
            assert len(tool_collection.tools) == 1, "Expected 1 tool"
            assert tool_collection.tools[0].name == "echo_tool", "Expected tool name to be 'echo_tool'"
            assert tool_collection.tools[0](text="Hello") == "Hello", "Expected tool to echo the input text"
