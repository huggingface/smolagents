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


import numpy as np
import PIL.Image
import pytest
from pydantic import BaseModel, ValidationError

from smolagents.agent_types import _AGENT_TYPE_MAPPING
from smolagents.default_tools import FinalAnswerTool, PydanticFinalAnswerTool

from .test_tools import ToolTesterMixin
from .utils.markers import require_torch


class TestFinalAnswerTool(ToolTesterMixin):
    def setup_method(self):
        self.inputs = {"answer": "Final answer"}
        self.tool = FinalAnswerTool()

    def test_exact_match_arg(self):
        result = self.tool("Final answer")
        assert result == "Final answer"

    def test_exact_match_kwarg(self):
        result = self.tool(answer=self.inputs["answer"])
        assert result == "Final answer"

    @require_torch
    def test_agent_type_output(self, inputs):
        for input_type, input in inputs.items():
            output = self.tool(**input, sanitize_inputs_outputs=True)
            agent_type = _AGENT_TYPE_MAPPING[input_type]
            assert isinstance(output, agent_type)

    @pytest.fixture
    def inputs(self, shared_datadir):
        import torch

        return {
            "string": {"answer": "Text input"},
            "image": {"answer": PIL.Image.open(shared_datadir / "000000039769.png").resize((512, 512))},
            "audio": {"answer": torch.Tensor(np.ones(3000))},
        }


class TestPydanticFinalAnswerTool(ToolTesterMixin):
    def setup_method(self):
        class TestModel(BaseModel):
            name: str
            age: int
            is_active: bool = True

        self.output_model = TestModel
        self.tool = PydanticFinalAnswerTool(output_model=self.output_model)
        self.valid_dict = {"answer": {"name": "John", "age": 30, "is_active": True}}

    def test_agent_type_output(self):
        result = self.tool(self.valid_dict)
        assert isinstance(result, self.output_model)
        assert result.name == "John"
        assert result.age == 30
        assert result.is_active is True

    def test_invalid_dict_input(self):
        invalid_dict = {"answer": {"name": "John", "age": "not_an_integer"}}
        with pytest.raises(ValidationError) as exc_info:
            self.tool(**invalid_dict)
        assert "age" in str(exc_info.value)

    def test_invalid_type_input(self):
        with pytest.raises(AssertionError) as exc_info:
            self.tool(123)
        assert "The answer must be a dictionary" in str(exc_info.value)

    def test_missing_required_field(self):
        invalid_dict = {"answer": {"name": "John"}}
        with pytest.raises(ValidationError) as exc_info:
            self.tool(**invalid_dict)
        assert "age" in str(exc_info.value)

    def test_optional_field_default(self):
        minimal_dict = {"answer": {"name": "John", "age": 30}}
        result = self.tool(**minimal_dict)
        assert result.is_active is True
