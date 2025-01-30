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
import json
import os
import unittest
from pathlib import Path
from typing import Optional

import pytest
from transformers.testing_utils import get_tests_dir

from smolagents import ChatMessage, HfApiModel, TransformersModel, models, tool
from smolagents.models import parse_json_if_needed


class ModelTests(unittest.TestCase):
    def test_get_json_schema_has_nullable_args(self):
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

        assert (
            "nullable" in models.get_tool_json_schema(get_weather)["function"]["parameters"]["properties"]["celsius"]
        )

    def test_chatmessage_has_model_dumps_json(self):
        message = ChatMessage("user", [{"type": "text", "text": "Hello!"}])
        data = json.loads(message.model_dump_json())
        assert data["content"] == [{"type": "text", "text": "Hello!"}]

    def test_get_hfapi_message_no_tool(self):
        model = HfApiModel(max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])

    @pytest.mark.skipif(not os.getenv("RUN_ALL"), reason="RUN_ALL environment variable not set")
    def test_get_hfapi_message_no_tool_external_provider(self):
        model = HfApiModel(model="Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=10)
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        model(messages, stop_sequences=["great"])

    def test_transformers_message_no_tool(self):
        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=5,
            device_map="auto",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]
        output = model(messages, stop_sequences=["great"]).content
        assert output == "assistant\nHello"

    def test_transformers_message_vl_no_tool(self):
        from PIL import Image

        img = Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png")
        model = TransformersModel(
            model_id="llava-hf/llava-interleave-qwen-0.5b-hf",
            max_new_tokens=5,
            device_map="auto",
            do_sample=False,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "image": img}]}]
        output = model(messages, stop_sequences=["great"]).content
        assert output == "Hello! How can"

    def test_parse_json_if_needed(self):
        args = "abc"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == "abc"

        args = '{"a": 3}'
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == {"a": 3}

        args = "3"
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3

        args = 3
        parsed_args = parse_json_if_needed(args)
        assert parsed_args == 3
