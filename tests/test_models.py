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
import json
from typing import Optional

from smolagents import models, tool, ChatMessage, HfApiModel


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
            "nullable"
            in models.get_json_schema(get_weather)["function"]["parameters"][
                "properties"
            ]["celsius"]
        )

    def test_chatmessage_has_model_dumps_json(self):
        message = ChatMessage("user", "Hello!")
        data = json.loads(message.model_dump_json())
        assert data["content"] == "Hello!"

    def test_get_hfapi_message_no_tool(self):
        model = HfApiModel()
        messages = [{"role": "user", "content": "Hello!"}]
        model(messages, stop_sequences=["great"])
