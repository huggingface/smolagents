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

import pytest

from smolagents import CodeAgent


class FakeLangChainChatOpenAI:
    model_id = "fake-langchain-chat-openai"

    def generate(self, messages, stop_sequences=None):
        for message_list in messages:
            for _ in message_list:
                pass


def test_issue_1968():
    model = FakeLangChainChatOpenAI()
    agent = CodeAgent(model=model, tools=[], planning_interval=3)

    with pytest.raises(TypeError, match="'ChatMessage' object is not iterable"):
        agent.run("hello")
