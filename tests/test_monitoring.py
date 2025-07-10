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

import PIL.Image
import pytest

from smolagents import (
    CodeAgent,
    RunResult,
    ToolCallingAgent,
    stream_to_gradio,
)
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    Model,
    TokenUsage,
)


class FakeLLMModel(Model):
    def __init__(self, give_token_usage: bool = True):
        self.give_token_usage = give_token_usage

    def generate(self, prompt, tools_to_call_from=None, **kwargs):
        if tools_to_call_from is not None:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="fake_id",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "image"}),
                    )
                ],
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""<code>
final_answer('This is the final answer.')
</code>""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )


class MonitoringTester(unittest.TestCase):
    def test_code_agent_metrics(self):
        agent = CodeAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
        )
        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_toolcalling_agent_metrics(self):
        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 10)
        self.assertEqual(agent.monitor.total_output_token_count, 20)

    def test_code_agent_metrics_max_steps(self):
        class FakeLLMModelMalformedAnswer(Model):
            def generate(self, prompt, **kwargs):
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Malformed answer",
                    token_usage=TokenUsage(input_tokens=10, output_tokens=20),
                )

        agent = CodeAgent(
            tools=[],
            model=FakeLLMModelMalformedAnswer(),
            max_steps=1,
        )

        agent.run("Fake task")

        self.assertEqual(agent.monitor.total_input_token_count, 20)
        self.assertEqual(agent.monitor.total_output_token_count, 40)

    def test_code_agent_metrics_generation_error(self):
        class FakeLLMModelGenerationException(Model):
            def generate(self, prompt, **kwargs):
                raise Exception("Cannot generate")

        agent = CodeAgent(
            tools=[],
            model=FakeLLMModelGenerationException(),
            max_steps=1,
        )
        with pytest.raises(Exception) as e:
            agent.run("Fake task")
        assert "Cannot generate" in str(e.value)

    def test_streaming_agent_text_output(self):
        agent = CodeAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            planning_interval=2,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        self.assertEqual(len(outputs), 11)
        plan_message = outputs[1]
        self.assertEqual(plan_message.role, "assistant")
        self.assertIn("<code>", plan_message.content)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("This is the final answer.", final_message.content)

    def test_streaming_agent_image_output(self):
        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            verbosity_level=100,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(
            stream_to_gradio(
                agent,
                task="Test task",
                additional_args=dict(image=PIL.Image.new("RGB", (100, 100))),
            )
        )

        self.assertEqual(len(outputs), 7)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIsInstance(final_message.content, dict)
        self.assertEqual(final_message.content["mime_type"], "image/png")

    def test_streaming_with_agent_error(self):
        class DummyModel(Model):
            def generate(self, prompt, **kwargs):
                return ChatMessage(role=MessageRole.ASSISTANT, content="Malformed call")

        agent = CodeAgent(
            tools=[],
            model=DummyModel(),
            max_steps=1,
        )

        # Use stream_to_gradio to capture the output
        outputs = list(stream_to_gradio(agent, task="Test task"))

        self.assertEqual(len(outputs), 11)
        final_message = outputs[-1]
        self.assertEqual(final_message.role, "assistant")
        self.assertIn("Malformed call", final_message.content)

    def test_run_return_full_result(self):
        agent = CodeAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "This is the final answer.")
        self.assertEqual(result.state, "success")
        self.assertEqual(result.token_usage, TokenUsage(input_tokens=10, output_tokens=20))
        self.assertIsInstance(result.messages, list)
        self.assertGreater(result.timing.duration, 0)

        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "image")
        self.assertEqual(result.state, "success")
        self.assertEqual(result.token_usage, TokenUsage(input_tokens=10, output_tokens=20))
        self.assertIsInstance(result.messages, list)
        self.assertGreater(result.timing.duration, 0)

        # Below 2 lines should be removed when the attributes are removed
        assert agent.monitor.total_input_token_count == 10
        assert agent.monitor.total_output_token_count == 20

    def test_run_result_no_token_usage(self):
        agent = CodeAgent(
            tools=[],
            model=FakeLLMModel(give_token_usage=False),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "This is the final answer.")
        self.assertEqual(result.state, "success")
        self.assertIsNone(result.token_usage)
        self.assertIsInstance(result.messages, list)
        self.assertGreater(result.timing.duration, 0)

        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(give_token_usage=False),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.output, "image")
        self.assertEqual(result.state, "success")
        self.assertIsNone(result.token_usage)
        self.assertIsInstance(result.messages, list)
        self.assertGreater(result.timing.duration, 0)
