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

import os
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import pytest

from smolagents.agent_types import AgentAudio, AgentImage, AgentText
from smolagents.gradio_ui import GradioUI, pull_messages_from_step, stream_to_gradio
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
from smolagents.models import ChatMessageStreamDelta
from smolagents.monitoring import Timing, TokenUsage


class GradioUITester(unittest.TestCase):
    def setUp(self):
        """Initialize test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_agent = Mock()
        self.ui = GradioUI(agent=self.mock_agent, file_upload_folder=self.temp_dir)
        self.allowed_types = [".pdf", ".docx", ".txt"]

    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)

    def test_upload_file_default_types(self):
        """Test default allowed file types"""
        default_types = [".pdf", ".docx", ".txt"]
        for file_type in default_types:
            with tempfile.NamedTemporaryFile(suffix=file_type) as temp_file:
                mock_file = Mock()
                mock_file.name = temp_file.name

                textbox, uploads_log = self.ui.upload_file(mock_file, [])

                self.assertIn("File uploaded:", textbox.value)
                self.assertEqual(len(uploads_log), 1)
                self.assertTrue(os.path.exists(os.path.join(self.temp_dir, os.path.basename(temp_file.name))))

    def test_upload_file_default_types_disallowed(self):
        """Test default disallowed file types"""
        disallowed_types = [".exe", ".sh", ".py", ".jpg"]
        for file_type in disallowed_types:
            with tempfile.NamedTemporaryFile(suffix=file_type) as temp_file:
                mock_file = Mock()
                mock_file.name = temp_file.name

                textbox, uploads_log = self.ui.upload_file(mock_file, [])

                self.assertEqual(textbox.value, "File type disallowed")
                self.assertEqual(len(uploads_log), 0)

    def test_upload_file_success(self):
        """Test successful file upload scenario"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [])

            self.assertIn("File uploaded:", textbox.value)
            self.assertEqual(len(uploads_log), 1)
            self.assertTrue(os.path.exists(os.path.join(self.temp_dir, os.path.basename(temp_file.name))))
            self.assertEqual(uploads_log[0], os.path.join(self.temp_dir, os.path.basename(temp_file.name)))

    def test_upload_file_none(self):
        """Test scenario when no file is selected"""
        textbox, uploads_log = self.ui.upload_file(None, [])

        self.assertEqual(textbox.value, "No file uploaded")
        self.assertEqual(len(uploads_log), 0)

    def test_upload_file_invalid_type(self):
        """Test disallowed file type"""
        with tempfile.NamedTemporaryFile(suffix=".exe") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [])

            self.assertEqual(textbox.value, "File type disallowed")
            self.assertEqual(len(uploads_log), 0)

    def test_upload_file_special_chars(self):
        """Test scenario with special characters in filename"""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            # Create a new temporary file with special characters
            special_char_name = os.path.join(os.path.dirname(temp_file.name), "test@#$%^&*.txt")
            shutil.copy(temp_file.name, special_char_name)
            try:
                mock_file = Mock()
                mock_file.name = special_char_name

                with patch("shutil.copy"):
                    textbox, uploads_log = self.ui.upload_file(mock_file, [])

                    self.assertIn("File uploaded:", textbox.value)
                    self.assertEqual(len(uploads_log), 1)
                    self.assertIn("test_____", uploads_log[0])
            finally:
                # Clean up the special character file
                if os.path.exists(special_char_name):
                    os.remove(special_char_name)

    def test_upload_file_custom_types(self):
        """Test custom allowed file types"""
        with tempfile.NamedTemporaryFile(suffix=".csv") as temp_file:
            mock_file = Mock()
            mock_file.name = temp_file.name

            textbox, uploads_log = self.ui.upload_file(mock_file, [], allowed_file_types=[".csv"])

            self.assertIn("File uploaded:", textbox.value)
            self.assertEqual(len(uploads_log), 1)


class TestStreamToGradio:
    """Tests for the stream_to_gradio function."""

    @patch("smolagents.gradio_ui.pull_messages_from_step")
    def test_stream_to_gradio_memory_step(self, mock_pull_messages):
        """Test streaming a memory step"""
        # Create mock agent and memory step
        mock_agent = Mock()
        mock_agent.run = Mock(return_value=[Mock(spec=ActionStep)])
        mock_agent.model = Mock()
        # Mock the pull_messages_from_step function to return some messages
        mock_message = Mock()
        mock_pull_messages.return_value = [mock_message]
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that pull_messages_from_step was called and the message was yielded
        mock_pull_messages.assert_called_once()
        assert result == [mock_message]

    def test_stream_to_gradio_stream_delta(self):
        """Test streaming a ChatMessageStreamDelta"""
        # Create mock agent and stream delta
        mock_agent = Mock()
        mock_delta = ChatMessageStreamDelta(content="Hello")
        mock_agent.run = Mock(return_value=[mock_delta])
        mock_agent.model = Mock()
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that the content was yielded
        assert result == ["Hello"]

    def test_stream_to_gradio_multiple_deltas(self):
        """Test streaming multiple ChatMessageStreamDeltas"""
        # Create mock agent and stream deltas
        mock_agent = Mock()
        mock_delta1 = ChatMessageStreamDelta(content="Hello")
        mock_delta2 = ChatMessageStreamDelta(content=" world")
        mock_agent.run = Mock(return_value=[mock_delta1, mock_delta2])
        mock_agent.model = Mock()
        # Call stream_to_gradio
        result = list(stream_to_gradio(mock_agent, "test task"))
        # Verify that the content was accumulated and yielded
        assert result == ["Hello", "Hello world"]

    @pytest.mark.parametrize(
        "task,task_images,reset_memory,additional_args",
        [
            ("simple task", None, False, None),
            ("task with images", ["image1.png", "image2.png"], False, None),
            ("task with reset", None, True, None),
            ("task with args", None, False, {"arg1": "value1"}),
            ("complex task", ["image.png"], True, {"arg1": "value1", "arg2": "value2"}),
        ],
    )
    def test_stream_to_gradio_parameters(self, task, task_images, reset_memory, additional_args):
        """Test that stream_to_gradio passes parameters correctly to agent.run"""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.run = Mock(return_value=[])
        # Call stream_to_gradio
        list(
            stream_to_gradio(
                mock_agent,
                task=task,
                task_images=task_images,
                reset_agent_memory=reset_memory,
                additional_args=additional_args,
            )
        )
        # Verify that agent.run was called with the right parameters
        mock_agent.run.assert_called_once_with(
            task, images=task_images, stream=True, reset=reset_memory, additional_args=additional_args
        )


class TestPullMessagesFromStep:
    def test_action_step_basic(
        self,
    ):
        """Test basic ActionStep processing."""
        step = ActionStep(
            step_number=1,
            model_output="This is the model output",
            observations="Some execution logs",
            error=None,
            timing=Timing(start_time=1.0, end_time=3.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 5  # step number, model_output, logs, footnote, divider
        for message, expected_content in zip(
            messages,
            [
                "**Step 1**",
                "This is the model output",
                "execution logs",
                "Input tokens: 100 | Output tokens: 50 | Duration: 2.5",
                "-----",
            ],
        ):
            assert expected_content in message.content

    def test_action_step_with_tool_calls(self):
        """Test ActionStep with tool calls."""
        step = ActionStep(
            step_number=2,
            tool_calls=[ToolCall(name="test_tool", arguments={"answer": "Test answer"}, id="tool_call_1")],
            observations="Tool execution logs",
            timing=Timing(start_time=1.0, end_time=2.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 5  # step, tool call, logs, footnote, divider
        assert messages[1].content == "Test answer"
        assert "Used tool test_tool" in messages[1].metadata["title"]

    @pytest.mark.parametrize(
        "tool_name, args, expected",
        [
            ("python_interpreter", "print('Hello')", "```python\nprint('Hello')\n```"),
            ("regular_tool", {"key": "value"}, "{'key': 'value'}"),
            ("string_args_tool", "simple string", "simple string"),
        ],
    )
    def test_action_step_tool_call_formats(self, tool_name, args, expected):
        """Test different formats of tool calls."""
        tool_call = Mock()
        tool_call.name = tool_name
        tool_call.arguments = args
        step = ActionStep(
            step_number=1,
            tool_calls=[tool_call],
            timing=Timing(start_time=1.0, end_time=2.5),
            token_usage=TokenUsage(input_tokens=100, output_tokens=50),
        )
        messages = list(pull_messages_from_step(step))
        tool_message = next(
            msg
            for msg in messages
            if msg.role == "assistant" and msg.metadata and msg.metadata.get("title", "").startswith("🛠️")
        )
        assert expected in tool_message.content

    def test_action_step_with_error(self):
        """Test ActionStep with error."""
        step = ActionStep(
            step_number=3,
            error="This is an error message",
            timing=Timing(start_time=1.0, end_time=2.0),
            token_usage=TokenUsage(input_tokens=100, output_tokens=200),
        )
        messages = list(pull_messages_from_step(step))
        error_message = next((m for m in messages if "error" in str(m.content).lower()), None)
        assert error_message is not None
        assert "This is an error message" in error_message.content

    def test_action_step_with_images(self):
        """Test ActionStep with observation images."""
        step = ActionStep(
            step_number=4,
            observations_images=["image1.png", "image2.jpg"],
            token_usage=TokenUsage(input_tokens=100, output_tokens=200),
            timing=Timing(start_time=1.0, end_time=2.0),
        )
        with patch("smolagents.gradio_ui.AgentImage") as mock_agent_image:
            mock_agent_image.return_value.to_string.side_effect = lambda: "path/to/image.png"
            messages = list(pull_messages_from_step(step))
            image_messages = [m for m in messages if "image" in str(m).lower()]
            assert len(image_messages) == 2
            assert "path/to/image.png" in str(image_messages[0])

    @pytest.mark.parametrize(
        "skip_model_outputs, expected_messages_length, token_usage",
        [(False, 4, TokenUsage(input_tokens=80, output_tokens=30)), (True, 2, None)],
    )
    def test_planning_step(self, skip_model_outputs, expected_messages_length, token_usage):
        """Test PlanningStep processing."""
        step = PlanningStep(
            plan="1. First step\n2. Second step",
            model_input_messages=Mock(),
            model_output_message=Mock(),
            token_usage=token_usage,
            timing=Timing(start_time=1.0, end_time=2.0),
        )
        messages = list(pull_messages_from_step(step, skip_model_outputs=skip_model_outputs))
        assert len(messages) == expected_messages_length  # [header, plan,] footnote, divider
        expected_contents = [
            "**Planning step**",
            "1. First step\n2. Second step",
            "Input tokens: 80 | Output tokens: 30" if token_usage else "",
            "-----",
        ]
        for message, expected_content in zip(messages, expected_contents[-expected_messages_length:]):
            assert expected_content in message.content

        if not token_usage:
            assert "Input tokens: 80 | Output tokens: 30" not in message.content

    @pytest.mark.parametrize(
        "answer_type, answer_value, expected_content",
        [
            (AgentText, "This is a text answer", "**Final answer:**\nThis is a text answer\n"),
            (lambda: "Plain string", "Plain string", "**Final answer:** Plain string"),
        ],
    )
    def test_final_answer_step(self, answer_type, answer_value, expected_content):
        """Test FinalAnswerStep with different answer types."""
        try:
            final_answer = answer_type()
        except TypeError:
            with patch.object(answer_type, "to_string", return_value=answer_value):
                final_answer = answer_type(answer_value)
        step = FinalAnswerStep(
            output=final_answer,
        )
        messages = list(pull_messages_from_step(step))
        assert len(messages) == 1
        assert messages[0].content == expected_content

    def test_final_answer_step_image(self):
        """Test FinalAnswerStep with image answer."""
        with patch.object(AgentImage, "to_string", return_value="path/to/image.png"):
            step = FinalAnswerStep(output=AgentImage("path/to/image.png"))
            messages = list(pull_messages_from_step(step))
            assert len(messages) == 1
            assert messages[0].content["path"] == "path/to/image.png"
            assert messages[0].content["mime_type"] == "image/png"

    def test_final_answer_step_audio(self):
        """Test FinalAnswerStep with audio answer."""
        with patch.object(AgentAudio, "to_string", return_value="path/to/audio.wav"):
            step = FinalAnswerStep(output=AgentAudio("path/to/audio.wav"))
            messages = list(pull_messages_from_step(step))
            assert len(messages) == 1
            assert messages[0].content["path"] == "path/to/audio.wav"
            assert messages[0].content["mime_type"] == "audio/wav"

    def test_unsupported_step_type(self):
        """Test handling of unsupported step types."""

        class UnsupportedStep(Mock):
            pass

        step = UnsupportedStep()
        with pytest.raises(ValueError, match="Unsupported step type"):
            list(pull_messages_from_step(step))
