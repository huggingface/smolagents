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

from smolagents.gradio_ui import GradioUI, stream_to_gradio
from smolagents.memory import ActionStep
from smolagents.models import ChatMessageStreamDelta


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
        mock_agent.model.last_input_token_count = 100
        mock_agent.model.last_output_token_count = 200
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
        mock_agent.model.last_input_token_count = 100
        mock_agent.model.last_output_token_count = 200
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
        mock_agent.model.last_input_token_count = 100
        mock_agent.model.last_output_token_count = 200
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
