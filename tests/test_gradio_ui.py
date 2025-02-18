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

from smolagents.gradio_ui import GradioUI


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
