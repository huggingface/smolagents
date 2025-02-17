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
from unittest.mock import Mock

import pytest

from smolagents.gradio_ui import GradioUI


@pytest.fixture
def file_upload_dir(tmpdir):
    return tmpdir.mkdir("file_uploads")


@pytest.fixture
def ui(file_upload_dir):
    mock_agent = Mock()
    return GradioUI(agent=mock_agent, file_upload_folder=file_upload_dir)


class TestGradioUI:
    def test_upload_file_allows_empty_file(self, ui, tmpdir):
        empty_file_path = tmpdir.join("empty.txt")
        empty_file_path.write("")

        with open(empty_file_path) as f:
            textbox, uploads_log = ui.upload_file(f, [])

            assert "File uploaded:" in textbox.value
            assert len(uploads_log) == 1
            assert os.path.exists(tmpdir / empty_file_path.basename)

    def test_upload_file_default_types_disallowed(self, ui, datadir):
        with open(datadir / "image.png") as file:
            textbox, uploads_log = ui.upload_file(file, [])

            assert textbox.value == "File type disallowed"
            assert len(uploads_log) == 0

    @pytest.mark.parametrize("sample_file_name", ["empty.pdf", "file.txt", "sample.docx"])
    def test_upload_file_success(self, ui, sample_file_name, datadir, file_upload_dir):
        with open(datadir / sample_file_name) as f:
            textbox, uploads_log = ui.upload_file(
                f,
                [],
                allowed_file_types=[
                    "text/plain",
                    "application/pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ],
            )

            assert "File uploaded:" in textbox.value
            assert len(uploads_log) == 1

            assert (file_upload_dir / sample_file_name).exists()
            assert uploads_log[0] == file_upload_dir / sample_file_name

    def test_upload_file_creates_new_file_if_filenames_clash(self, ui, file_upload_dir, tmpdir):
        empty_file_path = tmpdir.join("empty.txt")
        empty_file_path.write("")

        with open(empty_file_path) as f:
            _, uploads_log = ui.upload_file(f, [])

            assert len(uploads_log) == 1

            textbox, uploads_log = ui.upload_file(f, [])

            assert "File uploaded:" in textbox.value
            assert len(uploads_log) == 1

            # now there should be 2 files, even though file-names are duplicated
            assert len(file_upload_dir.listdir()) == 2
            assert {"empty.txt", "empty_1.txt"} == {f.basename for f in file_upload_dir.listdir() if f.isfile()}

    def test_upload_file_none(self, ui):
        """Test scenario when no file is selected"""
        textbox, uploads_log = ui.upload_file(None, [])

        assert textbox.value == "No file uploaded"
        assert len(uploads_log) == 0

    def test_upload_file_invalid_type(self, ui, datadir):
        """Test disallowed file type"""
        with open(datadir / "empty.pdf") as file:
            textbox, uploads_log = ui.upload_file(file, [], allowed_file_types=["text/plain"])

            assert textbox.value == "File type disallowed"
            assert len(uploads_log) == 0

    def test_upload_file_special_chars(self, ui, tmpdir):
        special_char_name = tmpdir.join("test@#$%^&*.txt")
        special_char_name.write("something")

        with open(special_char_name) as mock_file:
            textbox, uploads_log = ui.upload_file(mock_file, [])

            assert "File uploaded:" in textbox.value
            assert len(uploads_log) == 1
            assert "test_____" in uploads_log[0]
