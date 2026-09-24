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
from pathlib import Path

import pytest

from smolagents.default_tools import FileSystemTool, PathNotAllowedError


class TestFileSystemTool:
    def test_read_file_success(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!\n这是中文测试。"
        test_file.write_text(test_content, encoding="utf-8")

        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        result = tool("read_file", str(test_file))

        assert result == test_content

    def test_read_file_with_encoding(self, tmp_path):
        test_file = tmp_path / "test_gbk.txt"
        test_content = "中文测试内容"
        test_file.write_text(test_content, encoding="gbk")

        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        result = tool("read_file", str(test_file), encoding="gbk")

        assert result == test_content

    def test_list_directory_success(self, tmp_path):
        file1 = tmp_path / "file1.txt"
        file1.write_text("content1")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        file2 = subdir / "file2.txt"
        file2.write_text("content2")

        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        result = tool("list_directory", str(tmp_path))

        assert isinstance(result, list)
        assert len(result) == 2

        names = [e["name"] for e in result]
        assert "file1.txt" in names
        assert "subdir" in names

        for entry in result:
            if entry["name"] == "file1.txt":
                assert entry["type"] == "file"
            elif entry["name"] == "subdir":
                assert entry["type"] == "directory"

    def test_access_outside_allowed_paths_raises_error(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        forbidden_dir = tmp_path / "forbidden"
        forbidden_dir.mkdir()

        forbidden_file = forbidden_dir / "secret.txt"
        forbidden_file.write_text("secret content")

        tool = FileSystemTool(allowed_paths=[str(allowed_dir)])

        with pytest.raises(PathNotAllowedError) as exc_info:
            tool("read_file", str(forbidden_file))

        assert "is not allowed" in str(exc_info.value)

        with pytest.raises(PathNotAllowedError) as exc_info:
            tool("list_directory", str(forbidden_dir))

        assert "is not allowed" in str(exc_info.value)

    def test_path_traversal_attack_blocked(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        forbidden_file = tmp_path / "secret.txt"
        forbidden_file.write_text("secret content")

        tool = FileSystemTool(allowed_paths=[str(allowed_dir)])

        traversal_path = os.path.join(str(allowed_dir), "..", "secret.txt")

        with pytest.raises(PathNotAllowedError):
            tool("read_file", traversal_path)

    def test_multiple_allowed_paths(self, tmp_path):
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        file1 = dir1 / "file1.txt"
        file1.write_text("content1")

        dir2 = tmp_path / "dir2"
        dir2.mkdir()
        file2 = dir2 / "file2.txt"
        file2.write_text("content2")

        tool = FileSystemTool(allowed_paths=[str(dir1), str(dir2)])

        result1 = tool("read_file", str(file1))
        assert result1 == "content1"

        result2 = tool("read_file", str(file2))
        assert result2 == "content2"

    def test_invalid_operation_raises_value_error(self, tmp_path):
        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        with pytest.raises(ValueError, match="Invalid operation"):
            tool("invalid_operation", str(tmp_path))

    def test_read_non_existent_file_raises_file_not_found_error(self, tmp_path):
        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        non_existent_path = str(tmp_path / "non_existent.txt")
        with pytest.raises(FileNotFoundError, match="File not found"):
            tool("read_file", non_existent_path)

    def test_list_non_existent_directory_raises_file_not_found_error(self, tmp_path):
        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        non_existent_path = str(tmp_path / "non_existent")
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            tool("list_directory", non_existent_path)

    def test_read_directory_raises_is_a_directory_error(self, tmp_path):
        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        with pytest.raises(IsADirectoryError, match="Path is a directory"):
            tool("read_file", str(tmp_path))

    def test_list_file_raises_not_a_directory_error(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        tool = FileSystemTool(allowed_paths=[str(tmp_path)])

        with pytest.raises(NotADirectoryError, match="Path is not a directory"):
            tool("list_directory", str(test_file))

    def test_empty_allowed_paths_raises_error(self):
        with pytest.raises(ValueError, match="allowed_paths must not be empty"):
            FileSystemTool(allowed_paths=[])

    def test_non_existent_allowed_path_raises_error(self, tmp_path):
        non_existent_path = tmp_path / "non_existent"

        with pytest.raises(ValueError, match="Allowed path does not exist"):
            FileSystemTool(allowed_paths=[str(non_existent_path)])

    def test_subdirectory_access_allowed(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        subdir = allowed_dir / "subdir"
        subdir.mkdir()

        test_file = subdir / "test.txt"
        test_file.write_text("content")

        tool = FileSystemTool(allowed_paths=[str(allowed_dir)])

        result = tool("read_file", str(test_file))
        assert result == "content"

        result_list = tool("list_directory", str(subdir))
        assert isinstance(result_list, list)

    def test_import_from_smolagents(self):
        from smolagents import FileSystemTool as FST
        from smolagents import PathNotAllowedError as PNAE

        assert FST is FileSystemTool
        assert PNAE is PathNotAllowedError
