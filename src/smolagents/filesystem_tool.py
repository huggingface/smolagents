#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from __future__ import annotations

from pathlib import Path

from .tools import Tool


class PathNotAllowedError(Exception):
    """Exception raised when attempting to access a path outside the allowed paths."""

    pass


class FileSystemTool(Tool):
    name = "file_system"
    description = """A tool for interacting with the local file system. Supports reading text files and listing directory contents.
All operations are restricted to paths within the allowed paths specified during initialization.

Operations:
- read_file: Reads a text file and returns its content as a string.
- list_directory: Lists the contents of a directory, returning a list of entries with name and type (file/directory).

All errors are raised as exceptions:
- PathNotAllowedError: Access to path outside allowed paths
- FileNotFoundError: File or directory does not exist
- IsADirectoryError: Attempted to read a directory as a file
- NotADirectoryError: Attempted to list a file as a directory
- ValueError: Invalid operation name
"""
    inputs = {
        "operation": {
            "type": "string",
            "description": "The operation to perform. Must be either 'read_file' or 'list_directory'.",
        },
        "path": {
            "type": "string",
            "description": "The path to the file or directory.",
        },
        "encoding": {
            "type": "string",
            "description": "The encoding to use when reading files. Only applicable for 'read_file' operation. Defaults to 'utf-8'.",
            "nullable": True,
        },
    }
    output_type = "object"
    output_schema = {
        "oneOf": [
            {
                "type": "string",
                "description": "File content (returned by read_file operation)",
            },
            {
                "type": "array",
                "description": "Directory entries (returned by list_directory operation)",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the file or directory"},
                        "type": {"type": "string", "description": "Type of entry: 'file' or 'directory'"},
                    },
                    "required": ["name", "type"],
                },
            },
        ],
    }

    def __init__(self, allowed_paths: list[str | Path], *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not allowed_paths:
            raise ValueError("allowed_paths must not be empty. At least one allowed path is required.")
        self._allowed_paths = [Path(p).resolve() for p in allowed_paths]
        for allowed_path in self._allowed_paths:
            if not allowed_path.exists():
                raise ValueError(f"Allowed path does not exist: {allowed_path}")
        self.is_initialized = True

    def _is_path_allowed(self, target_path: Path) -> bool:
        target_absolute = target_path.resolve()
        for allowed_path in self._allowed_paths:
            try:
                target_absolute.relative_to(allowed_path)
                return True
            except ValueError:
                continue
        return False

    def _validate_path(self, path: str | Path) -> Path:
        target_path = Path(path)
        if not self._is_path_allowed(target_path):
            allowed_paths_str = ", ".join(str(p) for p in self._allowed_paths)
            raise PathNotAllowedError(
                f"Access to path '{path}' is not allowed. "
                f"Only paths within the following directories are accessible: {allowed_paths_str}"
            )
        return target_path.resolve()

    def forward(
        self,
        operation: str,
        path: str,
        encoding: str | None = None,
    ) -> str | list[dict]:
        if operation == "read_file":
            return self._read_file(path, encoding or "utf-8")
        elif operation == "list_directory":
            return self._list_directory(path)
        else:
            raise ValueError(
                f"Invalid operation: '{operation}'. Must be one of: 'read_file', 'list_directory'"
            )

    def _read_file(self, path: str, encoding: str) -> str:
        target_path = self._validate_path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if target_path.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        return target_path.read_text(encoding=encoding)

    def _list_directory(self, path: str) -> list[dict]:
        target_path = self._validate_path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not target_path.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {path}")
        entries = []
        for entry in sorted(target_path.iterdir()):
            entry_type = "directory" if entry.is_dir() else "file"
            entries.append({"name": entry.name, "type": entry_type})
        return entries
