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
import ast
import inspect
import json
import re
import types
from typing import Dict, Tuple, Union

from rich.console import Console
from transformers.utils.import_utils import _is_package_available

_pygments_available = _is_package_available("pygments")


def is_pygments_available():
    return _pygments_available


console = Console(width=200)

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


class AgentError(Exception):
    """Base class for other agent-related exceptions"""

    def __init__(self, message):
        super().__init__(message)
        self.message = message
        console.print(f"[bold red]{message}[/bold red]")


class AgentParsingError(AgentError):
    """Exception raised for errors in parsing in the agent"""

    pass


class AgentExecutionError(AgentError):
    """Exception raised for errors in execution in the agent"""

    pass


class AgentMaxStepsError(AgentError):
    """Exception raised for errors in execution in the agent"""

    pass


class AgentGenerationError(AgentError):
    """Exception raised for errors in generation in the agent"""

    pass


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace(
            '\\"', "'"
        )
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place-4:place+5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blobs(code_blob: str) -> str:
    """Parses the LLM's output to get any code blob inside. Will retrun the code directly if it's code."""
    pattern = r"```(?:py|python)?\n(.*?)\n```"
    matches = re.findall(pattern, code_blob, re.DOTALL)
    if len(matches) == 0:
        try:  # Maybe the LLM outputted a code blob directly
            ast.parse(code_blob)
            return code_blob
        except SyntaxError:
            pass

        if "final" in code_blob and "answer" in code_blob:
            raise ValueError(
                f"""
The code blob is invalid, because the regex pattern {pattern} was not found in {code_blob=}. It seems like you're trying to return the final answer, you can do it as follows:
Code:
```py
final_answer("YOUR FINAL ANSWER HERE")
```<end_code>""".strip()
            )
        raise ValueError(
            f"""
The code blob is invalid, because the regex pattern {pattern} was not found in {code_blob=}. Make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_code>""".strip()
        )
    return "\n\n".join(match.strip() for match in matches)


def parse_json_tool_call(json_blob: str) -> Tuple[str, Union[str, None]]:
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    tool_name_key, tool_arguments_key = None, None
    for possible_tool_name_key in ["action", "tool_name", "tool", "name", "function"]:
        if possible_tool_name_key in tool_call:
            tool_name_key = possible_tool_name_key
    for possible_tool_arguments_key in [
        "action_input",
        "tool_arguments",
        "tool_args",
        "parameters",
    ]:
        if possible_tool_arguments_key in tool_call:
            tool_arguments_key = possible_tool_arguments_key
    if tool_name_key is not None:
        if tool_arguments_key is not None:
            return tool_call[tool_name_key], tool_call[tool_arguments_key]
        else:
            return tool_call[tool_name_key], None
    error_msg = "No tool name key found in tool call!" + f" Tool call: {json_blob}"
    raise AgentParsingError(error_msg)


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(
    content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT
) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: MAX_LENGTH_TRUNCATE_CONTENT // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-MAX_LENGTH_TRUNCATE_CONTENT // 2 :]
        )


class ImportFinder(ast.NodeVisitor):
    def __init__(self):
        self.packages = set()

    def visit_Import(self, node):
        for alias in node.names:
            # Get the base package name (before any dots)
            base_package = alias.name.split(".")[0]
            self.packages.add(base_package)

    def visit_ImportFrom(self, node):
        if node.module:  # for "from x import y" statements
            # Get the base package name (before any dots)
            base_package = node.module.split(".")[0]
            self.packages.add(base_package)


def get_method_source(method):
    """Get source code for a method, including bound methods."""
    if isinstance(method, types.MethodType):
        method = method.__func__
    return inspect.getsource(method).strip()


def is_same_method(method1, method2):
    """Compare two methods by their source code."""
    try:
        source1 = get_method_source(method1)
        source2 = get_method_source(method2)

        # Remove method decorators if any
        source1 = "\n".join(
            line for line in source1.split("\n") if not line.strip().startswith("@")
        )
        source2 = "\n".join(
            line for line in source2.split("\n") if not line.strip().startswith("@")
        )

        return source1 == source2
    except (TypeError, OSError):
        return False


def is_same_item(item1, item2):
    """Compare two class items (methods or attributes) for equality."""
    if callable(item1) and callable(item2):
        return is_same_method(item1, item2)
    else:
        return item1 == item2


def instance_to_source(instance, base_cls=None):
    """Convert an instance to its class source code representation."""
    cls = instance.__class__
    class_name = cls.__name__

    # Start building class lines
    class_lines = []
    if base_cls:
        class_lines.append(f"class {class_name}({base_cls.__name__}):")
    else:
        class_lines.append(f"class {class_name}:")

    # Add docstring if it exists and differs from base
    if cls.__doc__ and (not base_cls or cls.__doc__ != base_cls.__doc__):
        class_lines.append(f'    """{cls.__doc__}"""')

    # Add class-level attributes
    class_attrs = {
        name: value
        for name, value in cls.__dict__.items()
        if not name.startswith("__")
        and not callable(value)
        and not (
            base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value
        )
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            if "\n" in value:
                class_lines.append(f'    {name} = """{value}"""')
            else:
                class_lines.append(f'    {name} = "{value}"')
        else:
            class_lines.append(f"    {name} = {repr(value)}")

    if class_attrs:
        class_lines.append("")

    # Add methods
    methods = {
        name: func
        for name, func in cls.__dict__.items()
        if callable(func)
        and not (
            base_cls
            and hasattr(base_cls, name)
            and getattr(base_cls, name).__code__.co_code == func.__code__.co_code
        )
    }

    for name, method in methods.items():
        method_source = inspect.getsource(method)
        # Clean up the indentation
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(
            ["    " + line if line.strip() else line for line in method_lines]
        )
        class_lines.append(method_source)
        class_lines.append("")

    # Find required imports using ImportFinder
    import_finder = ImportFinder()
    import_finder.visit(ast.parse("\n".join(class_lines)))
    required_imports = import_finder.packages

    # Build final code with imports
    final_lines = []

    # Add base class import if needed
    if base_cls:
        final_lines.append(f"from {base_cls.__module__} import {base_cls.__name__}")

    # Add discovered imports
    for package in required_imports:
        final_lines.append(f"import {package}")

    if final_lines:  # Add empty line after imports
        final_lines.append("")

    # Add the class code
    final_lines.extend(class_lines)

    return "\n".join(final_lines)


__all__ = ["AgentError"]
