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
import base64
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import textwrap
import types
from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from smolagents.memory import AgentLogger


__all__ = ["AgentError","UserInputError","StringBuffer","Display"]


@lru_cache
def _is_package_available(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


@lru_cache
def _is_pillow_available():
    return importlib.util.find_spec("PIL") is not None


BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "calendar",
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

class Display(ABC):
    @abstractmethod
    def display(self):
        pass

class StringBuffer(str):
    def __init__(self, value=""):
        if isinstance(value, list):
            self._buffer = []
            for v in value:
                if isinstance(v, StringBuffer):
                    self._buffer.extend(v._buffer)
                else:
                    self._buffer.append(v)
        elif value is not None:
            self._buffer = [value]
        else:
            self._buffer = []

    @property
    def buffer(self):
        return self._buffer

    def append(self, string):
        """追加字符串到缓冲区"""
        if not isinstance(string, str):
            raise ValueError("Only strings can be appended.")
        self._buffer.append(string)

    def clear(self):
        """清空缓冲区"""
        self._buffer.clear()

    def to_string(self):
        """返回拼接后的完整字符串"""
        return ''.join([str(v) for v in self._buffer])

    def __iadd__(self, other):
        """支持 += 操作"""
        if isinstance(other, str):
            self.append(other)
        elif isinstance(other, StringBuffer):
            self._buffer.extend(other._buffer)
        else:
            raise TypeError(f"Unsupported operand type(s) for +=: 'StringBuffer' and '{type(other).__name__}'")
        return self

    def __add__(self, other):
        """支持 + 操作"""
        new_buffer = StringBuffer()
        new_buffer._buffer.extend(self._buffer)
        if isinstance(other, str):
            new_buffer.append(other)
        elif isinstance(other, StringBuffer):
            new_buffer._buffer.extend(other._buffer)
        else:
            raise TypeError(f"Unsupported operand type(s) for +: 'StringBuffer' and '{type(other).__name__}'")
        return new_buffer

    def __radd__(self, other):
        """支持反向 + 操作"""
        if isinstance(other, str):
            new_buffer = StringBuffer(other)
            new_buffer._buffer.extend(self._buffer)
            return new_buffer
        else:
            raise TypeError(f"Unsupported operand type(s) for +: '{type(other).__name__}' and 'StringBuffer'")

    def format(self, *args, **kwargs):
        """支持 format 操作"""
        formatted_str = self.to_string().format(*args, **kwargs)
        return StringBuffer(formatted_str)

    def __len__(self):
        """返回当前缓冲区中字符串的总长度"""
        return len(self.to_string())

    def __getitem__(self, index):
        return self.to_string()[index]

    def __str__(self):
        """转换为字符串表示"""
        return self.to_string()

    def __repr__(self):
        """调试信息"""
        return f"StringBuffer({self._buffer})"

    # 实现 str 类的其他方法
    def capitalize(self):
        return self.to_string().capitalize()

    def casefold(self):
        return self.to_string().casefold()

    def center(self, width, fillchar=' '):
        return self.to_string().center(width, fillchar)

    def count(self, sub, start=0, end=None):
        return self.to_string().count(sub, start, end)

    def encode(self, encoding='utf-8', errors='strict'):
        return self.to_string().encode(encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        return self.to_string().endswith(suffix, start, end)

    def expandtabs(self, tabsize=8):
        return self.to_string().expandtabs(tabsize)

    def find(self, sub, start=0, end=None):
        return self.to_string().find(sub, start, end)

    def index(self, sub, start=0, end=None):
        return self.to_string().index(sub, start, end)

    def isalnum(self):
        return self.to_string().isalnum()

    def isalpha(self):
        return self.to_string().isalpha()

    def isascii(self):
        return self.to_string().isascii()

    def isdecimal(self):
        return self.to_string().isdecimal()

    def isdigit(self):
        return self.to_string().isdigit()

    def isidentifier(self):
        return self.to_string().isidentifier()

    def islower(self):
        return self.to_string().islower()

    def isnumeric(self):
        return self.to_string().isnumeric()

    def isprintable(self):
        return self.to_string().isprintable()

    def isspace(self):
        return self.to_string().isspace()

    def istitle(self):
        return self.to_string().istitle()

    def isupper(self):
        return self.to_string().isupper()

    def join(self, iterable):
        return self.to_string().join(iterable)

    def ljust(self, width, fillchar=' '):
        return self.to_string().ljust(width, fillchar)

    def lower(self):
        return self.to_string().lower()

    def lstrip(self, chars=None):
        return self.to_string().lstrip(chars)

    def partition(self, sep):
        return self.to_string().partition(sep)

    def removeprefix(self, prefix):
        return self.to_string().removeprefix(prefix)

    def removesuffix(self, suffix):
        return self.to_string().removesuffix(suffix)

    def replace(self, old, new, count=-1):
        return self.to_string().replace(old, new, count)

    def rfind(self, sub, start=0, end=None):
        return self.to_string().rfind(sub, start, end)

    def rindex(self, sub, start=0, end=None):
        return self.to_string().rindex(sub, start, end)

    def rjust(self, width, fillchar=' '):
        return self.to_string().rjust(width, fillchar)

    def rpartition(self, sep):
        return self.to_string().rpartition(sep)

    def rsplit(self, sep=None, maxsplit=-1):
        return self.to_string().rsplit(sep, maxsplit)

    def rstrip(self, chars=None):
        return self.to_string().rstrip(chars)

    def split(self, sep=None, maxsplit=-1):
        return self.to_string().split(sep, maxsplit)

    def splitlines(self, keepends=False):
        return self.to_string().splitlines(keepends)

    def startswith(self, prefix, start=0, end=None):
        return self.to_string().startswith(prefix, start, end)

    def strip(self, chars=None):
        return self.to_string().strip(chars)

    def swapcase(self):
        return self.to_string().swapcase()

    def title(self):
        return self.to_string().title()

    def translate(self, table):
        return self.to_string().translate(table)

    def upper(self):
        return self.to_string().upper()

    def zfill(self, width):
        return self.to_string().zfill(width)

class UserInputError(BaseException):
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return "input"
    def __str__(self):
        return "input"

def escape_code_brackets(text: str) -> str:
    """Escapes square brackets in code segments while preserving Rich styling tags."""

    def replace_bracketed_content(match):
        content = match.group(1)
        cleaned = re.sub(
            r"bold|red|green|blue|yellow|magenta|cyan|white|black|italic|dim|\s|#[0-9a-fA-F]{6}", "", content
        )
        return f"\\[{content}\\]" if cleaned.strip() else f"[{content}]"

    return re.sub(r"\[([^\]]*)\]", replace_bracketed_content, text)


class AgentError(Exception):
    """Base class for other agent-related exceptions"""

    def __init__(self, message, logger: "AgentLogger"):
        super().__init__(message)
        self.message = message
        logger.log_error(message)

    def dict(self) -> Dict[str, str]:
        return {"type": self.__class__.__name__, "message": str(self.message)}


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


def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # For any other type, convert to string
        return str(obj)


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace('\\"', "'")
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
            f"'{json_blob[place - 4 : place + 5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blobs(code_blob: str) -> str:
    """Parses the LLM's output to get any code blob inside. Will return the code directly if it's code."""
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
Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
Here is your code snippet:
{code_blob}
It seems like you're trying to return the final answer, you can do it as follows:
Code:
```py
final_answer("YOUR FINAL ANSWER HERE")
```<end_code>""".strip()
            )
        raise ValueError(
            f"""
Your code snippet is invalid, because the regex pattern {pattern} was not found in it.
Here is your code snippet:
{code_blob}
Make sure to include code with the correct pattern, for instance:
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


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
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
    return get_source(method)


def is_same_method(method1, method2):
    """Compare two methods by their source code."""
    try:
        source1 = get_method_source(method1)
        source2 = get_method_source(method2)

        # Remove method decorators if any
        source1 = "\n".join(line for line in source1.split("\n") if not line.strip().startswith("@"))
        source2 = "\n".join(line for line in source2.split("\n") if not line.strip().startswith("@"))

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
        and not (base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value)
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            # multiline value
            if "\n" in value:
                escaped_value = value.replace('"""', r"\"\"\"")  # Escape triple quotes
                class_lines.append(f'    {name} = """{escaped_value}"""')
            else:
                class_lines.append(f"    {name} = {json.dumps(value)}")
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
            base_cls and hasattr(base_cls, name) and getattr(base_cls, name).__code__.co_code == func.__code__.co_code
        )
    }

    for name, method in methods.items():
        method_source = get_source(method)
        # Clean up the indentation
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(["    " + line if line.strip() else line for line in method_lines])
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


def get_source(obj) -> str:
    """Get the source code of a class or callable object (e.g.: function, method).
    First attempts to get the source code using `inspect.getsource`.
    In a dynamic environment (e.g.: Jupyter, IPython), if this fails,
    falls back to retrieving the source code from the current interactive shell session.

    Args:
        obj: A class or callable object (e.g.: function, method)

    Returns:
        str: The source code of the object, dedented and stripped

    Raises:
        TypeError: If object is not a class or callable
        OSError: If source code cannot be retrieved from any source
        ValueError: If source cannot be found in IPython history

    Note:
        TODO: handle Python standard REPL
    """
    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")

    inspect_error = None
    try:
        return textwrap.dedent(inspect.getsource(obj)).strip()
    except OSError as e:
        # let's keep track of the exception to raise it if all further methods fail
        inspect_error = e
    try:
        import IPython

        shell = IPython.get_ipython()
        if not shell:
            raise ImportError("No active IPython shell found")
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return textwrap.dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history")
    except ImportError:
        # IPython is not available, let's just raise the original inspect error
        raise inspect_error
    except ValueError as e:
        # IPython is available but we couldn't find the source code, let's raise the error
        raise e from inspect_error


def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_image_url(base64_image):
    return f"data:image/png;base64,{base64_image}"


def make_init_file(folder: str):
    os.makedirs(folder, exist_ok=True)
    # Create __init__
    with open(os.path.join(folder, "__init__.py"), "w"):
        pass
