#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""This module contains utilities exclusively taken from `transformers` repository.

Since they are not specific to `transformers` and that `transformers` is an heavy dependencies, those helpers have
been duplicated.

TODO: move them to `huggingface_hub` to avoid code duplication.
"""

import inspect
import json
import re
import types
from copy import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from huggingface_hub.utils import is_torch_available


def get_imports(code: str) -> List[str]:
    """
    Extracts all the libraries (not relative imports) that are imported in a code.

    Args:
        code (`str`): Code text to inspect.

    Returns:
        `list[str]`: List of all packages required to use the input code.
    """
    # filter out try/except block so in custom code we can have try/except imports
    code = re.sub(r"\s*try\s*:.*?except.*?:", "", code, flags=re.DOTALL)

    # filter out imports under is_flash_attn_2_available block for avoid import issues in cpu only environment
    code = re.sub(
        r"if is_flash_attn[a-zA-Z0-9_]+available\(\):\s*(from flash_attn\s*.*\s*)+",
        "",
        code,
        flags=re.MULTILINE,
    )

    # Imports of the form `import xxx` or `import xxx as yyy`
    imports = re.findall(r"^\s*import\s+(\S+?)(?:\s+as\s+\S+)?\s*$", code, flags=re.MULTILINE)
    # Imports of the form `from xxx import yyy`
    imports += re.findall(r"^\s*from\s+(\S+)\s+import", code, flags=re.MULTILINE)
    # Only keep the top-level module
    imports = [imp.split(".")[0] for imp in imports if not imp.startswith(".")]
    return list(set(imports))


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas"""


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas"""


def get_json_schema(func: Callable) -> Dict:
    """
    This function generates a JSON schema for a given function, based on its docstring and type hints. This is
    mostly used for passing lists of tools to a chat template. The JSON schema contains the name and description of
    the function, as well as the names, types and descriptions for each of its arguments. `get_json_schema()` requires
    that the function has a docstring, and that each argument has a description in the docstring, in the standard
    Google docstring format shown below. It also requires that all the function arguments have a valid Python type hint.

    Although it is not required, a `Returns` block can also be added, which will be included in the schema. This is
    optional because most chat templates ignore the return value of the function.

    Args:
        func: The function to generate a JSON schema for.

    Returns:
        A dictionary containing the JSON schema for the function.

    Examples:
    ```python
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    '''
    >>>    return x * y
    >>>
    >>> print(get_json_schema(multiply))
    {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            },
            "required": ["x", "y"]
        }
    }
    ```

    The general use for these schemas is that they are used to generate tool descriptions for chat templates that
    support them, like so:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.utils import get_json_schema
    >>>
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    return x * y
    >>>    '''
    >>>
    >>> multiply_schema = get_json_schema(multiply)
    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    >>> messages = [{"role": "user", "content": "What is 179 x 4571?"}]
    >>> formatted_chat = tokenizer.apply_chat_template(
    >>>     messages,
    >>>     tools=[multiply_schema],
    >>>     chat_template="tool_use",
    >>>     return_dict=True,
    >>>     return_tensors="pt",
    >>>     add_generation_prompt=True
    >>> )
    >>> # The formatted chat can now be passed to model.generate()
    ```

    Each argument description can also have an optional `(choices: ...)` block at the end, such as
    `(choices: ["tea", "coffee"])`, which will be parsed into an `enum` field in the schema. Note that this will
    only be parsed correctly if it is at the end of the line:

    ```python
    >>> def drink_beverage(beverage: str):
    >>>    '''
    >>>    A function that drinks a beverage
    >>>
    >>>    Args:
    >>>        beverage: The beverage to drink (choices: ["tea", "coffee"])
    >>>    '''
    >>>    pass
    >>>
    >>> print(get_json_schema(drink_beverage))
    ```
    {
        'name': 'drink_beverage',
        'description': 'A function that drinks a beverage',
        'parameters': {
            'type': 'object',
            'properties': {
                'beverage': {
                    'type': 'string',
                    'enum': ['tea', 'coffee'],
                    'description': 'The beverage to drink'
                    }
                },
            'required': ['beverage']
        }
    }
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"Cannot generate JSON schema for {func.__name__} because it has no docstring!"
        )
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = _parse_google_format_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:  # We allow a missing return docstring since most templates ignore it
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'"
            )

        desc = param_descriptions[arg]

        lines = [line.rstrip() for line in desc.strip().splitlines()]
        for i in reversed(range(len(lines))):
            match = re.search(r"\(choices:\s*(.*?)\)", lines[i], flags=re.IGNORECASE)
            if match:
                try:
                    schema["enum"] = json.loads(match.group(1))
                except json.JSONDecodeError as e:
                    raise DocstringParsingException(
                        f"Invalid JSON in choices enum for argument '{arg}': {match.group(1)}"
                    ) from e
                lines[i] = lines[i][: match.start()].rstrip()
                break
        desc = "\n".join(lines).strip()

        schema["description"] = desc

    output = {"name": func.__name__, "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


# Extracts the initial segment of the docstring, containing the function description
description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"(?:^|\n)"  # Match the start of the args block, or a newline
    r"\s*(\w+)\s*(?:\([^)]*?\))?:\s*"  # Capture the argument name (ignore the type) and strip spacing
    r"(.*?)\s*"  # Capture the argument description, which can span multiple lines, and strip trailing spacing
    r"(?=\n\s*\w+\s*(?:\([^)]*?\))?:|\Z)",  # Stop when you hit the next argument (with or without type) or the end of the block
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(
    r"\n\s*Returns:\n\s*"
    r"(?:[^)]*?:\s*)?"  # Ignore the return type if present
    r"(.*?)"  # Capture the return description
    r"[\n\s]*(Raises:|\Z)",
    re.DOTALL,
)


def _parse_google_format_docstring(doc: str) -> tuple[str, dict[str, str], str | None]:
    lines = doc.strip().splitlines()
    headers = {"Args:", "Returns:", "Raises:"}
    sections = {}
    current_lines = []
    description_lines = []
    current_section = None

    for line in lines:
        if (header := line.strip()) in headers:
            if current_section:
                sections[current_section] = current_lines
            else:
                description_lines = current_lines
            current_section = header[:-1].lower()  # strip colon
            current_lines = []
        else:
            current_lines.append(line)

    # Capture the final section or all as description if no headers
    if current_section:
        sections[current_section] = current_lines
    elif not description_lines:
        description_lines = lines

    args = {}
    if "args" in sections:
        for match in args_split_re.finditer("\n".join(sections["args"])):
            args[match.group(1)] = match.group(2).strip()

    return_doc = None
    if "returns" in sections:
        lines = sections["returns"]
        if lines:
            first = lines[0].strip()
            if match := re.match(r"^(\w[\w.\[\], ]*):\s*(.*)", first):
                lines[0] = match.group(2).strip()
            return_doc = "\n".join(lines).strip()

    return "\n".join(description_lines).strip(), args, return_doc


def _convert_type_hints_to_json_schema(func: Callable, error_on_missing_type_hints: bool = True) -> Dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)

    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty and error_on_missing_type_hints:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param_name not in properties:
            properties[param_name] = {}

        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["nullable"] = True

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def _parse_type_hint(hint: str) -> Dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException(
                "Couldn't parse this type hint, likely due to a custom class or object: ",
                hint,
            )

    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        # Recurse into each of the subtypes in the Union, except None, which is handled separately at the end
        subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
        if len(subtypes) == 1:
            # A single non-null type can be expressed directly
            return_dict = subtypes[0]
        elif all(isinstance(subtype["type"], str) for subtype in subtypes):
            # A union of basic types can be expressed as a list in the schema
            return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
        else:
            # A union of more complex types requires "anyOf"
            return_dict = {"anyOf": subtypes}
        if type(None) in args:
            return_dict["nullable"] = True
        return return_dict

    elif origin is list:
        if not args:
            return {"type": "array"}
        else:
            # Lists can only have a single type argument, so recurse into it
            return {"type": "array", "items": _parse_type_hint(args[0])}

    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace('typing.', '')} is a Tuple with a single element, which "
                "we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain "
                "more than one element, we recommend "
                "using a List[] type instead, or if it really is a single element, remove the Tuple[] wrapper and just "
                "pass the element directly."
            )
        if ... in args:
            raise TypeHintParsingException(
                "Conversion of '...' is not supported in Tuple type hints. "
                "Use List[] types for variable-length"
                " inputs instead."
            )
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}

    elif origin is dict:
        # The JSON equivalent to a dict is 'object', which mandates that all keys are strings
        # However, we can specify the type of the dict values with "additionalProperties"
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out

    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object: ", hint)


_BASE_TYPE_MAPPING = {
    int: {"type": "integer"},
    float: {"type": "number"},
    str: {"type": "string"},
    bool: {"type": "boolean"},
    Any: {"type": "any"},
    types.NoneType: {"type": "null"},
}


def _get_json_schema_type(param_type: str) -> Dict[str, str]:
    if param_type in _BASE_TYPE_MAPPING:
        return copy(_BASE_TYPE_MAPPING[param_type])
    if str(param_type) == "Image":
        from PIL.Image import Image

        if param_type == Image:
            return {"type": "image"}
    if str(param_type) == "Tensor" and is_torch_available():
        from torch import Tensor

        if param_type == Tensor:
            return {"type": "audio"}
    return {"type": "object"}
