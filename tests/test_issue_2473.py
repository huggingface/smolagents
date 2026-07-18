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

import pytest

from smolagents.local_python_executor import InterpreterError, evaluate_python_code


def test_issue_2473():
    dangerous_integer_operations = [
        ("pow_result = 2 ** 10_000_001", {}),
        ("shift_result = 1 << 10_000_001", {}),
        ("mult_result = left * right", {"left": 1 << 5_000_000, "right": 1 << 5_000_000}),
    ]

    for code, state in dangerous_integer_operations:
        with pytest.raises(InterpreterError):
            evaluate_python_code(code, state=state, timeout_seconds=None)
