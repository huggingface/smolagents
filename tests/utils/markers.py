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
"""Markers for tests ."""

import os
from importlib.util import find_spec

import pytest


require_run_all = pytest.mark.skipif(not os.getenv("RUN_ALL"), reason="requires RUN_ALL environment variable")
require_soundfile = pytest.mark.skipif(find_spec("soundfile") is None, reason="requires soundfile")
require_torch = pytest.mark.skipif(find_spec("torch") is None, reason="requires torch")
