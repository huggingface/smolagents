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

from smolagents.models import get_clean_message_list


def _text_content(*texts):
    return [{"type": "text", "text": text} for text in texts]


def test_flatten_single_message_joins_all_text_blocks():
    messages = [{"role": "user", "content": _text_content("a", "b")}]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert result == [{"role": "user", "content": "a\nb"}]


def test_flatten_merges_same_role_message_with_all_blocks():
    messages = [
        {"role": "user", "content": _text_content("a")},
        {"role": "user", "content": _text_content("b", "c")},
    ]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert result == [{"role": "user", "content": "a\nb\nc"}]


def test_flatten_empty_content_returns_empty_string():
    messages = [{"role": "user", "content": []}]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert result == [{"role": "user", "content": ""}]


def test_flatten_empty_follow_up_message_does_not_crash():
    messages = [
        {"role": "user", "content": _text_content("a")},
        {"role": "user", "content": []},
    ]
    result = get_clean_message_list(messages, flatten_messages_as_text=True)
    assert result == [{"role": "user", "content": "a"}]


def test_flatten_rejects_non_text_blocks():
    messages = [{"role": "user", "content": [{"type": "audio", "data": "x"}]}]
    with pytest.raises(ValueError):
        get_clean_message_list(messages, flatten_messages_as_text=True)
