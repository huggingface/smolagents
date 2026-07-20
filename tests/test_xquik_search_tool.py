# Copyright 2026 The HuggingFace Team. All rights reserved.
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
from unittest.mock import Mock, patch

from examples.xquik_search_tool import XquikSearchPostsTool, format_post


def test_xquik_search_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("XQUIK_API_KEY", raising=False)

    assert XquikSearchPostsTool().forward("agents") == "Set XQUIK_API_KEY before calling this tool."


def test_xquik_search_bounds_results_and_formats_posts(monkeypatch):
    monkeypatch.setenv("XQUIK_API_KEY", "xq_test")
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "tweets": [
            {
                "author": {"username": "huggingface"},
                "likeCount": 4,
                "text": "Small agents are useful.",
                "url": "https://x.com/huggingface/status/1",
            }
        ]
    }

    with patch("examples.xquik_search_tool.requests.get", return_value=response) as get:
        result = XquikSearchPostsTool().forward("agents", 100)

    get.assert_called_once_with(
        "https://xquik.com/api/v1/x/tweets/search",
        headers={"x-api-key": "xq_test"},
        params={"q": "agents", "limit": 20},
        timeout=30,
    )
    response.raise_for_status.assert_called_once_with()
    assert result == "@huggingface (likes: 4)\nSmall agents are useful.\nURL: https://x.com/huggingface/status/1"


def test_format_post_supports_flat_author_data():
    assert format_post({"fullText": "Hello", "username": "alice"}) == "@alice\nHello"
