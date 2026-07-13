# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import example modules without installing them as a package.
MEMANTO_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "memanto"
sys.path.insert(0, str(MEMANTO_EXAMPLE_DIR))

from memanto_client import MemantoClient  # noqa: E402
from memanto_tools import (  # noqa: E402
    MemantoAnswerTool,
    MemantoRecallTool,
    MemantoRememberTool,
    create_memanto_tools,
)


@pytest.fixture
def client():
    return MemantoClient(base_url="http://memanto.test", agent_id="test-agent")


def test_recall_tool_formats_memories(client):
    client.recall = MagicMock(
        return_value=[
            {"type": "preference", "title": "Style", "content": "User prefers JSON responses."},
        ]
    )
    tool = MemantoRecallTool(client)
    result = tool.forward("What does the user prefer?")

    assert "preference" in result
    assert "JSON responses" in result
    client.recall.assert_called_once_with("What does the user prefer?")


def test_recall_tool_handles_empty_results(client):
    client.recall = MagicMock(return_value=[])
    tool = MemantoRecallTool(client)

    assert tool.forward("unknown topic") == "No relevant memories found."


def test_remember_tool_stores_memory(client):
    client.remember = MagicMock(return_value="mem_123")
    tool = MemantoRememberTool(client)
    result = tool.forward("User prefers dark mode.", memory_type="preference")

    assert "mem_123" in result
    client.remember.assert_called_once_with("User prefers dark mode.", memory_type="preference")


def test_create_memanto_tools_defaults(client):
    tools = create_memanto_tools(client)
    assert [tool.name for tool in tools] == ["recall_memory", "remember"]


def test_create_memanto_tools_with_answer(client):
    tools = create_memanto_tools(client, include_answer=True)
    assert [tool.name for tool in tools] == ["recall_memory", "remember", "answer_from_memory"]


def test_answer_tool_returns_grounded_response(client):
    client.answer = MagicMock(return_value="The user prefers JSON.")
    tool = MemantoAnswerTool(client)
    result = tool.forward("What format does the user prefer?")

    assert result == "The user prefers JSON."
    client.answer.assert_called_once_with("What format does the user prefer?")


def test_answer_tool_handles_empty_response(client):
    client.answer = MagicMock(return_value="")
    tool = MemantoAnswerTool(client)

    assert tool.forward("Unknown?") == "No answer could be generated from memory."


def test_client_has_memories(client):
    client.recall_recent = MagicMock(return_value=[{"content": "existing"}])
    assert client.has_memories() is True

    client.recall_recent = MagicMock(return_value=[])
    assert client.has_memories() is False


def test_client_activate_sets_session_token(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"session_token": "tok_abc"}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client._client, "post", return_value=mock_response) as mock_post:
        token = client.activate()

    assert token == "tok_abc"
    assert client.session_token == "tok_abc"
    mock_post.assert_called_once_with("/api/v2/agents/test-agent/activate")
