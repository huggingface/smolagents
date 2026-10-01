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
"""Tests that validation paths raise typed exceptions instead of bare AssertionError.

Bare `assert` statements are silently stripped when Python runs with -O, making
validation invisible in optimised deployments. Each test here verifies the
specific exception type that should be raised so callers can distinguish e.g. a
bad user config (ValueError) from a type mismatch (TypeError).
"""

import pytest

from smolagents.models import ChatMessage, MessageRole, get_clean_message_list


# ---------------------------------------------------------------------------
# models.py – get_clean_message_list
# ---------------------------------------------------------------------------


def _make_message(role: str = "user", content=None) -> ChatMessage:
    return ChatMessage(role=MessageRole(role), content=content if content is not None else [{"type": "text", "text": "hi"}])


class TestGetCleanMessageListExceptions:
    def test_non_dict_element_raises_type_error(self):
        """Content block that is not a dict should raise TypeError, not AssertionError."""
        msg = _make_message(content=["not_a_dict"])
        with pytest.raises(TypeError, match="should be a dict"):
            get_clean_message_list([msg], role_conversions={})

    def test_image_with_text_only_raises_value_error(self):
        """Image element with flatten_messages_as_text=True should raise ValueError."""
        msg = _make_message(content=[{"type": "image", "image": b"fake"}])
        with pytest.raises(ValueError, match="Cannot use images with"):
            get_clean_message_list([msg], role_conversions={}, flatten_messages_as_text=True)

    def test_consecutive_role_non_list_content_raises_type_error(self):
        """When merging consecutive same-role messages, non-list content raises TypeError."""
        msg1 = _make_message(content=[{"type": "text", "text": "a"}])
        msg2 = ChatMessage(role=MessageRole.USER, content="plain string")
        with pytest.raises(TypeError, match="wrong content"):
            get_clean_message_list([msg1, msg2], role_conversions={})


# ---------------------------------------------------------------------------
# agents.py – prompt_templates validation
# ---------------------------------------------------------------------------


class TestAgentPromptTemplateExceptions:
    def test_missing_prompt_template_key_raises_value_error(self):
        """Incomplete prompt_templates dict should raise ValueError, not AssertionError."""
        from smolagents import EMPTY_PROMPT_TEMPLATES
        from smolagents.agents import CodeAgent

        class _FakeModel:
            def __call__(self, *a, **kw):
                pass

        # Provide a templates dict with one key removed
        incomplete = {k: v for i, (k, v) in enumerate(EMPTY_PROMPT_TEMPLATES.items()) if i != 0}
        with pytest.raises(ValueError, match="missing"):
            CodeAgent(tools=[], model=_FakeModel(), prompt_templates=incomplete)

    def test_managed_agent_without_name_raises_value_error(self):
        """Managed agent missing name/description should raise ValueError."""
        from unittest.mock import MagicMock

        from smolagents.agents import CodeAgent

        class _FakeModel:
            def __call__(self, *a, **kw):
                pass

        nameless = MagicMock()
        nameless.name = None
        nameless.description = "desc"
        with pytest.raises(ValueError, match="name and a description"):
            CodeAgent(tools=[], model=_FakeModel(), managed_agents=[nameless])

    def test_non_basetool_in_tools_raises_type_error(self):
        """Non-BaseTool element in tools list should raise TypeError."""
        from smolagents.agents import CodeAgent

        class _FakeModel:
            def __call__(self, *a, **kw):
                pass

        with pytest.raises(TypeError, match="instance of BaseTool"):
            CodeAgent(tools=["not_a_tool"], model=_FakeModel())


# ---------------------------------------------------------------------------
# tools.py – BaseTool.__init_subclass__ / validate
# ---------------------------------------------------------------------------


class TestToolValidationExceptions:
    def test_input_not_dict_raises_type_error(self):
        """Input spec that is not a dict should raise TypeError."""
        from smolagents.tools import Tool

        class BadTool(Tool):
            name = "bad_tool"
            description = "test"
            inputs = {"x": "not_a_dict"}
            output_type = "string"

            def forward(self, x: str) -> str:
                return x

        with pytest.raises(TypeError, match="should be a dictionary"):
            BadTool()

    def test_input_missing_required_keys_raises_value_error(self):
        """Input spec missing 'type' or 'description' should raise ValueError."""
        from smolagents.tools import Tool

        class BadTool(Tool):
            name = "bad_tool"
            description = "test"
            inputs = {"x": {"type": "string"}}  # missing 'description'
            output_type = "string"

            def forward(self, x: str) -> str:
                return x

        with pytest.raises(ValueError, match="'type' and 'description'"):
            BadTool()

    def test_invalid_output_type_raises_value_error(self):
        """output_type not in AUTHORIZED_TYPES should raise ValueError."""
        from smolagents.tools import Tool

        class BadTool(Tool):
            name = "bad_tool"
            description = "test"
            inputs = {"x": {"type": "string", "description": "an input"}}
            output_type = "invalid_type"

            def forward(self, x: str) -> str:
                return x

        with pytest.raises(ValueError, match="output_type"):
            BadTool()
