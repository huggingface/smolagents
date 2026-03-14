"""Tests for interleaved thinking / reasoning_content support (issue #1869).

Validates that reasoning_content from thinking models (DeepSeek-R1, Kimi,
Minimax, Ollama) is correctly preserved across the agent loop.
"""

import json
from unittest.mock import MagicMock

from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, _extract_reasoning_content
from smolagents.monitoring import Timing


class TestChatMessageReasoning:
    """ChatMessage correctly handles the reasoning_content field."""

    def test_reasoning_content_stored(self):
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="The answer is 42",
            reasoning_content="Let me think about this carefully...",
        )
        assert msg.reasoning_content == "Let me think about this carefully..."
        assert msg.content == "The answer is 42"

    def test_reasoning_content_default_none(self):
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Hello")
        assert msg.reasoning_content is None

    def test_reasoning_content_in_json_serialization(self):
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Result",
            reasoning_content="My reasoning...",
        )
        data = json.loads(msg.model_dump_json())
        assert data["reasoning_content"] == "My reasoning..."

    def test_reasoning_content_json_without_reasoning(self):
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Result")
        data = json.loads(msg.model_dump_json())
        assert data.get("reasoning_content") is None

    def test_from_dict_with_reasoning(self):
        data = {
            "role": "assistant",
            "content": "Result",
            "reasoning_content": "Step 1: analyze...",
        }
        msg = ChatMessage.from_dict(data)
        assert msg.reasoning_content == "Step 1: analyze..."

    def test_from_dict_without_reasoning(self):
        data = {"role": "assistant", "content": "Result"}
        msg = ChatMessage.from_dict(data)
        assert msg.reasoning_content is None

    def test_render_as_markdown_with_reasoning(self):
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Final answer",
            reasoning_content="My chain of thought",
        )
        rendered = msg.render_as_markdown()
        assert "My chain of thought" in rendered
        assert "Final answer" in rendered
        # Reasoning should appear before content
        assert rendered.index("My chain of thought") < rendered.index("Final answer")

    def test_render_as_markdown_without_reasoning(self):
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="Just an answer")
        rendered = msg.render_as_markdown()
        assert rendered == "Just an answer"
        assert "<think>" not in rendered


class TestExtractReasoningContent:
    """The helper correctly extracts reasoning from different provider formats."""

    def test_deepseek_kimi_format(self):
        mock_msg = MagicMock()
        mock_msg.reasoning_content = "DeepSeek thinking..."
        mock_msg.reasoning = None
        mock_msg.reasoning_details = None
        assert _extract_reasoning_content(mock_msg) == "DeepSeek thinking..."

    def test_ollama_format(self):
        mock_msg = MagicMock()
        mock_msg.reasoning_content = None
        mock_msg.reasoning = "Ollama reasoning..."
        mock_msg.reasoning_details = None
        assert _extract_reasoning_content(mock_msg) == "Ollama reasoning..."

    def test_minimax_format(self):
        mock_msg = MagicMock()
        mock_msg.reasoning_content = None
        mock_msg.reasoning = None
        mock_msg.reasoning_details = "Minimax analysis..."
        assert _extract_reasoning_content(mock_msg) == "Minimax analysis..."

    def test_no_reasoning_returns_none(self):
        mock_msg = MagicMock()
        mock_msg.reasoning_content = None
        mock_msg.reasoning = None
        mock_msg.reasoning_details = None
        mock_msg.model_extra = {}
        assert _extract_reasoning_content(mock_msg) is None

    def test_model_extra_fallback(self):
        """SDK that doesn't expose the field as attribute but has it in model_extra."""
        mock_msg = MagicMock(spec=[])
        mock_msg.model_extra = {"reasoning_content": "Hidden in extras"}
        assert _extract_reasoning_content(mock_msg) == "Hidden in extras"

    def test_priority_order(self):
        """reasoning_content takes priority over reasoning and reasoning_details."""
        mock_msg = MagicMock()
        mock_msg.reasoning_content = "Primary"
        mock_msg.reasoning = "Secondary"
        mock_msg.reasoning_details = "Tertiary"
        assert _extract_reasoning_content(mock_msg) == "Primary"

    def test_non_thinking_model(self):
        """Standard models like GPT-4o return None for all fields."""
        mock_msg = MagicMock()
        mock_msg.reasoning_content = None
        mock_msg.reasoning = None
        mock_msg.reasoning_details = None
        mock_msg.model_extra = None
        assert _extract_reasoning_content(mock_msg) is None

    def test_model_extra_reasoning_field(self):
        """Fallback to model_extra for 'reasoning' field name."""
        mock_msg = MagicMock(spec=[])
        mock_msg.model_extra = {"reasoning": "Reasoning in extras"}
        assert _extract_reasoning_content(mock_msg) == "Reasoning in extras"

    def test_model_extra_reasoning_details_field(self):
        """Fallback to model_extra for 'reasoning_details' field name."""
        mock_msg = MagicMock(spec=[])
        mock_msg.model_extra = {"reasoning_details": "Details in extras"}
        assert _extract_reasoning_content(mock_msg) == "Details in extras"

    def test_empty_string_treated_as_none(self):
        """Empty string reasoning_content should not be returned (falsy)."""
        mock_msg = MagicMock()
        mock_msg.reasoning_content = ""
        mock_msg.reasoning = None
        mock_msg.reasoning_details = None
        mock_msg.model_extra = {}
        assert _extract_reasoning_content(mock_msg) is None


class TestBackwardCompatibility:
    """Existing functionality is not broken by the new field."""

    def test_existing_chatmessage_creation_still_works(self):
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello",
            tool_calls=None,
            raw={"some": "data"},
        )
        assert msg.content == "Hello"
        assert msg.reasoning_content is None

    def test_from_dict_old_format(self):
        old_data = {
            "role": "assistant",
            "content": "Old message",
            "tool_calls": None,
        }
        msg = ChatMessage.from_dict(old_data)
        assert msg.content == "Old message"
        assert msg.reasoning_content is None

    def test_raw_not_included_in_serialization(self):
        """raw field is excluded from model_dump_json as before."""
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello",
            raw={"secret": "data"},
        )
        data = json.loads(msg.model_dump_json())
        assert "raw" not in data

    def test_chatmessage_with_tool_calls_unaffected(self):
        """Messages with tool calls still work correctly."""
        from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction

        tool_call = ChatMessageToolCall(
            id="call_1",
            type="function",
            function=ChatMessageToolCallFunction(name="my_tool", arguments={"arg": "val"}),
        )
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[tool_call],
        )
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.reasoning_content is None


class TestMemoryReasoningPropagation:
    """ActionStep correctly stores and propagates reasoning_content."""

    def _make_action_step(self, **kwargs) -> ActionStep:
        """Helper to create an ActionStep with reasonable defaults."""
        defaults = dict(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.0),
        )
        defaults.update(kwargs)
        return ActionStep(**defaults)

    def test_action_step_default_reasoning_content_is_none(self):
        step = self._make_action_step()
        assert step.reasoning_content is None

    def test_action_step_stores_reasoning_content(self):
        step = self._make_action_step(reasoning_content="My reasoning here")
        assert step.reasoning_content == "My reasoning here"

    def test_to_messages_includes_reasoning_when_present(self):
        """When reasoning_content is set, to_messages includes it on the assistant ChatMessage."""
        step = self._make_action_step(
            model_output="Final answer text",
            reasoning_content="Chain of thought",
        )
        messages = step.to_messages()
        # Should have at least the assistant message
        assert len(messages) > 0
        assistant_msg = messages[0]
        assert assistant_msg.role == MessageRole.ASSISTANT
        assert assistant_msg.reasoning_content == "Chain of thought"

    def test_to_messages_excludes_reasoning_when_absent(self):
        """When reasoning_content is None, to_messages returns None reasoning on assistant ChatMessage."""
        step = self._make_action_step(
            model_output="Final answer text",
            reasoning_content=None,
        )
        messages = step.to_messages()
        assert len(messages) > 0
        assistant_msg = messages[0]
        assert assistant_msg.reasoning_content is None

    def test_to_messages_summary_mode_skips_model_output(self):
        """In summary_mode, model_output is skipped, so no assistant message with reasoning."""
        step = self._make_action_step(
            model_output="Final answer text",
            reasoning_content="Some reasoning",
        )
        messages = step.to_messages(summary_mode=True)
        # In summary_mode, the model_output assistant message is not included
        for msg in messages:
            if msg.role == MessageRole.ASSISTANT:
                # If an assistant message does appear in summary_mode for other reasons, check it
                pass
        # Main assertion: summary_mode should not include the model_output assistant message
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]
        assert len(assistant_msgs) == 0

    def test_to_messages_without_model_output_no_reasoning(self):
        """If model_output is None, no assistant message (and no reasoning) is added."""
        step = self._make_action_step(
            model_output=None,
            reasoning_content="This should not appear",
        )
        messages = step.to_messages()
        assistant_msgs = [m for m in messages if m.role == MessageRole.ASSISTANT]
        assert len(assistant_msgs) == 0


class TestGetCleanMessageListPropagation:
    """get_clean_message_list propagates reasoning_content correctly."""

    def test_reasoning_content_propagated_in_clean_list(self):
        from smolagents.models import get_clean_message_list

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Hello"}],
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": "The answer"}],
                reasoning_content="My deep thoughts",
            ),
        ]
        clean = get_clean_message_list(messages)
        # Find the assistant message in the output
        assistant_msgs = [m for m in clean if m["role"] == MessageRole.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].get("reasoning_content") == "My deep thoughts"

    def test_reasoning_content_absent_not_in_clean_list(self):
        from smolagents.models import get_clean_message_list

        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Hello"}],
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": "Simple answer"}],
                reasoning_content=None,
            ),
        ]
        clean = get_clean_message_list(messages)
        assistant_msgs = [m for m in clean if m["role"] == MessageRole.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert "reasoning_content" not in assistant_msgs[0]
