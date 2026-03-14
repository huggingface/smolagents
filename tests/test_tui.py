from unittest.mock import Mock, patch

import pytest

from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep, ToolCall
from smolagents.models import ChatMessage, ChatMessageStreamDelta, MessageRole
from smolagents.monitoring import Timing, TokenUsage
from smolagents.tui import TerminalCommand, TerminalUI


def make_mock_agent():
    agent = Mock()
    agent.memory = Mock()
    agent.monitor = Mock()
    agent.state = {"cached": 123}
    agent.python_executor = Mock()
    agent.python_executor.state = {"cached": "value"}
    return agent


def test_handle_help_command():
    ui = TerminalUI(make_mock_agent())

    result = ui.handle_command("/help")

    assert result is not None
    assert result.command == TerminalCommand.HELP
    assert "/reset" in result.message
    assert "/details" in result.message


def test_handle_details_command_toggles_visibility():
    ui = TerminalUI(make_mock_agent())

    first = ui.handle_command("/details")
    second = ui.handle_command("/details")

    assert first is not None
    assert first.command == TerminalCommand.DETAILS
    assert ui.show_details is False
    assert second is not None
    assert second.command == TerminalCommand.DETAILS


def test_handle_detail_alias():
    ui = TerminalUI(make_mock_agent())

    result = ui.handle_command("/detail")

    assert result is not None
    assert result.command == TerminalCommand.DETAILS


def test_handle_clear_command():
    ui = TerminalUI(make_mock_agent())

    result = ui.handle_command("/clear")

    assert result is not None
    assert result.command == TerminalCommand.CLEAR
    assert result.message is None


def test_handle_unknown_command():
    ui = TerminalUI(make_mock_agent())

    result = ui.handle_command("/does-not-exist")

    assert result is not None
    assert result.command == TerminalCommand.HELP
    assert "Unknown command" in result.message


def test_reset_command_clears_state_and_memory():
    agent = make_mock_agent()
    ui = TerminalUI(agent)

    result = ui.handle_command("/reset")

    assert result is not None
    assert result.command == TerminalCommand.RESET
    agent.memory.reset.assert_called_once()
    agent.monitor.reset.assert_called_once()
    assert agent.state == {}
    assert agent.python_executor.state == {"__name__": "__main__"}


def test_stream_delta_aggregation_and_flush():
    ui = TerminalUI(make_mock_agent())

    first = ui.append_stream_delta(ChatMessageStreamDelta(content="Hello"))
    second = ui.append_stream_delta(ChatMessageStreamDelta(content=" world"))
    flushed = ui.flush_stream()

    assert first == "Hello"
    assert second == "Hello world"
    assert flushed == "Hello world"
    assert ui.flush_stream() == ""


def test_format_action_step_event_contains_compact_fields():
    ui = TerminalUI(make_mock_agent())
    ui.show_details = True

    step = ActionStep(
        step_number=2,
        timing=Timing(start_time=0.0, end_time=2.0),
        tool_calls=[ToolCall(name="web_search", arguments={"query": "x"}, id="1")],
        observations="Search completed.",
        token_usage=TokenUsage(input_tokens=10, output_tokens=5),
    )

    rendered = ui.format_event(step)

    assert "Step 2" in rendered
    assert "web_search" in rendered
    assert "Search completed" in rendered
    assert "tokens in/out" in rendered


def test_format_planning_and_final_events():
    ui = TerminalUI(make_mock_agent())
    ui.show_details = True

    planning = PlanningStep(
        model_input_messages=[ChatMessage(role=MessageRole.USER, content="task")],
        model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content="plan"),
        plan="First gather data, then summarize.",
        timing=Timing(start_time=0.0, end_time=1.0),
        token_usage=TokenUsage(input_tokens=2, output_tokens=3),
    )
    final = FinalAnswerStep(output="done")

    planning_rendered = ui.format_event(planning)
    final_rendered = ui.format_event(final)

    assert "Planning" in planning_rendered
    assert "First gather data" in planning_rendered
    assert "Final answer" in final_rendered
    assert "done" in final_rendered


def test_format_action_step_hides_details_by_default():
    ui = TerminalUI(make_mock_agent())
    step = ActionStep(
        step_number=1,
        timing=Timing(start_time=0.0, end_time=1.0),
        tool_calls=[ToolCall(name="web_search", arguments={"query": "x"}, id="1")],
        observations="Some observation",
        token_usage=TokenUsage(input_tokens=1, output_tokens=2),
    )

    rendered = ui.format_event(step)

    assert "Step 1" in rendered
    assert "web_search" not in rendered
    assert "Some observation" not in rendered


def test_launch_raises_when_textual_missing():
    ui = TerminalUI(make_mock_agent())

    with patch("smolagents.tui._is_package_available", return_value=False):
        with pytest.raises(ModuleNotFoundError, match="smolagents\\[tui\\]"):
            ui.launch()


def test_status_label_throbber_frames():
    ui = TerminalUI(make_mock_agent())

    assert ui.status_label(running=False) == "Ready"
    assert ui.status_label(running=True, tick=0) == "Agent running |"
    assert ui.status_label(running=True, tick=1) == "Agent running /"
    assert ui.status_label(running=True, tick=2) == "Agent running -"
    assert ui.status_label(running=True, tick=3) == "Agent running \\"
