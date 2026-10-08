#!/usr/bin/env python
# coding=utf-8

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import io
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from smolagents.agent_types import AgentType
from smolagents.agents import MultiStepAgent
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep
from smolagents.models import ChatMessageStreamDelta, agglomerate_stream_deltas
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import _is_package_available


class _NullWriter(io.TextIOBase):
    def write(self, s: str) -> int:
        return len(s)


class TerminalCommand(str, Enum):
    HELP = "/help"
    DETAILS = "/details"
    RESET = "/reset"
    CLEAR = "/clear"
    EXIT = "/exit"


@dataclass
class CommandResponse:
    command: TerminalCommand
    message: str | None = None
    should_exit: bool = False


class TerminalUI:
    """Terminal user interface for interacting with a ``MultiStepAgent``.

    The class keeps command parsing and stream/event formatting separate from the Textual app,
    which makes it straightforward to unit test behavior without requiring the TUI runtime.
    """

    HELP_TEXT = (
        "Available commands:\n"
        "- `/help`: Show available commands\n"
        "- `/details`: Toggle reasoning/tool details visibility\n"
        "- `/reset`: Reset agent context and state\n"
        "- `/clear`: Clear visible transcript\n"
        "- `/exit`: Quit the TUI"
    )
    HIDDEN_DETAILS_NOTE = "_Details are hidden by default. Use `/details` to show reasoning and tool calls._"
    THROBBER_FRAMES = ("|", "/", "-", "\\")

    def __init__(self, agent: MultiStepAgent, reset_agent_memory: bool = False):
        self.agent = agent
        self.reset_agent_memory = reset_agent_memory
        self.show_details = False
        self._stream_deltas: list[ChatMessageStreamDelta] = []
        self._install_quiet_logger()

    def _install_quiet_logger(self) -> None:
        """Use a non-terminal logger so Rich Live output never corrupts TUI rendering."""
        quiet_console = Console(file=_NullWriter(), force_terminal=False, color_system=None, highlight=False)
        quiet_logger = AgentLogger(level=LogLevel.OFF, console=quiet_console)
        self.agent.logger = quiet_logger
        if hasattr(self.agent, "monitor"):
            self.agent.monitor.logger = quiet_logger

    def parse_command(self, user_input: str) -> TerminalCommand | None:
        normalized = user_input.strip().lower()
        if normalized == "/detail":
            return TerminalCommand.DETAILS
        for command in TerminalCommand:
            if normalized == command.value:
                return command
        return None

    def handle_command(self, user_input: str) -> CommandResponse | None:
        command = self.parse_command(user_input)
        if command is None:
            if user_input.strip().startswith("/"):
                return CommandResponse(command=TerminalCommand.HELP, message="Unknown command. Use `/help`.")
            return None

        if command == TerminalCommand.HELP:
            return CommandResponse(command=command, message=self.HELP_TEXT)
        if command == TerminalCommand.DETAILS:
            self.show_details = not self.show_details
            return CommandResponse(
                command=command,
                message=(
                    "Reasoning/tool details are now visible."
                    if self.show_details
                    else "Reasoning/tool details are now hidden."
                ),
            )
        if command == TerminalCommand.CLEAR:
            return CommandResponse(command=command)
        if command == TerminalCommand.EXIT:
            return CommandResponse(command=command, should_exit=True)

        # /reset
        self.reset_agent_context()
        return CommandResponse(command=command, message="Agent context has been reset.")

    def reset_agent_context(self) -> None:
        """Reset memory and mutable run state so the next turn starts cleanly."""
        self._stream_deltas = []
        if hasattr(self.agent, "memory"):
            self.agent.memory.reset()
        if hasattr(self.agent, "monitor"):
            self.agent.monitor.reset()
        if hasattr(self.agent, "state") and isinstance(self.agent.state, dict):
            self.agent.state.clear()

        executor = getattr(self.agent, "python_executor", None)
        if executor is not None and hasattr(executor, "state") and isinstance(executor.state, dict):
            executor.state = {"__name__": "__main__"}

    def append_stream_delta(self, event: ChatMessageStreamDelta) -> str:
        self._stream_deltas.append(event)
        return agglomerate_stream_deltas(self._stream_deltas).render_as_markdown()

    def flush_stream(self) -> str:
        if not self._stream_deltas:
            return ""
        output = agglomerate_stream_deltas(self._stream_deltas).render_as_markdown().strip()
        self._stream_deltas = []
        return output

    def iter_stream_events(self, prompt: str):
        return self.agent.run(
            prompt,
            stream=True,
            reset=self.reset_agent_memory,
            images=None,
            additional_args=None,
        )

    @staticmethod
    def _format_token_timing(token_usage, timing) -> str:
        details = []
        if token_usage is not None:
            details.append(f"tokens in/out: {token_usage.input_tokens}/{token_usage.output_tokens}")
        duration = getattr(timing, "duration", None) if timing is not None else None
        if duration is not None:
            details.append(f"duration: {duration:.2f}s")
        return " | ".join(details)

    @staticmethod
    def _truncate(value: Any, max_chars: int = 700) -> str:
        text = str(value).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def status_label(self, running: bool, tick: int = 0) -> str:
        if not running:
            return "Ready"
        frame = self.THROBBER_FRAMES[tick % len(self.THROBBER_FRAMES)]
        return f"Agent running {frame}"

    def format_event(self, event: ActionStep | PlanningStep | FinalAnswerStep) -> str:
        if isinstance(event, ActionStep):
            chunks = [f"**Step {event.step_number}**"]
            meta = self._format_token_timing(event.token_usage, event.timing)
            if meta:
                chunks.append(f"`{meta}`")
            if not self.show_details:
                return "\n\n".join(chunks)
            if event.tool_calls:
                chunks.append(f"`tools:` {', '.join(tool_call.name for tool_call in event.tool_calls)}")
            if event.code_action:
                chunks.append(f"```python\n{self._truncate(event.code_action)}\n```")
            if event.observations:
                chunks.append(f"`observation:`\n```\n{self._truncate(event.observations)}\n```")
            if event.error:
                chunks.append(f"`error:` {self._truncate(event.error)}")
            return "\n\n".join(chunks)

        if isinstance(event, PlanningStep):
            output = ["**Planning**"]
            meta = self._format_token_timing(event.token_usage, event.timing)
            if meta:
                output.append(f"`{meta}`")
            if self.show_details:
                output.insert(1, self._truncate(event.plan))
            return "\n\n".join(output)

        final_answer = event.output
        if isinstance(final_answer, AgentType):
            rendered = final_answer.to_string()
        else:
            rendered = str(final_answer)
        return f"**Final answer**\n\n{self._truncate(rendered)}"

    def launch(self) -> None:
        if not _is_package_available("textual"):
            raise ModuleNotFoundError(
                "Please install 'tui' extra to use the TerminalUI: `pip install 'smolagents[tui]'`"
            )

        from textual.app import App, ComposeResult
        from textual.containers import VerticalScroll
        from textual.widgets import Header, Input, Static

        ui = self

        class _TerminalApp(App):
            CSS = """
            Screen {
                layout: vertical;
            }

            #status {
                height: 1;
                padding: 0 1;
                color: $text-muted;
                background: $surface;
            }

            #transcript-container {
                height: 1fr;
                padding: 0 1;
            }

            #transcript {
                width: 100%;
            }

            #prompt {
                dock: bottom;
            }
            """

            BINDINGS = [("ctrl+c", "quit", "Quit")]

            def __init__(self):
                super().__init__()
                self._entries: list[str] = []
                self._is_running = False
                self._streaming_visible = False
                self._thread: threading.Thread | None = None
                self._spinner_tick = 0
                self._spinner_timer = None

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with VerticalScroll(id="transcript-container"):
                    yield Static("", id="transcript")
                yield Static("", id="status")
                yield Input(placeholder="Ask smolagent, or /help", id="prompt")

            def on_mount(self) -> None:
                self._entries = [self._top_banner()]
                self._render()
                self.query_one("#prompt", Input).focus()

            def _set_status(self, value: str) -> None:
                self.query_one("#status", Static).update(value)

            def _start_spinner(self) -> None:
                self._spinner_tick = 0
                if self._spinner_timer is None:
                    self._spinner_timer = self.set_interval(0.1, self._tick_spinner, pause=True)
                self._spinner_timer.resume()
                self._tick_spinner()

            def _stop_spinner(self) -> None:
                if self._spinner_timer is not None:
                    self._spinner_timer.pause()

            def _tick_spinner(self) -> None:
                if not self._is_running:
                    return
                self._set_status(ui.status_label(running=True, tick=self._spinner_tick))
                self._spinner_tick += 1

            def _top_banner(self) -> str:
                banner = "**smolagent TUI**\n\nType `/help` for commands."
                if not ui.show_details:
                    banner += f"\n\n{ui.HIDDEN_DETAILS_NOTE}"
                return banner

            def _render(self) -> None:
                transcript = self.query_one("#transcript", Static)
                content = "\n\n---\n\n".join(self._entries)
                transcript.update(Markdown(content if content else ""))
                self.query_one("#transcript-container", VerticalScroll).scroll_end(animate=False)

            def _add_entry(self, content: str) -> None:
                self._entries.append(content)
                self._render()

            def _clear_entries(self) -> None:
                self._entries = [self._top_banner()]
                self._streaming_visible = False
                self._render()

            def _update_stream_entry(self, content: str) -> None:
                if not ui.show_details:
                    return
                stream_entry = f"**Assistant (streaming)**\n\n{content}"
                if self._streaming_visible and self._entries:
                    self._entries[-1] = stream_entry
                else:
                    self._entries.append(stream_entry)
                    self._streaming_visible = True
                self._render()

            def _finalize_stream_entry(self) -> None:
                stream_output = ui.flush_stream()
                if not stream_output:
                    self._streaming_visible = False
                    return
                if not ui.show_details:
                    self._streaming_visible = False
                    return
                final_entry = f"**Assistant**\n\n{stream_output}"
                if self._streaming_visible and self._entries:
                    self._entries[-1] = final_entry
                else:
                    self._entries.append(final_entry)
                self._streaming_visible = False
                self._render()

            def _set_running(self, running: bool) -> None:
                self._is_running = running
                prompt_input = self.query_one("#prompt", Input)
                prompt_input.disabled = running
                if running:
                    self._start_spinner()
                else:
                    self._stop_spinner()
                    self._set_status("")
                if not running:
                    prompt_input.focus()

            def _run_agent_in_thread(self, prompt: str) -> None:
                try:
                    for event in ui.iter_stream_events(prompt):
                        if isinstance(event, ChatMessageStreamDelta):
                            rendered = ui.append_stream_delta(event)
                            self.call_from_thread(self._update_stream_entry, rendered)
                            continue

                        if isinstance(event, (ActionStep, PlanningStep, FinalAnswerStep)):
                            self.call_from_thread(self._finalize_stream_entry)
                            formatted = ui.format_event(event)
                            self.call_from_thread(self._add_entry, formatted)

                    self.call_from_thread(self._finalize_stream_entry)
                except Exception as error:  # pragma: no cover - visual fallback path
                    self.call_from_thread(self._add_entry, f"**Error**\n\n{error}")
                finally:
                    self.call_from_thread(self._set_running, False)

            def _start_agent_run(self, prompt: str) -> None:
                self._set_running(True)
                self._thread = threading.Thread(target=self._run_agent_in_thread, args=(prompt,), daemon=True)
                self._thread.start()

            def on_input_submitted(self, event: Input.Submitted) -> None:
                user_input = event.value.strip()
                event.input.value = ""
                if not user_input:
                    return

                if self._is_running:
                    self._set_status("Agent run already in progress")
                    return

                command_result = ui.handle_command(user_input)
                if command_result is not None:
                    if command_result.command == TerminalCommand.CLEAR:
                        self._clear_entries()
                    if command_result.message:
                        self._add_entry(f"**System**\n\n{command_result.message}")
                    if command_result.should_exit:
                        self.exit()
                    return

                self._add_entry(f"**You**\n\n{user_input}")
                self._start_agent_run(user_input)

        _TerminalApp().run()


__all__ = ["TerminalUI", "TerminalCommand", "CommandResponse"]
