import json
from enum import IntEnum
from typing import TYPE_CHECKING

from rich.console import Console
from rich.rule import Rule
from rich.syntax import Syntax


if TYPE_CHECKING:
    from smolagents.memory import AgentMemory


class LogLevel(IntEnum):
    ERROR = 0  # Only errors
    INFO = 1  # Normal output (default)
    DEBUG = 2  # Detailed output


class AgentLogger:
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
        self.console = Console()

    def log(self, *args, level: str | LogLevel = LogLevel.INFO, **kwargs):
        """Logs a message to the console.

        Args:
            level (LogLevel, optional): Defaults to LogLevel.INFO.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def replay(self, agent_memory: "AgentMemory", full: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            with_memory (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        memory = []
        for step_log in agent_memory.steps:
            memory.extend(step_log.to_messages(show_model_input_messages=full))

        self.console.log("Replaying the agent's steps:")
        ix = 0
        for step in memory:
            role = step["role"].strip()
            if ix > 0 and role == "system":
                role == "memory"
            theme = "default"
            match role:
                case "assistant":
                    theme = "monokai"
                    ix += 1
                case "system":
                    theme = "monokai"
                case "tool-response":
                    theme = "github_dark"

            content = step["content"]
            try:
                content = eval(content)
            except Exception:
                content = [step["content"]]

            for substep_ix, item in enumerate(content):
                self.console.log(
                    Rule(
                        f"{role.upper()}, STEP {ix}, SUBSTEP {substep_ix + 1}/{len(content)}",
                        align="center",
                        style="orange",
                    ),
                    Syntax(
                        json.dumps(item, indent=4) if isinstance(item, dict) else str(item),
                        lexer="json",
                        theme=theme,
                        word_wrap=True,
                    ),
                )


__all__ = ["AgentLogger"]
