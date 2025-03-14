from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, Protocol, TypedDict, Union

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep(ABC):
    """
    Abstract base class for memory steps.

    A memory step represents a single step in an agent's execution.
    Different types of steps (action, planning, task, system prompt) inherit from this class.
    """

    def dict(self):
        """Convert the memory step to a dictionary."""
        return asdict(self)

    @abstractmethod
    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Convert the memory step to a list of messages that can be used as input to the LLM.

        Returns:
            List[Dict[str, Any]]: A list of messages.
        """
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """
    A memory step that represents an action taken by the agent.

    This includes the model's input and output, tool calls, observations, and any errors.
    """

    model_input_messages: List[Message] | None = None
    tool_calls: List[ToolCall] | None = None
    start_time: float | None = None
    end_time: float | None = None
    step_number: int | None = None
    error: AgentError | None = None
    duration: float | None = None
    model_output_message: ChatMessage = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: List[str] | None = None
    action_output: Any = None

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "step": self.step_number,
            "error": self.error.dict() if self.error else None,
            "duration": self.duration,
            "model_output_message": self.model_output_message,
            "model_output": self.model_output,
            "observations": self.observations,
            "action_output": make_json_serializable(self.action_output),
        }

    def to_messages(self, summary_mode: bool = False, show_model_input_messages: bool = False) -> List[Message]:
        messages = []
        if self.model_input_messages is not None and show_model_input_messages:
            messages.append(Message(role=MessageRole.SYSTEM, content=self.model_input_messages))
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": (f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else "")
                            + f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Here are the observed images:"}]
                    + [
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )
        return messages


@dataclass
class PlanningStep(MemoryStep):
    """
    A memory step that represents a planning step taken by the agent.

    This includes the model's input and output for facts and plan generation.
    """

    model_input_messages: List[Message]
    model_output_message_facts: ChatMessage
    facts: str
    model_output_message_plan: ChatMessage
    plan: str

    def to_messages(self, summary_mode: bool, **kwargs) -> List[Message]:
        messages = []
        messages.append(
            Message(
                role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[FACTS LIST]:\n{self.facts.strip()}"}]
            )
        )

        if not summary_mode:  # This step is not shown to a model writing a plan to avoid influencing the new plan
            messages.append(
                Message(
                    role=MessageRole.ASSISTANT, content=[{"type": "text", "text": f"[PLAN]:\n{self.plan.strip()}"}]
                )
            )
        return messages


@dataclass
class TaskStep(MemoryStep):
    """
    A memory step that represents a task given to the agent.

    This includes the task description and any associated images.
    """

    task: str
    task_images: List[str] | None = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    """
    A memory step that represents the system prompt given to the agent.
    """

    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class MemoryProvider(Protocol):
    """
    Protocol defining the interface for memory providers.

    Memory providers are responsible for storing and retrieving memory steps.
    """

    def reset(self) -> None:
        """Reset the memory."""
        ...

    def get_succinct_steps(self) -> list[dict]:
        """Get a succinct representation of the memory steps."""
        ...

    def get_full_steps(self) -> list[dict]:
        """Get a full representation of the memory steps."""
        ...

    def replay(self, logger: AgentLogger, detailed: bool = False) -> None:
        """Replay the memory steps."""
        ...

    def write_to_messages(self, summary_mode: bool = False) -> List[Dict[str, Any]]:
        """Convert the memory steps to a list of messages."""
        ...


class AgentMemory:
    """
    Default implementation of agent memory.

    This class stores memory steps in a list and provides methods to access and manipulate them.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep]] = []

    def reset(self):
        """Reset the memory by clearing all steps."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """
        Get a succinct representation of the memory steps.

        This excludes model_input_messages to reduce verbosity.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """
        Get a full representation of the memory steps.

        Returns:
            list[dict]: A list of dictionaries representing the memory steps.
        """
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """
        Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        for step in self.steps:
            if isinstance(step, SystemPromptStep) and detailed:
                logger.log_markdown(title="System prompt", content=step.system_prompt, level=LogLevel.ERROR)
            elif isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages)
                logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.facts + "\n" + step.plan, level=LogLevel.ERROR)

    def write_to_messages(self, summary_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Convert the memory steps to a list of messages.

        Args:
            summary_mode (bool, optional): If True, exclude certain details to create a summary. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of messages.
        """
        messages = self.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages


# For backward compatibility
__all__ = [
    "AgentMemory",
    "MemoryStep",
    "ActionStep",
    "PlanningStep",
    "TaskStep",
    "SystemPromptStep",
    "ToolCall",
    "Message",
]
