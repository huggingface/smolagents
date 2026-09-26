import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory", "SummaryStep"]


logger = getLogger(__name__)


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
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[ChatMessage] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | list[dict[str, Any]] | None = None
    code_action: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ]
            if self.model_input_messages
            else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": make_json_serializable(get_dict_from_nested_dataclasses(self.model_output_message))
            if self.model_output_message
            else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
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
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[ChatMessage]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def dict(self):
        return {
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ],
            "model_output_message": make_json_serializable(
                get_dict_from_nested_dataclasses(self.model_output_message)
            ),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


@dataclass
class SummaryStep(MemoryStep):
    """A step that holds a summary of older steps that were compressed to save context space."""

    summary: str
    summarized_step_count: int

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"[Summary of previous {self.summarized_step_count} steps]\n{self.summary}"}],
            )
        ]


SUMMARIZATION_PROMPT = """Below is the history of previous steps taken by an AI agent. Summarize the key information, decisions, tool results, and findings into a concise paragraph. Preserve important facts, URLs, numbers, and conclusions. Omit redundant reasoning or failed attempts unless the failure is informative.

Steps to summarize:
{steps_text}

Write a concise summary:"""


class AgentMemory:
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str, max_steps_before_summary: int | None = None):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep | SummaryStep] = []
        self.max_steps_before_summary = max_steps_before_summary

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def summarize_if_needed(self, model) -> bool:
        """Summarize older steps if memory exceeds the configured threshold.

        Keeps the most recent steps intact and replaces older ones with a SummaryStep.
        Returns True if summarization was performed.

        Args:
            model: A model instance with a `generate` method to produce the summary.
        """
        if self.max_steps_before_summary is None:
            return False

        action_steps = [s for s in self.steps if isinstance(s, (ActionStep, PlanningStep))]
        if len(action_steps) <= self.max_steps_before_summary:
            return False

        keep_recent = self.max_steps_before_summary // 2
        steps_to_summarize = []
        steps_to_keep = []
        action_count = 0
        for step in reversed(self.steps):
            if isinstance(step, (ActionStep, PlanningStep)):
                action_count += 1
            if action_count <= keep_recent:
                steps_to_keep.insert(0, step)
            else:
                steps_to_summarize.insert(0, step)

        if not steps_to_summarize:
            return False

        steps_text = self._steps_to_text(steps_to_summarize)
        prompt = SUMMARIZATION_PROMPT.format(steps_text=steps_text)
        messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": prompt}])]

        response = model.generate(messages)
        summary_text = response.content if isinstance(response.content, str) else str(response.content)

        summary_step = SummaryStep(
            summary=summary_text,
            summarized_step_count=len(steps_to_summarize),
        )

        task_steps = [s for s in steps_to_summarize if isinstance(s, TaskStep)]
        self.steps = task_steps + [summary_step] + steps_to_keep
        logger.info(f"Summarized {len(steps_to_summarize)} steps into a single summary.")
        return True

    @staticmethod
    def _steps_to_text(steps: list) -> str:
        """Convert a list of steps to a plain text representation for summarization."""
        parts = []
        for step in steps:
            if isinstance(step, TaskStep):
                parts.append(f"Task: {step.task}")
            elif isinstance(step, ActionStep):
                if step.model_output:
                    output = step.model_output[:500] if len(step.model_output) > 500 else step.model_output
                    parts.append(f"Thought: {output}")
                if step.tool_calls:
                    for tc in step.tool_calls:
                        parts.append(f"Tool call: {tc.name}({tc.arguments})")
                if step.observations:
                    obs = step.observations[:500] if len(step.observations) > 500 else step.observations
                    parts.append(f"Observation: {obs}")
                if step.error:
                    parts.append(f"Error: {step.error}")
            elif isinstance(step, PlanningStep):
                parts.append(f"Plan: {step.plan[:500] if len(step.plan) > 500 else step.plan}")
        return "\n".join(parts)

    def get_succinct_steps(self) -> list[dict]:
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )


class CallbackRegistry:
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self):
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
