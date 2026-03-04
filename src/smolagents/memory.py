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


__all__ = ["AgentMemory", "MemorySummaryStep"]


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
class MemorySummaryStep(MemoryStep):
    """A step that holds a consolidated summary of earlier memory steps.

    When ``AgentMemory.consolidate`` is called the oldest interaction steps
    are replaced by a single ``MemorySummaryStep`` so that the context sent
    to the model stays bounded.

    Args:
        summary: The LLM-generated summary text.
    """

    summary: str

    def to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        return [
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": f"[Consolidated summary of earlier steps]\n{self.summary}",
                    }
                ],
            )
        ]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


class AgentMemory:
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.
        max_memory_steps (`int` or `None`, default `None`):
            Maximum number of interaction steps (``ActionStep`` and ``PlanningStep``) to
            keep in full detail.  When the number of such steps exceeds this
            value, older steps are summarised into a single
            ``MemorySummaryStep`` via :meth:`consolidate`.
            ``None`` disables automatic consolidation.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep | MemorySummaryStep]`) -- List of steps taken by the agent, which can include tasks, actions, planning steps, and memory summary steps.
    """

    def __init__(self, system_prompt: str, max_memory_steps: int | None = None):
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep | MemorySummaryStep] = []
        self.max_memory_steps = max_memory_steps

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def consolidate(self, model) -> bool:
        """Summarise the oldest interaction steps to keep memory bounded.

        When the number of ``ActionStep`` / ``PlanningStep`` entries exceeds
        ``max_memory_steps``, older steps are summarised by *model* into a
        single ``MemorySummaryStep``.  The ``TaskStep`` at the beginning of
        the conversation is preserved.  Any existing ``MemorySummaryStep``
        entries are folded into the new summary to prevent unbounded growth.

        Args:
            model: A model instance with a ``generate`` method (the same model
                used by the agent).

        Returns:
            ``True`` if consolidation was performed, ``False`` otherwise.
        """
        if self.max_memory_steps is None:
            return False

        interaction_indices = [
            i for i, s in enumerate(self.steps) if isinstance(s, (ActionStep, PlanningStep))
        ]
        if len(interaction_indices) <= self.max_memory_steps:
            return False

        # Keep the most recent `max_memory_steps` interaction steps untouched.
        n_to_summarise = len(interaction_indices) - self.max_memory_steps
        indices_to_summarise = interaction_indices[:n_to_summarise]

        # Also fold any existing MemorySummaryStep entries into the new summary
        # to prevent unbounded growth of summary steps.
        existing_summary_indices = [
            i for i, s in enumerate(self.steps) if isinstance(s, MemorySummaryStep)
        ]

        # Build text from the steps that will be consolidated.
        summary_parts: list[str] = []

        # Include existing summaries first for context continuity.
        for idx in existing_summary_indices:
            step = self.steps[idx]
            summary_parts.append(f"[Previous summary] {step.summary}")

        for idx in indices_to_summarise:
            step = self.steps[idx]
            if isinstance(step, ActionStep):
                part = f"[Step {step.step_number}]"
                if step.model_output:
                    part += f" Thought: {step.model_output}"
                if step.tool_calls:
                    part += f" | Tool calls: {[tc.name for tc in step.tool_calls]}"
                if step.observations:
                    obs_trunc = step.observations[:500]
                    part += f" | Observation: {obs_trunc}"
                if step.error:
                    part += f" | Error: {step.error}"
                summary_parts.append(part)
            elif isinstance(step, PlanningStep):
                summary_parts.append(f"[Planning] {step.plan[:500]}")

        text_to_summarise = "\n".join(summary_parts)

        prompt_messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": (
                            "You are a memory consolidation assistant. "
                            "Summarise the following agent interaction history into a concise but "
                            "information-preserving summary. Keep key facts, tool results, errors, "
                            "and decisions. Be concise."
                        ),
                    }
                ],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": f"Summarise these agent steps:\n\n{text_to_summarise}",
                    }
                ],
            ),
        ]

        try:
            response = model.generate(prompt_messages)
            # Prefer a normalized text representation of the response (e.g. markdown),
            # and fall back to truncation when no usable text is available.
            if hasattr(response, "render_as_markdown"):
                rendered = response.render_as_markdown()
            elif isinstance(response.content, str):
                rendered = response.content
            elif isinstance(response.content, list):
                # Extract text from structured content blocks (e.g. [{"type": "text", "text": "..."}])
                rendered = " ".join(
                    block.get("text", "") for block in response.content if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                rendered = ""
            rendered = (rendered or "").strip()
            summary_text = rendered or text_to_summarise[:2000]
        except Exception as e:
            logger.warning("Memory consolidation LLM call failed (%s); falling back to truncation.", e)
            summary_text = text_to_summarise[:2000]

        summary_step = MemorySummaryStep(summary=summary_text)

        # Remove summarised steps and old summary steps; insert new summary in their place.
        indices_set = set(indices_to_summarise) | set(existing_summary_indices)
        new_steps: list = []
        inserted = False
        for i, step in enumerate(self.steps):
            if i in indices_set:
                if not inserted:
                    new_steps.append(summary_step)
                    inserted = True
                # skip this old step
            else:
                new_steps.append(step)
        self.steps = new_steps
        return True

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
            elif isinstance(step, MemorySummaryStep):
                logger.log_rule("Memory summary", level=LogLevel.ERROR)
                logger.log_markdown(title="Consolidated memory:", content=step.summary, level=LogLevel.ERROR)

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
