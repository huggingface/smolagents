from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Dict, List, TypedDict, Union
from smolagents.agent_types import AgentInput
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
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
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
    task_id: str = None

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
        if self.model_output is not None and not summary_mode and not isinstance(self.action_output,AgentInput):
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            pass
            # messages.append(
            #     Message(
            #         role=MessageRole.ASSISTANT,
            #         content=[
            #             {
            #                 "type": "text",
            #                 "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
            #             }
            #         ],
            #     )
            # )

        messages.append(
            Message(
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
    model_input_messages: List[Message]
    model_output_message_facts: ChatMessage
    facts: str
    model_output_message_plan: ChatMessage
    plan: str
    task_id: str = None

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
    task: str
    task_images: List[str] | None = None
    task_id: str = None

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        content = [{"type": "text", "text": f"""Task:{self.task}
Rules:
1. Carefully review all observation information: the task, image path,execution logs, context, tool.
2. Analyze the tool's inputs from the metadata to understand required and optional parameters.
3. Even if the user explicitly requests it,do not access the memory variable's attribute if you are unsure of the variable type.
4. Ensure all required parameters about the task are included from observation otherwise you can use 'user_input' tool to ask user for more information.
5. You must plan forward to proceed in a series of steps, in a cycle of 'Thought:', 'Code:', and 'Observation:' sequences,specily the 'Code:' sequence must end with '<end_code>'.
6. In the end you have to return f-string which contains all steps answer as the final answer using the `final_answer` tool.
7. Always respond in Chinese."""}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False, **kwargs) -> List[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


class AgentMemory:
    def __init__(self, system_prompt: str,memory_size:int=100):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: List[Union[TaskStep, ActionStep, PlanningStep]] = FIFOList(maxsize=memory_size)

    def reset(self):
        self.steps.clear()

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

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


class FIFOList:
    def __init__(self, maxsize):
        if not isinstance(maxsize, int):
            raise TypeError("maxsize must be an integer")
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")

        self._maxsize = maxsize
        self._list = []

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        return self._list[index]

    def __setitem__(self, index, value):
        self._list[index] = value

    def __iter__(self):
        return iter(self._list)

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return f"FIFOList(maxsize={self._maxsize}, items={self._list})"

    def append(self, item):
        if len(self._list) >= self._maxsize:
            task_id = getattr(self._list.pop(0), "task_id")
            while task_id is not None:
                if task_id == getattr(self._list[0], "task_id"):
                    self._list.pop(0)
                else:
                    task_id = None

        self._list.append(item)
        return len(self._list)

    def extend(self, items):
        for item in items:
            self.append(item)

    def clear(self):
        self._list.clear()


__all__ = ["AgentMemory"]
