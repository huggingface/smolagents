#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
import json
import os
import re
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Type, TypeAlias, TypedDict, Union

import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

try:
    from langfuse import Langfuse
except Exception:  # pragma: no cover - optional dependency
    Langfuse = None

from .agent_types import AgentAudio, AgentImage
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor
from .memory import (
    ActionStep,
    AgentMemory,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from .models import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    MessageRole,
    Model,
)
from .monitoring import (
    AgentLogger,
    LogLevel,
    Monitor,
)
from .remote_executors import DockerExecutor, E2BExecutor
from .tools import ReceiveMessagesTool, SendMessageTool, Tool, validate_tool_arguments
from .utils import (
    AGENT_GRADIO_APP_TEMPLATE,
    AgentError,
    AgentMaxRuntimeError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)


logger = getLogger(__name__)

langfuse = None
if Langfuse is not None:
    langfuse = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def get_variable_names(self, template: str) -> set[str]:
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class ActionOutput:
    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        messages (list[dict]): The agent's memory, as a list of messages.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    messages: list[dict]
    token_usage: TokenUsage | None
    timing: Timing


StreamEvent: TypeAlias = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]


class MultiStepAgent(ABC):
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        instructions (`str`, *optional*): Custom instructions for the agent, will be inserted in the system prompt.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        max_runtime (`int`, default `300`): Maximum runtime allowed for `run()` in seconds.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
            <Deprecated version="1.17.0">
            Parameter `grammar` is deprecated and will be removed in version 1.20.
            </Deprecated>
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable | list[Callable]]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list[Callable]`, *optional*): List of validation functions to run before accepting a final answer.
            Each function should:
            - Take the final answer and the agent's memory as arguments.
            - Return a boolean indicating whether the final answer is valid.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        agent_id: int = 0,
        queue_dict: dict | None = None,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        max_runtime: int = 300,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | dict[Type[MemoryStep], Callable | list[Callable]] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.agent_id = agent_id

        if queue_dict is None:
            queue_dict = {agent_id: Queue()}
        else:
            queue_dict.setdefault(agent_id, Queue())
        self.queue_dict = queue_dict
        self.queue = queue_dict[agent_id]
        self._send_message_tool = SendMessageTool(queue_dict)
        self._receive_messages_tool = ReceiveMessagesTool(agent_id, queue_dict)
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        self.max_steps = max_steps
        self.max_runtime = max_runtime
        self.step_number = 0
        if grammar is not None:
            warnings.warn(
                "Parameter 'grammar' is deprecated and will be removed in version 1.20.",
                FutureWarning,
            )
        self.grammar = grammar
        self.planning_interval = planning_interval
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks if final_answer_checks is not None else []
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None
        self.memory = AgentMemory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self._setup_step_callbacks(step_callbacks)
        self.stream_outputs = False

        self.interrupt_switch = False
        
        # Initialize Langfuse trace if available
        self.trace = None
        if hasattr(langfuse, "trace"):
            try:
                self.trace = langfuse.trace(name=f"Agent_{agent_id}_trace")
            except Exception:
                self.trace = None


    def send_message(self, target_id: int, message: Any) -> None:
        """Send a message to the target agent's queue."""
        span = None
        if self.trace:
            span = self.trace.span(name=f"send_message_{self.agent_id}")
            span.update(inputs={"target_id": target_id, "message": str(message)})
        try:
            self._send_message_tool(target_id, message)
            self.logger.log(f"Agent {self.agent_id} sent message to Agent {target_id}", level=LogLevel.INFO)
            if span:
                span.update(outputs={"status": "success"})
        except ValueError:
            if span:
                span.update(outputs={"status": "error", "error": f"Target {target_id} not found"})
            self.logger.log(
                f"Agent {self.agent_id} failed to send message: Target {target_id} not found",
                level=LogLevel.WARNING,
            )
        finally:
            if span:
                span.end()

    def receive_messages(self) -> None:
        """Process all incoming messages in the agent's queue."""
        for message in self._receive_messages_tool():
            self.process_message(message)

    def process_message(self, message: Any) -> None:
        """Handle a received message."""
        self.logger.log(f"Agent {self.agent_id} received message: {message}", level=LogLevel.INFO)
        # Example: If message is a task, treat it as a new task
        if isinstance(message, str):
            self.task = message
            self.memory.steps.append(TaskStep(task=self.task))
            # Process task (e.g., generate code or execute)
            self._process_task()
        elif isinstance(message, dict) and "tool_call" in message:
            # Handle tool call results or delegated tasks
            self._process_tool_call(message["tool_call"])

    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}
            # Ensure managed agents can be called as tools by the model: set their inputs and output_type
            for agent in self.managed_agents.values():
                agent.inputs = {
                    "task": {"type": "string", "description": "Long detailed description of the task."},
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                    },
                }
                agent.output_type = "string"

    def _setup_tools(self, tools, add_base_tools):
        assert all(isinstance(tool, Tool) for tool in tools), "All elements must be instance of Tool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        # Initialize step callbacks registry
        self.step_callbacks = CallbackRegistry()
        if step_callbacks:
            # Register callbacks list only for ActionStep for backward compatibility
            if isinstance(step_callbacks, list):
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            # Register callbacks dict for specific step classes
            elif isinstance(step_callbacks, dict):
                for step_cls, callbacks in step_callbacks.items():
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                raise ValueError("step_callbacks must be a list or a dict")
        # Register monitor update_metrics only for ActionStep for backward compatibility
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str | None = None,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_runtime: int | None = None,  # Maximum runtime in seconds
    ):
        """
        Run the agent in a decentralized manner, processing tasks from its queue.
        The vanilla method use a centralized ReAct loop, but this method allows for decentralized processing.

        Args:
            task (`str`, *optional*): Initial task to perform.
            reset (`bool`): Whether to reset the conversation or keep it going.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Additional variables for the agent.
            max_runtime (`int`, *optional*): Maximum runtime in seconds before terminating. If None, uses
            the agent's `max_runtime` attribute.
        """

        self.task = task
        self.interrupt_switch = False
        self.final_result = None
        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()
            self.step_number = 0
            self.interrupt_switch = False

        if task:
            self.task = task
            self.memory.steps.append(TaskStep(task=self.task, task_images=images))
            self.logger.log_task(
                content=self.task.strip(),
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                level=LogLevel.INFO,
                title=f"Agent {self.agent_id}",
            )

        if additional_args is not None:
            self.state.update(additional_args)

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        # Decentralized loop: process messages and tasks
        if max_runtime is None:
            max_runtime = self.max_runtime

        start_time = time.time()
        try:
            while True:
                if max_runtime is not None and time.time() - start_time >= max_runtime:
                    raise AgentMaxRuntimeError("Reached maximum runtime.", self.logger)

                self.step_number += 1

                if self.step_number > self.max_steps:
                    raise AgentMaxStepsError("Reached maximum number of steps.", self.logger)

                if self.interrupt_switch:
                    break

                self.receive_messages()

                if self.task:
                    self._process_task()
                else:
                    break

                time.sleep(1)  # Prevent busy loop; adjust as needed
            return self.final_result
        except AgentMaxStepsError as e:
            self.logger.log(f"Max steps reached: {e}", level=LogLevel.WARNING)
            final_answer = self._handle_max_steps_reached(self.task, images)
            return final_answer
        except AgentMaxRuntimeError as e:
            self.logger.log(f"Max runtime reached: {e}", level=LogLevel.WARNING)
            final_answer = self._handle_max_runtime_reached(self.task, images)
            return final_answer
        except AgentError as e:
            self.logger.log(f"Agent error: {e}", level=LogLevel.ERROR)
            return str(e)
        except Exception as e:
            self.logger.log(f"Unexpected error: {e}", level=LogLevel.ERROR)
            raise AgentError(f"Unexpected error: {e}", self.logger)

    def _process_task(self) -> None:
        """Process the current task, e.g., generate code or delegate to another agent."""
        try:
            # Example: Agent 0 generates code, sends to Agent 1 for execution
            if self.agent_id == 0:
                code = self._generate_code()
                self.send_message(1, code)
            # Agent 1 might execute code or use tools
            elif self.agent_id == 1:
                self._process_tool_call({"name": "python_interpreter", "arguments": self.task})
        except AgentError as e:
            self.logger.log(f"Error processing task: {e}", level=LogLevel.ERROR)
            self.send_message(0, {"error": str(e)})

    def _generate_code(self) -> str:
        """Generate code for the task (customize based on CodeAgent logic)."""
        memory_messages = self.write_memory_to_messages()
        input_messages = memory_messages + [
            ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": f"Generate code for: {self.task}"}])
        ]
        chat_message = self.model.generate(input_messages)
        code = parse_code_blobs(chat_message.content) or chat_message.content
        self.logger.log_code(title=f"Agent {self.agent_id} generated code:", content=code, level=LogLevel.INFO)
        return code

    def _process_tool_call(self, tool_call: dict) -> None:
        """Execute a tool call received from another agent."""
        tool_name = tool_call["name"]
        arguments = tool_call["arguments"]
        try:
            result = self.execute_tool_call(tool_name, arguments)
            self.logger.log(f"Agent {self.agent_id} tool result: {result}", level=LogLevel.INFO)
            if tool_name == "final_answer":
                valid = True
                for check in self.final_answer_checks:
                    try:
                        if not check(result, self.memory):
                            valid = False
                    except Exception as e:
                        valid = False
                        self.logger.log(f"Final answer check failed: {e}", level=LogLevel.ERROR)
                if valid:
                    self.memory.steps.append(FinalAnswerStep(output=result))
                    self.final_result = result
                    self.task = None
                    self.interrupt_switch = True
                    # Optionally send result to another agent
                    self.send_message(0, {"tool_call": {"name": "final_answer", "arguments": result}})
            else:
                self.send_message(0, {"tool_call": {"name": "final_answer", "arguments": result}})
        except AgentToolExecutionError as e:
            self.logger.log(f"Tool execution error: {e}", level=LogLevel.ERROR)

    def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _handle_max_runtime_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task, images)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxRuntimeError("Reached max runtime.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _finalize_step(self, memory_step: MemoryStep):
        """Finalize a step by running registered callbacks."""
        if getattr(memory_step, "timing", None) and memory_step.timing.end_time is None:
            memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        span = None
        if self.trace:
            span = self.trace.span(name=f"planning_step_{self.agent_id}")
            span.update(inputs={"task": task, "is_first_step": is_first_step, "step": step})

        start_time = time.time()
        try:
            if is_first_step:
                input_messages = [
                    ChatMessage(
                        role=MessageRole.USER,
                        content=[
                            {
                                "type": "text",
                                "text": populate_template(
                                    self.prompt_templates["planning"]["initial_plan"],
                                    variables={
                                        "task": task,
                                        "tools": self.tools,
                                        "managed_agents": self.managed_agents,
                                    },
                                ),
                            }
                        ],
                    )
                ]
                if self.stream_outputs and hasattr(self.model, "generate_stream"):
                    plan_message_content = ""
                    output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                    input_tokens, output_tokens = 0, 0
                    with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                        for event in output_stream:
                            if event.content is not None:
                                plan_message_content += event.content
                                live.update(Markdown(plan_message_content))
                                if event.token_usage:
                                    output_tokens += event.token_usage.output_tokens
                                    input_tokens = event.token_usage.input_tokens
                            yield event
                else:
                    plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                    plan_message_content = plan_message.content
                    input_tokens, output_tokens = (
                        (
                            plan_message.token_usage.input_tokens,
                            plan_message.token_usage.output_tokens,
                        )
                        if plan_message.token_usage
                        else (None, None)
                    )
                plan = textwrap.dedent(
                    f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
                )
            else:
                # Summary mode removes the system prompt and previous planning messages output by the model.
                # Removing previous planning messages avoids influencing too much the new plan.
                memory_messages = self.write_memory_to_messages(summary_mode=True)
                plan_update_pre = ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                            ),
                        }
                    ],
                )
                plan_update_post = ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["update_plan_post_messages"],
                                variables={
                                    "task": task,
                                    "tools": self.tools,
                                    "managed_agents": self.managed_agents,
                                    "remaining_steps": (self.max_steps - step),
                                },
                            ),
                        }
                    ],
                )
                input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
                if self.stream_outputs and hasattr(self.model, "generate_stream"):
                    plan_message_content = ""
                    input_tokens, output_tokens = 0, 0
                    with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                        for event in self.model.generate_stream(
                            input_messages,
                            stop_sequences=["<end_plan>"],
                        ):  # type: ignore
                            if event.content is not None:
                                plan_message_content += event.content
                                live.update(Markdown(plan_message_content))
                                if event.token_usage:
                                    output_tokens += event.token_usage.output_tokens
                                    input_tokens = event.token_usage.input_tokens
                            yield event
                else:
                    plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                    plan_message_content = plan_message.content
                    if plan_message.token_usage is not None:
                        input_tokens, output_tokens = (
                            plan_message.token_usage.input_tokens,
                            plan_message.token_usage.output_tokens,
                        )
                plan = textwrap.dedent(
                    f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
                )

            log_headline = "Initial plan" if is_first_step else "Updated plan"
            self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
            planning_step = PlanningStep(
                model_input_messages=input_messages,
                plan=plan,
                model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
                token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                timing=Timing(start_time=start_time, end_time=time.time()),
            )
            if span:
                span.update(outputs={"plan": plan})
            yield planning_step
        except Exception as e:
            if span:
                span.update(outputs={"status": "error", "error": str(e)})
            raise
        finally:
            if span:
                span.end()

    @property
    def logs(self):
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes"""
        ...

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        Parse action from the LLM output

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> ChatMessage:
        """
        Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        span = None
        if self.trace:
            span = self.trace.span(name=f"final_answer_{self.agent_id}")
            span.update(inputs={"task": task})

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            )
        ]
        if images:
            messages[0].content += [{"type": "image", "image": image} for image in images]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            if span:
                span.update(outputs={"answer": chat_message.content})
            return chat_message
        except Exception as e:
            if span:
                span.update(outputs={"status": "error", "error": str(e)})
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[{"type": "text", "text": f"Error in generating final LLM output: {e}"}],
            )
        finally:
            if span:
                span.end()

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""

        app_template = AGENT_GRADIO_APP_TEMPLATE

        app_template = textwrap.dedent(
            """
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """
        ).strip()

        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        # Render the app.py file from Jinja2 template
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "class": self.__class__.__name__,
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "max_runtime": self.max_runtime,
            "verbosity_level": int(self.logger.level),
            "grammar": self.grammar,
            "planning_interval": self.planning_interval,
            "name": self.name,
            "description": self.description,
            "requirements": sorted(requirements),
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = getattr(importlib.import_module("smolagents.models"), model_info["class"])
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(Tool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            managed_agent_class = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(managed_agent_class.from_dict(agent_dict["managed_agents"][managed_agent_name]))
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "max_runtime": agent_dict.get("max_runtime"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "grammar": agent_dict.get("grammar"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        max_tool_threads (`int`, *optional*): Maximum number of threads for parallel tool calls.
            Higher values increase concurrency but resource usage as well.
            Defaults to `ThreadPoolExecutor`'s default.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        agent_id: int,  # Add
        queue_dict: dict,  # Add
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            agent_id=agent_id,  # Pass to parent
            queue_dict=queue_dict,  # Pass to parent
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        self.max_tool_threads = max_tool_threads
        self.trace = None
        if hasattr(langfuse, "trace"):
            try:
                self.trace = langfuse.trace(name=f"Agent_{agent_id}_trace")
            except Exception:
                self.trace = None

    def create_python_executor(self) -> PythonExecutor:
        if self.trace:
            span = self.trace.span(name=f"create_python_executor_{self.agent_id}")
            span.update(inputs={"executor_type": self.executor_type})
        try:
            match self.executor_type:
                case "e2b" | "docker":
                    if self.managed_agents:
                        raise Exception("Managed agents are not yet supported with remote code execution.")
                    if self.executor_type == "e2b":
                        executor = E2BExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
                    else:
                        executor = DockerExecutor(
                            self.additional_authorized_imports, self.logger, **self.executor_kwargs
                        )
                case "local":
                    executor = LocalPythonExecutor(
                        self.additional_authorized_imports,
                        **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
                    )
                case _:
                    raise ValueError(f"Unsupported executor type: {self.executor_type}")
            if self.trace:
                span.update(outputs={"executor": self.executor_type})
            return executor
        except Exception as e:
            if self.trace:
                span.update(outputs={"status": "error", "error": str(e)})
            raise
        finally:
            if self.trace:
                span.end()

    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | ToolOutput]:
        raise NotImplementedError("Decentralized agents process tasks via messages, not steps.")

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ToolCall | ToolOutput`: The tool call or tool output.
        """
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name, arguments=chat_tool_call.function.arguments, id=chat_tool_call.id
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [
                    executor.submit(process_single_tool_call, tool_call) for tool_call in parallel_calls.values()
                ]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [parallel_calls[k] for k in sorted(parallel_calls.keys())]
        memory_step.model_output = memory_step.model_output or ""
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            message = f"Tool call {tool_output.id}: calling '{tool_output.tool_call.name}' with arguments: {tool_output.tool_call.arguments}\n"
            memory_step.model_output += message
            memory_step.observations += tool_output.observation + "\n"
        memory_step.model_output = memory_step.model_output.rstrip("\n")
        memory_step.observations = (
            memory_step.observations.rstrip("\n") if memory_step.observations else memory_step.observations
        )

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        span = None
        if self.trace:
            span = self.trace.span(name=f"execute_tool_call_{self.agent_id}")
            span.update(inputs={"tool_name": tool_name, "arguments": str(arguments)})

        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            if span:
                span.update(outputs={"status": "error", "error": f"Unknown tool {tool_name}"})
                span.end()
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        error_msg = validate_tool_arguments(tool, arguments)
        if error_msg:
            raise AgentToolCallError(error_msg, self.logger)

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                result = tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            else:
                result = tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
            if span:
                span.update(outputs={"result": str(result)})
            return result

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            if span:
                span.update(outputs={"status": "error", "error": str(e)})
            raise AgentToolExecutionError(error_msg, self.logger) from e
        finally:
            if span:
                span.end()


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        executor_type (`Literal["local", "e2b", "docker", "wasm"]`, default `"local"`): Type of code executor.
        executor_kwargs (`dict`, *optional*): Additional arguments to pass to initialize the executor.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        use_structured_outputs_internally (`bool`, default `False`): Whether to use structured generation at each action step: improves performance for many models.

            <Added version="1.17.0"/>
        grammar (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
            <Deprecated version="1.17.0">
            Parameter `grammar` is deprecated and will be removed in version 1.20.
            </Deprecated>
        code_block_tags (`tuple[str, str]` | `Literal["markdown"]`, *optional*): Opening and closing tags for code blocks (regex strings). Pass a custom tuple, or pass 'markdown' to use ("```(?:python|py)", "\\n```"), leave empty to use ("<code>", "</code>").
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        agent_id: int,  # Add
        queue_dict: dict,  # Add
        prompt_templates: PromptTemplates | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor_type: Literal["local", "e2b", "docker", "wasm"] = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        grammar: dict[str, str] | None = None,
        code_block_tags: str | tuple[str, str] | None = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        self._use_structured_outputs_internally = use_structured_outputs_internally
        if use_structured_outputs_internally:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("structured_code_agent.yaml").read_text()
            )
        else:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
            )
        if grammar and use_structured_outputs_internally:
            raise ValueError("You cannot use 'grammar' and 'use_structured_outputs_internally' at the same time.")

        if isinstance(code_block_tags, str) and not code_block_tags == "markdown":
            raise ValueError("Only 'markdown' is supported for a string argument to `code_block_tags`.")
        self.code_block_tags = (
            code_block_tags
            if isinstance(code_block_tags, tuple)
            else ("```python", "```")
            if code_block_tags == "markdown"
            else ("<code>", "</code>")
        )

        super().__init__(
            tools=tools,
            model=model,
            agent_id=agent_id,  # Pass to parent
            queue_dict=queue_dict,  # Pass to parent
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        if executor_type not in {"local", "e2b", "docker", "wasm"}:
            raise ValueError(f"Unsupported executor type: {executor_type}")
        self.executor_type = executor_type
        self.executor_kwargs: dict[str, Any] = executor_kwargs or {}
        self.python_executor = self.create_python_executor()

        self.trace = None
        if hasattr(langfuse, "trace"):
            try:
                self.trace = langfuse.trace(name=f"Agent_{agent_id}_trace")
            except Exception:
                self.trace = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self):
        """Clean up resources used by the agent, such as the remote Python executor."""
        if hasattr(self.python_executor, "cleanup"):
            self.python_executor.cleanup()

    def create_python_executor(self) -> PythonExecutor:
        if self.trace:
            span = self.trace.span(name=f"create_python_executor_{self.agent_id}")
            span.update(inputs={"executor_type": self.executor_type})
        try:
            match self.executor_type:
                case "e2b" | "docker":
                    if self.managed_agents:
                        raise Exception("Managed agents are not yet supported with remote code execution.")
                    if self.executor_type == "e2b":
                        executor = E2BExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
                    else:
                        executor = DockerExecutor(
                            self.additional_authorized_imports, self.logger, **self.executor_kwargs
                        )
                case "local":
                    executor = LocalPythonExecutor(
                        self.additional_authorized_imports,
                        **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
                    )
                case _:
                    raise ValueError(f"Unsupported executor type: {self.executor_type}")
            if self.trace:
                span.update(outputs={"executor": self.executor_type})
            return executor
        except Exception as e:
            if self.trace:
                span.update(outputs={"status": "error", "error": str(e)})
            raise
        finally:
            if self.trace:
                span.end()

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
                "custom_instructions": self.instructions,
                "code_block_opening_tag": self.code_block_tags[0],
                "code_block_closing_tag": self.code_block_tags[1],
            },
        )
        return system_prompt

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | ActionOutput]:
        raise NotImplementedError("Decentralized agents process tasks via messages, not steps.")

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
            "code_block_tags": agent_dict.get("code_block_tags"),
        }
        # Filter out None values
        code_agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if v is not None}
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)
