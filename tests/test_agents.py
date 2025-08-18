# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import json
import os
import re
import tempfile
import uuid
from collections.abc import Generator
from contextlib import nullcontext as does_not_raise
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import (
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
)
from rich.console import Console

from smolagents import EMPTY_PROMPT_TEMPLATES
from smolagents.agent_types import AgentImage, AgentText
from smolagents.agents import (
    AgentError,
    AgentMaxStepsError,
    AgentToolCallError,
    CodeAgent,
    MultiStepAgent,
    RunResult,
    ToolCall,
    ToolCallingAgent,
    ToolOutput,
    populate_template,
)
from smolagents.default_tools import DuckDuckGoSearchTool, FinalAnswerTool, PythonInterpreterTool, VisitWebpageTool
from smolagents.memory import (
    ActionStep,
    CallbackRegistry,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    InferenceClientModel,
    MessageRole,
    Model,
    TransformersModel,
)
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.tools import Tool, tool
from smolagents.utils import (
    BASE_BUILTIN_MODULES,
    AgentExecutionError,
    AgentGenerationError,
    AgentToolExecutionError,
)


@dataclass
class ChoiceDeltaToolCallFunction:
    arguments: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ChoiceDeltaToolCall:
    index: Optional[int] = None
    id: Optional[str] = None
    function: Optional[ChoiceDeltaToolCallFunction] = None
    type: Optional[str] = None


@dataclass
class ChoiceDelta:
    content: Optional[str] = None
    function_call: Optional[str] = None
    refusal: Optional[str] = None
    role: Optional[str] = None
    tool_calls: Optional[list] = None


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


@pytest.fixture
def agent_logger():
    return AgentLogger(
        LogLevel.DEBUG, console=Console(record=True, no_color=True, force_terminal=False, file=io.StringIO())
    )


class FakeToolCallModel(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the python interpreter.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="python_interpreter", arguments={"code": "2*3.6452"}
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will return the final answer.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "7.2904"}),
                    )
                ],
            )


class FakeToolCallModelImage(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="fake_image_generation_tool",
                            arguments={"prompt": "An image of a cat"},
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments="image.png"),
                    )
                ],
            )


class FakeToolCallModelVL(Model):
    def generate(self, messages, tools_to_call_from=None, stop_sequences=None):
        if len(messages) < 3:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="fake_image_understanding_tool",
                            arguments={
                                "prompt": "What is in this image?",
                                "image": "image.png",
                            },
                        ),
                    )
                ],
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments="The image is a cat."),
                    )
                ],
            )


class FakeCodeModel(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = 2**3.6452
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer(7.2904)
</code>
""",
            )


class FakeCodeModelImageGeneration(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should generate an image. special_marker
<code>
image = image_generation_tool()
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer(image)
</code>
""",
            )


class FakeCodeModelPlanning(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "planning_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="llm plan update planning_marker",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        elif "action_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. action_marker
<code>
result = 2**3.6452
</code>
""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="llm plan again",
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )


class FakeCodeModelError(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
print("Flag!")
def error_function():
    raise ValueError("error")

error_function()
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I faced an error in the previous step.
<code>
final_answer("got an error")
</code>
""",
            )


class FakeCodeModelSyntaxError(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")
</code>
""",
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
final_answer("got an error")
</code>
""",
            )


class FakeCodeModelImport(Model):
    def generate(self, messages, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I can answer the question
<code>
import numpy as np
final_answer("got an error")
</code>
""",
        )


class FakeCodeModelFunctionDef(Model):
    def generate(self, messages, stop_sequences=None):
        prompt = str(messages)
        if "special_marker" not in prompt:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: Let's define the function. special_marker
<code>
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
</code>
    """,
            )
        else:  # We're at step 2
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I can now answer the initial question
<code>
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
</code>
""",
            )


class FakeCodeModelSingleStep(Model):
    def generate(self, messages, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
""",
        )


class FakeCodeModelNoReturn(Model):
    def generate(self, messages, stop_sequences=None):
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
<code>
result = python_interpreter(code="2*3.6452")
print(result)
```
""",
        )


class TestAgent:
    def test_fake_toolcalling_agent(self):
        agent = ToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert "7.2904" in output
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert "7.2904" in agent.memory.steps[1].observations
        assert agent.memory.steps[2].model_output == "I will return the final answer."

    def test_toolcalling_agent_handles_image_tool_outputs(self, shared_datadir):
        import PIL.Image

        @tool
        def fake_image_generation_tool(prompt: str) -> PIL.Image.Image:
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """

            import PIL.Image

            return PIL.Image.open(shared_datadir / "000000039769.png")

        agent = ToolCallingAgent(tools=[fake_image_generation_tool], model=FakeToolCallModelImage())
        output = agent.run("Make me an image.")
        assert isinstance(output, AgentImage)
        assert isinstance(agent.state["image.png"], PIL.Image.Image)

    def test_toolcalling_agent_handles_image_inputs(self, shared_datadir):
        import PIL.Image

        image = PIL.Image.open(shared_datadir / "000000039769.png")  # dummy input

        @tool
        def fake_image_understanding_tool(prompt: str, image: PIL.Image.Image) -> str:
            """Tool that creates a caption for an image.

            Args:
                prompt: The prompt
                image: The image
            """
            return "The image is a cat."

        agent = ToolCallingAgent(tools=[fake_image_understanding_tool], model=FakeToolCallModelVL())
        output = agent.run("Caption this image.", images=[image])
        assert output == "The image is a cat."

    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel(), verbosity_level=10)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.memory.steps[0].task == "What is 2 multiplied by 3.6452?"
        assert agent.memory.steps[2].tool_calls == [
            ToolCall(name="python_interpreter", arguments="final_answer(7.2904)", id="call_2")
        ]

    def test_additional_args_added_to_task(self):
        agent = CodeAgent(tools=[], model=FakeCodeModel())
        agent.run(
            "What is 2 multiplied by 3.6452?",
            additional_args={"instruction": "Remember this."},
        )
        assert "Remember this" in agent.task

    def test_reset_conversations(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel())
        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

        output = agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.memory.steps) == 5

        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.memory.steps) == 3

    def test_setup_agent_with_empty_toolbox(self):
        ToolCallingAgent(model=FakeToolCallModel(), tools=[])

    def test_fails_max_steps(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.memory.steps) == 7  # Task step + 5 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),  # use this callable because it never ends
            max_steps=5,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?", max_steps=3)
        assert len(agent.memory.steps) == 5  # Task step + 3 action steps + Final answer
        assert type(agent.memory.steps[-1].error) is AgentMaxStepsError
        assert isinstance(answer, str)

    def test_tool_descriptions_get_baked_in_system_prompt(self):
        tool = PythonInterpreterTool()
        tool.name = "fake_tool_name"
        tool.description = "fake_tool_description"
        agent = CodeAgent(tools=[tool], model=FakeCodeModel())
        agent.run("Empty task")
        assert agent.system_prompt is not None
        assert f"def {tool.name}(" in agent.system_prompt
        assert f'"""{tool.description}' in agent.system_prompt

    def test_module_imports_get_baked_in_system_prompt(self):
        agent = CodeAgent(tools=[], model=FakeCodeModel())
        agent.run("Empty task")
        for module in BASE_BUILTIN_MODULES:
            assert module in agent.system_prompt

    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = CodeAgent(tools=toolset_1, model=FakeCodeModel())
        assert len(agent.tools) == 1  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        with pytest.raises(ValueError) as e:
            agent = CodeAgent(tools=toolset_2, model=FakeCodeModel())
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        with pytest.raises(ValueError) as e:
            agent.name = "python_interpreter"
            agent.description = "empty"
            CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModel(), managed_agents=[agent])
        assert "Each tool or managed_agent should have a unique name!" in str(e)

        # check that python_interpreter base tool does not get added to CodeAgent
        agent = CodeAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 3  # added final_answer tool + search + visit_webpage

        # check that python_interpreter base tool gets added to ToolCallingAgent
        agent = ToolCallingAgent(tools=[], model=FakeCodeModel(), add_base_tools=True)
        assert len(agent.tools) == 4  # added final_answer tool + search + visit_webpage

    def test_function_persistence_across_steps(self):
        agent = CodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            max_steps=2,
            additional_authorized_imports=["numpy"],
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        agent = CodeAgent(tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty")
        assert agent.name == "managed_agent"
        assert agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        managed_agent = CodeAgent(
            tools=[], model=FakeCodeModelFunctionDef(), name="managed_agent", description="Empty"
        )
        manager_agent = CodeAgent(
            tools=[],
            model=FakeCodeModelFunctionDef(),
            managed_agents=[managed_agent],
        )
        assert "You can also give tasks to team members." not in managed_agent.system_prompt
        assert "{{managed_agents_descriptions}}" not in managed_agent.system_prompt
        assert "You can also give tasks to team members." in manager_agent.system_prompt

    def test_replay_shows_logs(self, agent_logger):
        agent = CodeAgent(
            tools=[],
            model=FakeCodeModelImport(),
            verbosity_level=0,
            additional_authorized_imports=["numpy"],
            logger=agent_logger,
        )
        agent.run("Count to 3")

        str_output = agent_logger.console.export_text()

        assert "New run" in str_output
        assert 'final_answer("got' in str_output
        assert "</code>" in str_output

        agent = ToolCallingAgent(tools=[PythonInterpreterTool()], model=FakeToolCallModel(), verbosity_level=0)
        agent.logger = agent_logger

        agent.run("What is 2 multiplied by 3.6452?")
        agent.replay()

        str_output = agent_logger.console.export_text()
        assert "arguments" in str_output

    def test_code_nontrivial_final_answer_works(self):
        class FakeCodeModelFinalAnswer(Model):
            def generate(self, messages, stop_sequences=None):
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""<code>
def nested_answer():
    final_answer("Correct!")

nested_answer()
</code>""",
                )

        agent = CodeAgent(tools=[], model=FakeCodeModelFinalAnswer())

        output = agent.run("Count to 3")
        assert output == "Correct!"

    def test_transformers_toolcalling_agent(self):
        @tool
        def weather_api(location: str, celsius: str = "") -> str:
            """
            Gets the weather in the next days at given location.
            Secretly this tool does not care about the location, it hates the weather everywhere.

            Args:
                location: the location
                celsius: the temperature type
            """
            return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

        model = TransformersModel(
            model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
            max_new_tokens=100,
            device_map="auto",
            do_sample=False,
        )
        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1)
        task = "What is the weather in Paris? "
        agent.run(task)
        assert agent.memory.steps[0].task == task
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        step_memory_dict = agent.memory.get_succinct_steps()[1]
        assert step_memory_dict["model_output_message"]["tool_calls"][0]["function"]["name"] == "weather_api"
        assert step_memory_dict["model_output_message"]["raw"]["completion_kwargs"]["max_new_tokens"] == 100
        assert "model_input_messages" in agent.memory.get_full_steps()[1]
        assert step_memory_dict["token_usage"]["total_tokens"] > 100
        assert step_memory_dict["timing"]["duration"] > 0.1

    def test_final_answer_checks(self):
        error_string = "failed with error"

        def check_always_fails(final_answer, agent_memory):
            assert False, "Error raised in check"

        agent = CodeAgent(model=FakeCodeModel(), tools=[], final_answer_checks=[check_always_fails])
        agent.run("Dummy task.")
        assert error_string in str(agent.write_memory_to_messages())
        assert "Error raised in check" in str(agent.write_memory_to_messages())

        agent = CodeAgent(
            model=FakeCodeModel(),
            tools=[],
            final_answer_checks=[lambda x, y: x == 7.2904],
        )
        output = agent.run("Dummy task.")
        assert output == 7.2904  # Check that output is correct
        assert len([step for step in agent.memory.steps if isinstance(step, ActionStep)]) == 2
        assert error_string not in str(agent.write_memory_to_messages())

    def test_generation_errors_are_raised(self):
        class FakeCodeModel(Model):
            def generate(self, messages, stop_sequences=None):
                assert False, "Generation failed"

        agent = CodeAgent(model=FakeCodeModel(), tools=[])
        with pytest.raises(AgentGenerationError) as e:
            agent.run("Dummy task.")
        assert len(agent.memory.steps) == 2
        assert "Generation failed" in str(e)

    def test_planning_step_with_injected_memory(self):
        """Test that agent properly uses update plan prompts when memory is injected before a run.

        This test verifies:
        1. Planning steps are created with the correct frequency
        2. Injected memory is included in planning context
        3. Messages are properly formatted with expected roles and content
        """
        planning_interval = 1
        max_steps = 4
        task = "Continuous task"
        previous_task = "Previous user request"

        # Create agent with planning capability
        agent = CodeAgent(
            tools=[],
            planning_interval=planning_interval,
            model=FakeCodeModelPlanning(),
            max_steps=max_steps,
        )

        # Inject memory before run to simulate existing conversation history
        previous_step = TaskStep(task=previous_task)
        agent.memory.steps.append(previous_step)

        # Run the agent
        agent.run(task, reset=False)

        # Extract and validate planning steps
        planning_steps = [step for step in agent.memory.steps if isinstance(step, PlanningStep)]
        assert len(planning_steps) > 2, "Expected multiple planning steps to be generated"

        # Verify first planning step incorporates injected memory
        first_planning_step = planning_steps[0]
        input_messages = first_planning_step.model_input_messages

        # Check message structure and content
        assert len(input_messages) == 4, (
            "First planning step should have 4 messages: system-plan-pre-update + memory + task + user-plan-post-update"
        )

        # Verify system message contains current task
        system_message = input_messages[0]
        assert system_message.role == "system", "First message should have system role"
        assert task in system_message.content[0]["text"], f"System message should contain the current task: '{task}'"

        # Verify memory message contains previous task
        memory_message = input_messages[1]
        assert previous_task in memory_message.content[0]["text"], (
            f"Memory message should contain previous task: '{previous_task}'"
        )

        # Verify task message contains current task
        task_message = input_messages[2]
        assert task in task_message.content[0]["text"], f"Task message should contain current task: '{task}'"

        # Verify user message for planning
        user_message = input_messages[3]
        assert user_message.role == "user", "Fourth message should have user role"

        # Verify second planning step has more context from first agent actions
        second_planning_step = planning_steps[1]
        second_messages = second_planning_step.model_input_messages

        # Check that conversation history is growing appropriately
        assert len(second_messages) == 6, "Second planning step should have 6 messages including tool interactions"

        # Verify all conversation elements are present
        conversation_text = "".join([msg.content[0]["text"] for msg in second_messages if hasattr(msg, "content")])
        assert previous_task in conversation_text, "Previous task should be included in the conversation history"
        assert task in conversation_text, "Current task should be included in the conversation history"
        assert "tools" in conversation_text, "Tool interactions should be included in the conversation history"


class CustomFinalAnswerTool(FinalAnswerTool):
    def forward(self, answer) -> str:
        return answer + "CUSTOM"


class MockTool(Tool):
    def __init__(self, name):
        self.name = name
        self.description = "Mock tool description"
        self.inputs = {}
        self.output_type = "string"

    def forward(self):
        return "Mock tool output"


class MockAgent:
    def __init__(self, name, tools, description="Mock agent description"):
        self.name = name
        self.tools = {t.name: t for t in tools}
        self.description = description


class DummyMultiStepAgent(MultiStepAgent):
    def step(self, memory_step: ActionStep) -> Generator[None]:
        yield None

    def initialize_system_prompt(self):
        pass


class FakeLLMModel(Model):
    def __init__(self, give_token_usage: bool = True):
        self.give_token_usage = give_token_usage

    def generate(self, prompt, tools_to_call_from=None, **kwargs):
        if tools_to_call_from is not None:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="I will call the final_answer tool.",
                tool_calls=[
                    ChatMessageToolCall(
                        id="fake_id",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="final_answer", arguments={"answer": "This is the final answer."}
                        ),
                    )
                ],
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )
        else:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""<code>
final_answer('This is the final answer.')
</code>""",
                token_usage=TokenUsage(input_tokens=10, output_tokens=20) if self.give_token_usage else None,
            )


class TestRunResult:
    def test_backward_compatibility(self):
        """Test that RunResult handles deprecated 'messages' parameter correctly."""

        # Test 1: Using new 'steps' parameter (should work without warning)
        result1 = RunResult(
            output="test output",
            state="success",
            steps=[{"type": "test", "content": "step1"}],
            token_usage=None,
            timing=Timing(start_time=0.0, end_time=1.0),
        )
        assert result1.steps == [{"type": "test", "content": "step1"}]

        # Test property access warning
        with pytest.warns(FutureWarning, match="deprecated"):
            messages = result1.messages
        assert messages == [{"type": "test", "content": "step1"}]

        # Test 2: Using deprecated 'messages' parameter (should show deprecation warning)
        with pytest.warns(FutureWarning, match="deprecated"):
            result2 = RunResult(
                output="test output",
                state="success",
                messages=[{"type": "test", "content": "message1"}],
                token_usage=None,
                timing=Timing(start_time=0.0, end_time=1.0),
            )
        assert result2.steps == [{"type": "test", "content": "message1"}]

        # Test 3: Using both 'steps' and 'messages' (should raise ValueError)
        with pytest.raises(ValueError, match="Cannot specify both"):
            RunResult(
                output="test output",
                state="success",
                steps=[{"type": "test", "content": "step1"}],
                messages=[{"type": "test", "content": "message1"}],
                token_usage=None,
                timing=Timing(start_time=0.0, end_time=1.0),
            )

    @pytest.mark.parametrize("agent_class", [CodeAgent, ToolCallingAgent])
    def test_no_token_usage(self, agent_class):
        agent = agent_class(
            tools=[],
            model=FakeLLMModel(give_token_usage=False),
            max_steps=1,
            return_full_result=True,
        )

        result = agent.run("Fake task")

        assert isinstance(result, RunResult)
        assert result.output == "This is the final answer."
        assert result.state == "success"
        assert result.token_usage is None
        assert isinstance(result.messages, list)
        assert result.timing.duration > 0

    @pytest.mark.parametrize(
        "init_return_full_result,run_return_full_result,expect_runresult",
        [
            (True, None, True),
            (False, None, False),
            (True, False, False),
            (False, True, True),
        ],
    )
    def test_full_result(self, init_return_full_result, run_return_full_result, expect_runresult):
        agent = ToolCallingAgent(
            tools=[],
            model=FakeLLMModel(),
            max_steps=1,
            return_full_result=init_return_full_result,
        )
        result = agent.run("Fake task", return_full_result=run_return_full_result)

        if expect_runresult:
            assert isinstance(result, RunResult)
            assert result.output == "This is the final answer."
            assert result.state == "success"
            assert result.token_usage == TokenUsage(input_tokens=10, output_tokens=20)
            assert isinstance(result.messages, list)
            assert result.timing.duration > 0
        else:
            assert isinstance(result, str)


class TestMultiStepAgent:
    def test_instantiation_disables_logging_to_terminal(self):
        fake_model = MagicMock()
        agent = DummyMultiStepAgent(tools=[], model=fake_model)
        assert agent.logger.level == -1, "logging to terminal should be disabled for testing using a fixture"

    def test_instantiation_with_prompt_templates(self, prompt_templates):
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), prompt_templates=prompt_templates)
        assert agent.prompt_templates == prompt_templates
        assert agent.prompt_templates["system_prompt"] == "This is a test system prompt."
        assert "managed_agent" in agent.prompt_templates
        assert agent.prompt_templates["managed_agent"]["task"] == "Task for {{name}}: {{task}}"
        assert agent.prompt_templates["managed_agent"]["report"] == "Report for {{name}}: {{final_answer}}"

    @pytest.mark.parametrize(
        "tools, expected_final_answer_tool",
        [([], FinalAnswerTool), ([CustomFinalAnswerTool()], CustomFinalAnswerTool)],
    )
    def test_instantiation_with_final_answer_tool(self, tools, expected_final_answer_tool):
        agent = DummyMultiStepAgent(tools=tools, model=MagicMock())
        assert "final_answer" in agent.tools
        assert isinstance(agent.tools["final_answer"], expected_final_answer_tool)

    def test_system_prompt_property(self):
        """Test that system_prompt property is read-only and calls initialize_system_prompt."""

        class SimpleAgent(MultiStepAgent):
            def initialize_system_prompt(self) -> str:
                return "Test system prompt"

            def step(self, memory_step: ActionStep) -> Generator[None]:
                yield None

        # Create a simple agent with mocked model
        model = MagicMock()
        agent = SimpleAgent(tools=[], model=model)

        # Test reading the property works and calls initialize_system_prompt
        assert agent.system_prompt == "Test system prompt"

        # Test setting the property raises AttributeError with correct message
        with pytest.raises(
            AttributeError,
            match=re.escape(
                """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
            ),
        ):
            agent.system_prompt = "New system prompt"

        # assert "read-only" in str(exc_info.value)
        # assert "Use 'self.prompt_templates[\"system_prompt\"]' instead" in str(exc_info.value)

    @pytest.mark.parametrize(
        "step_callbacks, expected_registry_state",
        [
            # Case 0: None as input (initializes empty registry)
            (
                None,
                {
                    "MemoryStep": 0,
                    "ActionStep": 1,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },  # Only monitor.update_metrics is registered for ActionStep
            ),
            # Case 1: List of callbacks (registers only for ActionStep: backward compatibility)
            (
                [MagicMock(), MagicMock()],
                {
                    "MemoryStep": 0,
                    "ActionStep": 3,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 2: Dict mapping specific step types to callbacks
            (
                {ActionStep: MagicMock(), PlanningStep: MagicMock()},
                {
                    "MemoryStep": 0,
                    "ActionStep": 2,
                    "PlanningStep": 1,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 3: Dict with list of callbacks for a step type
            (
                {ActionStep: [MagicMock(), MagicMock()]},
                {
                    "MemoryStep": 0,
                    "ActionStep": 3,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
            # Case 4: Dict with mixed single and list callbacks
            (
                {ActionStep: MagicMock(), MemoryStep: [MagicMock(), MagicMock()]},
                {
                    "MemoryStep": 2,
                    "ActionStep": 2,
                    "PlanningStep": 0,
                    "TaskStep": 0,
                    "SystemPromptStep": 0,
                    "FinalAnswerStep": 0,
                },
            ),
        ],
    )
    def test_setup_step_callbacks(self, step_callbacks, expected_registry_state):
        """Test that _setup_step_callbacks correctly sets up the callback registry."""
        # Create a dummy agent
        agent = DummyMultiStepAgent(tools=[], model=MagicMock())
        # Mock the monitor
        agent.monitor = MagicMock()

        # Call the method
        agent._setup_step_callbacks(step_callbacks)

        # Check that step_callbacks is a CallbackRegistry
        assert isinstance(agent.step_callbacks, CallbackRegistry)

        # Count callbacks for each step type
        actual_registry_state = {}
        for step_type in [MemoryStep, ActionStep, PlanningStep, TaskStep, SystemPromptStep, FinalAnswerStep]:
            callbacks = agent.step_callbacks._callbacks.get(step_type, [])
            actual_registry_state[step_type.__name__] = len(callbacks)

        # Verify registry state matches expected
        assert actual_registry_state == expected_registry_state

    def test_finalize_step_callbacks_with_list(self):
        # Create mock callbacks
        callback1 = MagicMock()
        callback2 = MagicMock()

        # Create a test agent with a list of callbacks
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), step_callbacks=[callback1, callback2])

        # Create steps of different types
        action_step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        planning_step = PlanningStep(
            timing=Timing(start_time=1.0),
            model_input_messages=[],
            model_output_message=ChatMessage(role="assistant", content="Test plan"),
            plan="Test planning step",
        )

        # Test with ActionStep
        agent._finalize_step(action_step)

        # Verify all callbacks were called
        callback1.assert_called_once_with(action_step, agent=agent)
        callback2.assert_called_once_with(action_step, agent=agent)

        # Reset mocks
        callback1.reset_mock()
        callback2.reset_mock()

        # Test with PlanningStep
        agent._finalize_step(planning_step)

        # Verify all callbacks were called again with the planning step
        callback1.assert_not_called()
        callback2.assert_not_called()

    def test_finalize_step_callbacks_by_type(self):
        # Create mock callbacks for different step types
        action_step_callback = MagicMock()
        action_step_callback_2 = MagicMock()
        planning_step_callback = MagicMock()
        step_callback = MagicMock()

        # Register callbacks for different step types
        step_callbacks = {
            ActionStep: [action_step_callback, action_step_callback_2],
            PlanningStep: planning_step_callback,
            MemoryStep: step_callback,
        }
        agent = DummyMultiStepAgent(tools=[], model=MagicMock(), step_callbacks=step_callbacks)

        # Create steps of different types
        action_step = ActionStep(step_number=1, timing=Timing(start_time=0.0))
        planning_step = PlanningStep(
            timing=Timing(start_time=1.0),
            model_input_messages=[],
            model_output_message=ChatMessage(role="assistant", content="Test plan"),
            plan="Test planning step",
        )

        # Test with ActionStep
        agent._finalize_step(action_step)

        # Verify correct callbacks were called
        action_step_callback.assert_called_once_with(action_step, agent=agent)
        action_step_callback_2.assert_called_once_with(action_step, agent=agent)
        step_callback.assert_called_once_with(action_step, agent=agent)
        planning_step_callback.assert_not_called()

        # Reset mocks
        action_step_callback.reset_mock()
        action_step_callback_2.reset_mock()
        planning_step_callback.reset_mock()
        step_callback.reset_mock()

        # Test with PlanningStep
        agent._finalize_step(planning_step)

        # Verify correct callbacks were called
        planning_step_callback.assert_called_once_with(planning_step, agent=agent)
        step_callback.assert_called_once_with(planning_step, agent=agent)
        action_step_callback.assert_not_called()
        action_step_callback_2.assert_not_called()

    def test_logs_display_thoughts_even_if_error(self):
        class FakeJsonModelNoCall(Model):
            def generate(self, messages, stop_sequences=None, tools_to_call_from=None):
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""I don't want to call tools today""",
                    tool_calls=None,
                    raw="""I don't want to call tools today""",
                )

        agent_toolcalling = ToolCallingAgent(model=FakeJsonModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_toolcalling.logger.console.capture() as capture:
            agent_toolcalling.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

        class FakeCodeModelNoCall(Model):
            def generate(self, messages, stop_sequences=None):
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="""I don't want to write an action today""",
                )

        agent_code = CodeAgent(model=FakeCodeModelNoCall(), tools=[], max_steps=1, verbosity_level=10)
        with agent_code.logger.console.capture() as capture:
            agent_code.run("Dummy task")
        assert "don't" in capture.get() and "want" in capture.get()

    def test_step_number(self):
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Model output.",
            tool_calls=None,
            raw="Model output.",
            token_usage=None,
        )
        max_steps = 2
        agent = CodeAgent(tools=[], model=fake_model, max_steps=max_steps)
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == 0, "step_number should be initialized to 0"
        agent.run("Test task")
        assert hasattr(agent, "step_number"), "step_number attribute should be defined"
        assert agent.step_number == max_steps + 1, "step_number should be max_steps + 1 after run method is called"

    @pytest.mark.parametrize(
        "step, expected_messages_list",
        [
            (
                1,
                [
                    [
                        ChatMessage(
                            role=MessageRole.USER, content=[{"type": "text", "text": "INITIAL_PLAN_USER_PROMPT"}]
                        ),
                    ],
                ],
            ),
            (
                2,
                [
                    [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=[{"type": "text", "text": "UPDATE_PLAN_SYSTEM_PROMPT"}],
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": "UPDATE_PLAN_USER_PROMPT"}],
                        ),
                    ],
                ],
            ),
        ],
    )
    def test_planning_step(self, step, expected_messages_list):
        fake_model = MagicMock()
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"

        planning_step = list(agent._generate_planning_step(task, is_first_step=(step == 1), step=step))[-1]
        expected_message_texts = {
            "INITIAL_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["initial_plan"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    answer_facts=planning_step.model_output_message.content,
                ),
            ),
            "UPDATE_PLAN_SYSTEM_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_pre_messages"], variables=dict(task=task)
            ),
            "UPDATE_PLAN_USER_PROMPT": populate_template(
                agent.prompt_templates["planning"]["update_plan_post_messages"],
                variables=dict(
                    task=task,
                    tools=agent.tools,
                    managed_agents=agent.managed_agents,
                    facts_update=planning_step.model_output_message.content,
                    remaining_steps=agent.max_steps - step,
                ),
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                expected_message.content[0]["text"] = expected_message_texts[expected_message.content[0]["text"]]
        assert isinstance(planning_step, PlanningStep)
        expected_model_input_messages = expected_messages_list[0]
        model_input_messages = planning_step.model_input_messages
        assert isinstance(model_input_messages, list)
        assert len(model_input_messages) == len(expected_model_input_messages)  # 2
        for message, expected_message in zip(model_input_messages, expected_model_input_messages):
            assert isinstance(message, ChatMessage)
            assert message.role in MessageRole.__members__.values()
            assert message.role == expected_message.role
            assert isinstance(message.content, list)
            for content, expected_content in zip(message.content, expected_message.content):
                assert content == expected_content
        # Test calls to model
        assert len(fake_model.generate.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.generate.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, ChatMessage)
                assert message.role in MessageRole.__members__.values()
                assert message.role == expected_message.role
                assert isinstance(message.content, list)
                for content, expected_content in zip(message.content, expected_message.content):
                    assert content == expected_content

    @pytest.mark.parametrize(
        "images, expected_messages_list",
        [
            (
                None,
                [
                    [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=[{"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"}],
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}],
                        ),
                    ]
                ],
            ),
            (
                ["image1.png"],
                [
                    [
                        ChatMessage(
                            role=MessageRole.SYSTEM,
                            content=[
                                {"type": "text", "text": "FINAL_ANSWER_SYSTEM_PROMPT"},
                                {"type": "image", "image": "image1.png"},
                            ],
                        ),
                        ChatMessage(
                            role=MessageRole.USER,
                            content=[{"type": "text", "text": "FINAL_ANSWER_USER_PROMPT"}],
                        ),
                    ]
                ],
            ),
        ],
    )
    def test_provide_final_answer(self, images, expected_messages_list):
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Final answer.",
            tool_calls=None,
            raw="Final answer.",
            token_usage=None,
        )
        agent = CodeAgent(
            tools=[],
            model=fake_model,
        )
        task = "Test task"
        final_answer = agent.provide_final_answer(task, images=images).content
        expected_message_texts = {
            "FINAL_ANSWER_SYSTEM_PROMPT": agent.prompt_templates["final_answer"]["pre_messages"],
            "FINAL_ANSWER_USER_PROMPT": populate_template(
                agent.prompt_templates["final_answer"]["post_messages"], variables=dict(task=task)
            ),
        }
        for expected_messages in expected_messages_list:
            for expected_message in expected_messages:
                for expected_content in expected_message.content:
                    if "text" in expected_content:
                        expected_content["text"] = expected_message_texts[expected_content["text"]]
        assert final_answer == "Final answer."
        # Test calls to model
        assert len(fake_model.generate.call_args_list) == 1
        for call_args, expected_messages in zip(fake_model.generate.call_args_list, expected_messages_list):
            assert len(call_args.args) == 1
            messages = call_args.args[0]
            assert isinstance(messages, list)
            assert len(messages) == len(expected_messages)
            for message, expected_message in zip(messages, expected_messages):
                assert isinstance(message, ChatMessage)
                assert message.role in MessageRole.__members__.values()
                assert message.role == expected_message.role
                assert isinstance(message.content, list)
                for content, expected_content in zip(message.content, expected_message.content):
                    assert content == expected_content

    def test_interrupt(self):
        fake_model = MagicMock()
        fake_model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Model output.",
            tool_calls=None,
            raw="Model output.",
            token_usage=None,
        )

        def interrupt_callback(memory_step, agent):
            agent.interrupt()

        agent = CodeAgent(
            tools=[],
            model=fake_model,
            step_callbacks=[interrupt_callback],
        )
        with pytest.raises(AgentError) as e:
            agent.run("Test task")
        assert "Agent interrupted" in str(e)

    @pytest.mark.parametrize(
        "tools, managed_agents, name, expectation",
        [
            # Valid case: no duplicates
            (
                [MockTool("tool1"), MockTool("tool2")],
                [MockAgent("agent1", [MockTool("tool3")])],
                "test_agent",
                does_not_raise(),
            ),
            # Invalid case: duplicate tool names
            ([MockTool("tool1"), MockTool("tool1")], [], "test_agent", pytest.raises(ValueError)),
            # Invalid case: tool name same as managed agent name
            (
                [MockTool("tool1")],
                [MockAgent("tool1", [MockTool("final_answer")])],
                "test_agent",
                pytest.raises(ValueError),
            ),
            # Valid case: tool name same as managed agent's tool name
            ([MockTool("tool1")], [MockAgent("agent1", [MockTool("tool1")])], "test_agent", does_not_raise()),
            # Invalid case: duplicate managed agent name and managed agent tool name
            ([MockTool("tool1")], [], "tool1", pytest.raises(ValueError)),
            # Valid case: duplicate tool names across managed agents
            (
                [MockTool("tool1")],
                [
                    MockAgent("agent1", [MockTool("tool2"), MockTool("final_answer")]),
                    MockAgent("agent2", [MockTool("tool2"), MockTool("final_answer")]),
                ],
                "test_agent",
                does_not_raise(),
            ),
        ],
    )
    def test_validate_tools_and_managed_agents(self, tools, managed_agents, name, expectation):
        fake_model = MagicMock()
        with expectation:
            DummyMultiStepAgent(
                tools=tools,
                model=fake_model,
                name=name,
                managed_agents=managed_agents,
            )

    def test_from_dict(self):
        # Create a test agent dictionary
        agent_dict = {
            "model": {"class": "TransformersModel", "data": {"model_id": "test/model"}},
            "tools": [
                {
                    "name": "valid_tool_function",
                    "code": 'from smolagents import Tool\nfrom typing import Any, Optional\n\nclass SimpleTool(Tool):\n    name = "valid_tool_function"\n    description = "A valid tool function."\n    inputs = {"input":{"type":"string","description":"Input string."}}\n    output_type = "string"\n\n    def forward(self, input: str) -> str:\n        """A valid tool function.\n\n        Args:\n            input (str): Input string.\n        """\n        return input.upper()',
                    "requirements": {"smolagents"},
                }
            ],
            "managed_agents": {},
            "prompt_templates": EMPTY_PROMPT_TEMPLATES,
            "max_steps": 15,
            "verbosity_level": 2,
            "planning_interval": 3,
            "name": "test_agent",
            "description": "Test agent description",
        }

        # Call from_dict
        with patch("smolagents.models.TransformersModel") as mock_model_class:
            mock_model_instance = mock_model_class.from_dict.return_value
            agent = DummyMultiStepAgent.from_dict(agent_dict)

        # Verify the agent was created correctly
        assert agent.model == mock_model_instance
        assert mock_model_class.from_dict.call_args.args[0] == {"model_id": "test/model"}
        assert agent.max_steps == 15
        assert agent.logger.level == 2
        assert agent.planning_interval == 3
        assert agent.name == "test_agent"
        assert agent.description == "Test agent description"
        # Verify the tool was created correctly
        assert sorted(agent.tools.keys()) == ["final_answer", "valid_tool_function"]
        assert agent.tools["valid_tool_function"].name == "valid_tool_function"
        assert agent.tools["valid_tool_function"].description == "A valid tool function."
        assert agent.tools["valid_tool_function"].inputs == {
            "input": {"type": "string", "description": "Input string."}
        }
        assert agent.tools["valid_tool_function"]("test") == "TEST"

        # Test overriding with kwargs
        with patch("smolagents.models.TransformersModel") as mock_model_class:
            agent = DummyMultiStepAgent.from_dict(agent_dict, max_steps=30)
        assert agent.max_steps == 30

    def test_multiagent_to_dict_from_dict_roundtrip(self):
        """Test that to_dict() and from_dict() work correctly for agents with managed agents."""
        # Create a managed agent
        managed_agent = CodeAgent(
            tools=[], model=MagicMock(), name="managed_agent", description="A managed agent for testing", max_steps=5
        )

        # Create a main agent with the managed agent
        main_agent = ToolCallingAgent(
            tools=[],
            managed_agents=[managed_agent],
            model=MagicMock(),
            name="main_agent",
            description="Main agent with managed agents",
            max_steps=10,
        )

        # Convert to dict
        agent_dict = main_agent.to_dict()

        # Verify managed_agents structure in dict
        assert "managed_agents" in agent_dict
        assert isinstance(agent_dict["managed_agents"], list)
        assert len(agent_dict["managed_agents"]) == 1

        managed_agent_dict = agent_dict["managed_agents"][0]
        assert managed_agent_dict["name"] == "managed_agent"
        assert managed_agent_dict["class"] == "CodeAgent"
        assert managed_agent_dict["description"] == "A managed agent for testing"
        assert managed_agent_dict["max_steps"] == 5

        # Test round-trip: from_dict should recreate the agent
        # Mock the model classes directly instead of patching smolagents.models.MagicMock
        with patch("smolagents.agents.importlib.import_module") as mock_import:
            # Mock the models module
            mock_models_module = MagicMock()
            mock_model_class = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_class.from_dict.return_value = mock_model_instance
            mock_models_module.MagicMock = mock_model_class

            # Mock the agents module
            mock_agents_module = MagicMock()
            mock_agents_module.CodeAgent = CodeAgent
            mock_agents_module.ToolCallingAgent = ToolCallingAgent

            def side_effect(module_name):
                if module_name == "smolagents.models":
                    return mock_models_module
                elif module_name == "smolagents.agents":
                    return mock_agents_module
                return MagicMock()

            mock_import.side_effect = side_effect

            recreated_agent = ToolCallingAgent.from_dict(agent_dict)

        # Verify the recreated agent has the same structure
        assert recreated_agent.name == "main_agent"
        assert recreated_agent.description == "Main agent with managed agents"
        assert recreated_agent.max_steps == 10
        assert len(recreated_agent.managed_agents) == 1

        recreated_managed_agent = list(recreated_agent.managed_agents.values())[0]
        assert recreated_managed_agent.name == "managed_agent"
        assert recreated_managed_agent.description == "A managed agent for testing"
        assert recreated_managed_agent.max_steps == 5


class TestToolCallingAgent:
    def test_toolcalling_agent_instructions(self):
        agent = ToolCallingAgent(tools=[], model=MagicMock(), instructions="Test instructions")
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

    def test_toolcalling_agent_passes_both_tools_and_managed_agents(self, test_tool):
        """Test that both tools and managed agents are passed to the model."""
        managed_agent = MagicMock()
        managed_agent.name = "managed_agent"
        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "test_value"}),
                )
            ],
        )
        agent = ToolCallingAgent(tools=[test_tool], managed_agents=[managed_agent], model=model)
        # Run the agent one step to trigger the model call
        next(agent.run("Test task", stream=True))
        # Check that the model was called with both tools and managed agents:
        # - Get all tool_to_call_from names passed to the model
        tools_to_call_from_names = [tool.name for tool in model.generate.call_args.kwargs["tools_to_call_from"]]
        # - Verify both regular tools and managed agents are included
        assert "test_tool" in tools_to_call_from_names  # The regular tool
        assert "managed_agent" in tools_to_call_from_names  # The managed agent
        assert "final_answer" in tools_to_call_from_names  # The final_answer tool (added by default)

    @patch("huggingface_hub.InferenceClient")
    def test_toolcalling_agent_api(self, mock_inference_client):
        mock_client = mock_inference_client.return_value
        mock_response = mock_client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": "weather_api", "arguments": {"location": "Paris", "date": "today"}}',
        )
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        model = InferenceClientModel(model_id="test-model")

        from smolagents import tool

        @tool
        def weather_api(location: str, date: str) -> str:
            """
            Gets the weather in the next days at given location.
            Args:
                location: the location
                date: the date
            """
            return f"The weather in {location} on date:{date} is sunny."

        agent = ToolCallingAgent(model=model, tools=[weather_api], max_steps=1)
        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."

        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatCompletionOutputToolCall(
                    function=ChatCompletionOutputFunctionDefinition(
                        name="weather_api", arguments='{"location": "Paris", "date": "today"}'
                    ),
                    id="call_0",
                    type="function",
                )
            ],
        )

        agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls[0].name == "weather_api"
        assert agent.memory.steps[1].tool_calls[0].arguments == {"location": "Paris", "date": "today"}
        assert agent.memory.steps[1].observations == "The weather in Paris on date:today is sunny."

    @patch("openai.OpenAI")
    def test_toolcalling_agent_stream_logs_multiple_tool_calls_observations(self, mock_openai_client, test_tool):
        """Test that ToolCallingAgent with stream_outputs=True logs the observations of all tool calls when multiple are called."""
        mock_client = mock_openai_client.return_value
        from smolagents import OpenAIServerModel

        # Mock streaming response with multiple tool calls
        mock_deltas = [
            ChoiceDelta(role=MessageRole.ASSISTANT),
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id="call_1",
                        function=ChoiceDeltaToolCallFunction(name="test_tool"),
                        type="function",
                    )
                ]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='{"in'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='put"'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments=': "out'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments="put1"))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=0, function=ChoiceDeltaToolCallFunction(arguments='"}'))]
            ),
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=1,
                        id="call_2",
                        function=ChoiceDeltaToolCallFunction(name="test_tool"),
                        type="function",
                    )
                ]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='{"in'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='put"'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments=': "out'))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments="put2"))]
            ),
            ChoiceDelta(
                tool_calls=[ChoiceDeltaToolCall(index=1, function=ChoiceDeltaToolCallFunction(arguments='"}'))]
            ),
        ]

        class MockChoice:
            def __init__(self, delta):
                self.delta = delta

        class MockChunk:
            def __init__(self, delta):
                self.choices = [MockChoice(delta)]
                self.usage = None

        mock_client.chat.completions.create.return_value = (MockChunk(delta) for delta in mock_deltas)

        # Mock usage for non-streaming fallback
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20

        model = OpenAIServerModel(model_id="fakemodel")

        agent = ToolCallingAgent(model=model, tools=[test_tool], max_steps=1, stream_outputs=True)
        agent.run("Dummy task")
        assert agent.memory.steps[1].model_output_message.tool_calls[0].function.name == "test_tool"
        assert agent.memory.steps[1].model_output_message.tool_calls[1].function.name == "test_tool"
        assert agent.memory.steps[1].observations == "Processed: output1\nProcessed: output2"

    @patch("openai.OpenAI")
    def test_toolcalling_agent_final_answer_cannot_be_called_with_parallel_tool_calls(
        self, mock_openai_client, test_tool
    ):
        """Test that ToolCallingAgent with stream_outputs=True returns the all tool calls when multiple are called."""
        mock_client = mock_openai_client.return_value

        from smolagents import OpenAIServerModel

        class ExtendedChatMessage(ChatMessage):
            def __init__(self, *args, usage, **kwargs):
                super().__init__(*args, **kwargs)

            def model_dump(self, include=None):
                return super().model_dump_json()

        class MockChoice:
            def __init__(self, chat_message):
                self.message = chat_message

        class MockChatCompletion:
            def __init__(self, chat_message):
                self.choices = [MockChoice(chat_message)]
                self.usage = MockTokenUsage(prompt_tokens=10, completion_tokens=20)

        class MockTokenUsage:
            def __init__(self, prompt_tokens, completion_tokens):
                self.prompt_tokens = prompt_tokens
                self.completion_tokens = completion_tokens

        from dataclasses import asdict

        class ExtendedChatCompletionOutputMessage(ChatCompletionOutputMessage):
            def __init__(self, *args, usage, **kwargs):
                super().__init__(*args, **kwargs)
                self.usage = usage

            def model_dump(self, include=None):
                print("TOOL CALLS", self.tool_calls)
                return {
                    "role": self.role,
                    "content": self.content,
                    "tool_calls": [asdict(tc) for tc in self.tool_calls],
                }

        mock_client.chat.completions.create.return_value = MockChatCompletion(
            ExtendedChatCompletionOutputMessage(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ChatMessageToolCall(
                        id="call_0",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "out1"}),
                    ),
                    ChatMessageToolCall(
                        id="1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="final_answer", arguments={"answer": "out1"}),
                    ),
                ],
                usage=MockTokenUsage(prompt_tokens=10, completion_tokens=20),
            )
        )

        model = OpenAIServerModel(model_id="fakemodel")

        agent = ToolCallingAgent(model=model, tools=[test_tool], max_steps=1)
        agent.run("Dummy task")
        assert agent.memory.steps[1].error is not None
        assert (
            "do not perform any other tool calls than the final answer tool call!"
            in agent.memory.steps[1].error.message
        )

    @patch("huggingface_hub.InferenceClient")
    def test_toolcalling_agent_api_misformatted_output(self, mock_inference_client):
        """Test that even misformatted json blobs don't interrupt the run for a ToolCallingAgent."""
        mock_client = mock_inference_client.return_value
        mock_response = mock_client.chat_completion.return_value
        mock_response.choices[0].message = ChatCompletionOutputMessage(
            role=MessageRole.ASSISTANT,
            content='{"name": weather_api", "arguments": {"location": "Paris", "date": "today"}}',
        )

        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        model = InferenceClientModel(model_id="test-model")

        logger = AgentLogger(console=Console(markup=False, no_color=True))

        agent = ToolCallingAgent(model=model, tools=[], max_steps=2, verbosity_level=1, logger=logger)
        with agent.logger.console.capture() as capture:
            agent.run("What's the weather in Paris?")
        assert agent.memory.steps[0].task == "What's the weather in Paris?"
        assert agent.memory.steps[1].tool_calls is None
        assert "The JSON blob you used is invalid" in agent.memory.steps[1].error.message
        assert "Error while parsing" in capture.get()
        assert len(agent.memory.steps) == 4

    @pytest.mark.skip(
        reason="Test is not properly implemented (GH-1255) because fake_tools should have the same name. "
        "Additionally, it uses CodeAgent instead of ToolCallingAgent (GH-1409)"
    )
    def test_change_tools_after_init(self):
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            """Fake tool"""
            return "2"

        class FakeCodeModel(Model):
            def generate(self, messages, stop_sequences=None):
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(fake_tool_1())\n</code>")

        agent = CodeAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = agent.run("Fake task.")
        assert answer == "2CUSTOM"

    def test_custom_final_answer_with_custom_inputs(self, test_tool):
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                return answer1 + " and " + answer2

        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[
                ChatMessageToolCall(
                    id="call_0",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="final_answer", arguments={"answer1": "1", "answer2": "2"}
                    ),
                ),
            ],
        )
        agent = ToolCallingAgent(tools=[test_tool, CustomFinalAnswerToolWithCustomInputs()], model=model)
        answer = agent.run("Fake task.")
        assert answer == "1 and 2"
        assert agent.memory.steps[-1].model_output_message.tool_calls[0].function.name == "final_answer"

    @pytest.mark.parametrize(
        "test_case",
        [
            # Case 0: Single valid tool call
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "test_value"}),
                    )
                ],
                "expected_observations": "Processed: test_value",
                "expected_final_outputs": ["Processed: test_value"],
                "expected_error": None,
            },
            # Case 1: Multiple tool calls
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value1"}),
                    ),
                    ChatMessageToolCall(
                        id="call_2",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "value2"}),
                    ),
                ],
                "expected_observations": "Processed: value1\nProcessed: value2",
                "expected_final_outputs": ["Processed: value1", "Processed: value2"],
                "expected_error": None,
            },
            # Case 2: Invalid tool name
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="nonexistent_tool", arguments={"input": "test"}),
                    )
                ],
                "expected_error": AgentToolExecutionError,
            },
            # Case 3: Tool execution error
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"input": "error"}),
                    )
                ],
                "expected_error": AgentToolExecutionError,
            },
            # Case 4: Empty tool calls list
            {
                "tool_calls": [],
                "expected_observations": "",
                "expected_final_outputs": [],
                "expected_error": None,
            },
            # Case 5: Final answer call
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="final_answer", arguments={"answer": "This is the final answer"}
                        ),
                    )
                ],
                "expected_observations": "This is the final answer",
                "expected_final_outputs": ["This is the final answer"],
                "expected_error": None,
            },
            # Case 6: Invalid arguments
            {
                "tool_calls": [
                    ChatMessageToolCall(
                        id="call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(name="test_tool", arguments={"wrong_param": "value"}),
                    )
                ],
                "expected_error": AgentToolCallError,
            },
        ],
    )
    def test_process_tool_calls(self, test_case, test_tool):
        # Create a ToolCallingAgent instance with the test tool
        agent = ToolCallingAgent(tools=[test_tool], model=MagicMock())
        # Create chat message with the specified tool calls for process_tool_calls
        chat_message = ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=test_case["tool_calls"])
        # Create a memory step for process_tool_calls
        memory_step = ActionStep(step_number=10, timing="mock_timing", model_output="")

        # Process tool calls
        if test_case["expected_error"]:
            with pytest.raises(test_case["expected_error"]):
                list(agent.process_tool_calls(chat_message, memory_step))
        else:
            final_outputs = list(agent.process_tool_calls(chat_message, memory_step))
            assert memory_step.model_output == ""
            assert memory_step.observations == test_case["expected_observations"]
            assert [
                final_output.output for final_output in final_outputs if isinstance(final_output, ToolOutput)
            ] == test_case["expected_final_outputs"]
            # Verify memory step tool calls were updated correctly
            if test_case["tool_calls"]:
                assert memory_step.tool_calls == [
                    ToolCall(name=tool_call.function.name, arguments=tool_call.function.arguments, id=tool_call.id)
                    for tool_call in test_case["tool_calls"]
                ]


class TestCodeAgent:
    def test_code_agent_instructions(self):
        agent = CodeAgent(tools=[], model=MagicMock(), instructions="Test instructions")
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

        agent = CodeAgent(
            tools=[], model=MagicMock(), instructions="Test instructions", use_structured_outputs_internally=True
        )
        assert agent.instructions == "Test instructions"
        assert "Test instructions" in agent.system_prompt

    @pytest.mark.parametrize("provide_run_summary", [False, True])
    def test_call_with_provide_run_summary(self, provide_run_summary):
        agent = CodeAgent(tools=[], model=MagicMock(), provide_run_summary=provide_run_summary)
        assert agent.provide_run_summary is provide_run_summary
        agent.name = "test_agent"
        agent.run = MagicMock(return_value="Test output")
        agent.write_memory_to_messages = MagicMock(
            return_value=[ChatMessage(role=MessageRole.ASSISTANT, content="Test summary")]
        )

        result = agent("Test request")
        expected_summary = "Here is the final answer from your managed agent 'test_agent':\nTest output"
        if provide_run_summary:
            expected_summary += (
                "\n\nFor more detail, find below a summary of this agent's work:\n"
                "<summary_of_work>\n\nTest summary\n---\n</summary_of_work>"
            )
        assert result == expected_summary

    def test_code_agent_image_output(self):
        from PIL import Image

        from smolagents import tool

        @tool
        def image_generation_tool():
            """Generate an image"""
            return Image.new("RGB", (100, 100), color="red")

        agent = CodeAgent(tools=[image_generation_tool], model=FakeCodeModelImageGeneration(), verbosity_level=1)
        output = agent.run("Make me an image from the latest trend on google trends.")
        assert isinstance(output, Image.Image)

    def test_errors_logging(self):
        class FakeCodeModel(Model):
            def generate(self, messages, stop_sequences=None):
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nsecret=3;['1', '2'][secret]\n</code>")

        agent = CodeAgent(tools=[], model=FakeCodeModel(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Test request")
        assert "secret\\\\" in repr(capture.get())

    def test_missing_import_triggers_advice_in_error_log(self):
        # Set explicit verbosity level to 1 to override the default verbosity level of -1 set in CI fixture
        agent = CodeAgent(tools=[], model=FakeCodeModelImport(), verbosity_level=1)

        with agent.logger.console.capture() as capture:
            agent.run("Count to 3")
        str_output = capture.get()
        assert "`additional_authorized_imports`" in str_output.replace("\n", "")

    def test_errors_show_offending_line_and_error(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Code execution failed at line 'error_function()'" in str(agent.memory.steps[1].error)
        assert "ValueError" in str(agent.memory.steps)

    def test_error_saves_previous_print_outputs(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelError())
        agent.run("What is 2 multiplied by 3.6452?")
        assert "Flag!" in str(agent.memory.steps[1].observations)

    def test_syntax_error_show_offending_lines(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=FakeCodeModelSyntaxError())
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert '    print("Failing due to unexpected indent")' in str(agent.memory.steps)
        assert isinstance(agent.memory.steps[-2], ActionStep)
        assert agent.memory.steps[-2].code_action == dedent("""a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")""")

    def test_end_code_appending(self):
        # Checking original output message
        orig_output = FakeCodeModelNoReturn().generate([])
        assert not orig_output.content.endswith("<end_code>")

        # Checking the step output
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=FakeCodeModelNoReturn(),
            max_steps=1,
        )
        answer = agent.run("What is 2 multiplied by 3.6452?")
        assert answer

        memory_steps = agent.memory.steps
        actions_steps = [s for s in memory_steps if isinstance(s, ActionStep)]

        outputs = [s.model_output for s in actions_steps if s.model_output]
        assert outputs
        assert all(o.endswith("</code>") for o in outputs)

        messages = [s.model_output_message for s in actions_steps if s.model_output_message]
        assert messages
        assert all(m.content.endswith("</code>") for m in messages)

    @pytest.mark.skip(
        reason="Test is not properly implemented (GH-1255) because fake_tools should have the same name. "
    )
    def test_change_tools_after_init(self):
        from smolagents import tool

        @tool
        def fake_tool_1() -> str:
            """Fake tool"""
            return "1"

        @tool
        def fake_tool_2() -> str:
            """Fake tool"""
            return "2"

        class FakeCodeModel(Model):
            def generate(self, messages, stop_sequences=None):
                return ChatMessage(role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(fake_tool_1())\n</code>")

        agent = CodeAgent(tools=[fake_tool_1], model=FakeCodeModel())

        agent.tools["final_answer"] = CustomFinalAnswerTool()
        agent.tools["fake_tool_1"] = fake_tool_2

        answer = agent.run("Fake task.")
        assert answer == "2CUSTOM"

    def test_local_python_executor_with_custom_functions(self):
        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="",
            tool_calls=None,
            raw="",
            token_usage=None,
        )
        agent = CodeAgent(tools=[], model=model, executor_kwargs={"additional_functions": {"open": open}})
        agent.run("Test run")
        assert "open" in agent.python_executor.static_tools

    @pytest.mark.parametrize("agent_dict_version", ["v1.9", "v1.10", "v1.20"])
    def test_from_folder(self, agent_dict_version, get_agent_dict):
        agent_dict = get_agent_dict(agent_dict_version)
        with (
            patch("smolagents.agents.Path") as mock_path,
            patch("smolagents.models.InferenceClientModel") as mock_model,
        ):
            import json

            mock_path.return_value.__truediv__.return_value.read_text.return_value = json.dumps(agent_dict)
            mock_model.from_dict.return_value.model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
            agent = CodeAgent.from_folder("ignored_dummy_folder")
        assert isinstance(agent, CodeAgent)
        assert agent.name == "test_agent"
        assert agent.description == "dummy description"
        assert agent.max_steps == 10
        assert agent.planning_interval == 2
        assert agent.additional_authorized_imports == ["pandas"]
        assert "pandas" in agent.authorized_imports
        assert agent.executor_type == "local"
        assert agent.executor_kwargs == {}
        assert agent.max_print_outputs_length is None
        assert agent.managed_agents == {}
        assert set(agent.tools.keys()) == {"final_answer"}
        assert agent.model == mock_model.from_dict.return_value
        assert mock_model.from_dict.call_args.args[0]["model_id"] == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.model.model_id == "Qwen/Qwen2.5-Coder-32B-Instruct"
        assert agent.logger.level == 2
        assert agent.prompt_templates["system_prompt"] == "dummy system prompt"

    def test_from_dict(self):
        # Create a test agent dictionary
        agent_dict = {
            "model": {"class": "InferenceClientModel", "data": {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct"}},
            "tools": [
                {
                    "name": "valid_tool_function",
                    "code": 'from smolagents import Tool\nfrom typing import Any, Optional\n\nclass SimpleTool(Tool):\n    name = "valid_tool_function"\n    description = "A valid tool function."\n    inputs = {"input":{"type":"string","description":"Input string."}}\n    output_type = "string"\n\n    def forward(self, input: str) -> str:\n        """A valid tool function.\n\n        Args:\n            input (str): Input string.\n        """\n        return input.upper()',
                    "requirements": {"smolagents"},
                }
            ],
            "managed_agents": {},
            "prompt_templates": EMPTY_PROMPT_TEMPLATES,
            "max_steps": 15,
            "verbosity_level": 2,
            "use_structured_output": False,
            "planning_interval": 3,
            "name": "test_code_agent",
            "description": "Test code agent description",
            "authorized_imports": ["pandas", "numpy"],
            "executor_type": "local",
            "executor_kwargs": {"max_print_outputs_length": 10_000},
            "max_print_outputs_length": 1000,
        }

        # Call from_dict
        with patch("smolagents.models.InferenceClientModel") as mock_model_class:
            mock_model_instance = mock_model_class.from_dict.return_value
            agent = CodeAgent.from_dict(agent_dict)

        # Verify the agent was created correctly with CodeAgent-specific parameters
        assert agent.model == mock_model_instance
        assert agent.additional_authorized_imports == ["pandas", "numpy"]
        assert agent.executor_type == "local"
        assert agent.executor_kwargs == {"max_print_outputs_length": 10_000}
        assert agent.max_print_outputs_length == 1000

        # Test with missing optional parameters
        minimal_agent_dict = {
            "model": {"class": "InferenceClientModel", "data": {"model_id": "Qwen/Qwen2.5-Coder-32B-Instruct"}},
            "tools": [],
            "managed_agents": {},
        }

        with patch("smolagents.models.InferenceClientModel"):
            agent = CodeAgent.from_dict(minimal_agent_dict)
        # Verify defaults are used
        assert agent.max_steps == 20  # default from MultiStepAgent.__init__

        # Test overriding with kwargs
        with patch("smolagents.models.InferenceClientModel"):
            agent = CodeAgent.from_dict(
                agent_dict,
                additional_authorized_imports=["requests"],
                executor_kwargs={"max_print_outputs_length": 5_000},
            )
        assert agent.additional_authorized_imports == ["requests"]
        assert agent.executor_kwargs == {"max_print_outputs_length": 5_000}

    def test_custom_final_answer_with_custom_inputs(self):
        class CustomFinalAnswerToolWithCustomInputs(FinalAnswerTool):
            inputs = {
                "answer1": {"type": "string", "description": "First part of the answer."},
                "answer2": {"type": "string", "description": "Second part of the answer."},
            }

            def forward(self, answer1: str, answer2: str) -> str:
                return answer1 + "CUSTOM" + answer2

        model = MagicMock()
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT, content="<code>\nfinal_answer(answer1='1', answer2='2')\n</code>"
        )
        agent = CodeAgent(tools=[CustomFinalAnswerToolWithCustomInputs()], model=model)
        answer = agent.run("Fake task.")
        assert answer == "1CUSTOM2"

    def test_use_structured_outputs_internally(self):
        expected_code = "print('Hello, world!')"
        model = MagicMock()
        # mock structured output generation
        model.generate.return_value = ChatMessage(
            role=MessageRole.ASSISTANT,
            content=json.dumps({"thought": "LLM-generated thought", "code": expected_code}),
        )
        agent = CodeAgent(
            tools=[], model=model, use_structured_outputs_internally=True
        )  # Use structured outputs internally
        tool_call: ToolCall = next(
            agent._step_stream(ActionStep(step_number=1, timing="mock_timing", model_output=""))
        )
        assert tool_call.arguments == expected_code


class TestMultiAgents:
    def test_multiagents_save(self, tmp_path):
        model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct", max_tokens=2096, temperature=0.5)

        web_agent = ToolCallingAgent(
            model=model,
            tools=[DuckDuckGoSearchTool(max_results=2), VisitWebpageTool()],
            name="web_agent",
            description="does web searches",
        )
        code_agent = CodeAgent(model=model, tools=[], name="useless", description="does nothing in particular")

        agent = CodeAgent(
            model=model,
            tools=[],
            additional_authorized_imports=["pandas", "datetime"],
            managed_agents=[web_agent, code_agent],
            max_print_outputs_length=1000,
            executor_type="local",
            executor_kwargs={"max_print_outputs_length": 10_000},
        )
        agent.save(tmp_path)

        expected_structure = {
            "managed_agents": {
                "useless": {"tools": {"files": ["final_answer.py"]}, "files": ["agent.json", "prompts.yaml"]},
                "web_agent": {
                    "tools": {"files": ["final_answer.py", "visit_webpage.py", "web_search.py"]},
                    "files": ["agent.json", "prompts.yaml"],
                },
            },
            "tools": {"files": ["final_answer.py"]},
            "files": ["app.py", "requirements.txt", "agent.json", "prompts.yaml"],
        }

        def verify_structure(current_path: Path, structure: dict):
            for dir_name, contents in structure.items():
                if dir_name != "files":
                    # For directories, verify they exist and recurse into them
                    dir_path = current_path / dir_name
                    assert dir_path.exists(), f"Directory {dir_path} does not exist"
                    assert dir_path.is_dir(), f"{dir_path} is not a directory"
                    verify_structure(dir_path, contents)
                else:
                    # For files, verify each exists in the current path
                    for file_name in contents:
                        file_path = current_path / file_name
                        assert file_path.exists(), f"File {file_path} does not exist"
                        assert file_path.is_file(), f"{file_path} is not a file"

        verify_structure(tmp_path, expected_structure)

        # Test that re-loaded agents work as expected.
        agent2 = CodeAgent.from_folder(tmp_path, planning_interval=5)
        assert agent2.planning_interval == 5  # Check that kwargs are used
        assert set(agent2.authorized_imports) == set(["pandas", "datetime"] + BASE_BUILTIN_MODULES)
        assert agent2.max_print_outputs_length == 1000
        assert agent2.executor_type == "local"
        assert agent2.executor_kwargs == {"max_print_outputs_length": 10_000}
        assert (
            agent2.managed_agents["web_agent"].tools["web_search"].max_results == 10
        )  # For now tool init parameters are forgotten
        assert agent2.model.kwargs["temperature"] == pytest.approx(0.5)

    def test_multiagents(self):
        class FakeModelMultiagentsManagerAgent(Model):
            model_id = "fake_model"

            def generate(
                self,
                messages,
                stop_sequences=None,
                tools_to_call_from=None,
            ):
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="""
Thought: Let's call our search agent.
<code>
result = search_agent("Who is the current US president?")
</code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role=MessageRole.ASSISTANT,
                            content="""
Thought: Let's return the report.
<code>
final_answer("Final report.")
</code>
""",
                        )

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent(Model):
            model_id = "fake_model"

            def generate(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
            ):
                return ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content="Here is the secret content: FLAG1",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = ToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
            verbosity_level=2,
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        with web_agent.logger.console.capture() as capture:
            report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."
        assert "FLAG1" in capture.get()  # Check that managed agent's output is properly logged

        # Test that visualization works
        with manager_toolcalling_agent.logger.console.capture() as capture:
            manager_toolcalling_agent.visualize()
        assert "├──" in capture.get()


@pytest.fixture
def prompt_templates():
    return {
        "system_prompt": "This is a test system prompt.",
        "managed_agent": {"task": "Task for {{name}}: {{task}}", "report": "Report for {{name}}: {{final_answer}}"},
        "planning": {
            "initial_plan": "The plan.",
            "update_plan_pre_messages": "custom",
            "update_plan_post_messages": "custom",
        },
        "final_answer": {"pre_messages": "custom", "post_messages": "custom"},
    }


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"arg": "bar"},
        {None: None},
        [1, 2, 3],
    ],
)
def test_tool_calling_agents_raises_tool_call_error_being_invoked_with_wrong_arguments(arguments):
    @tool
    def _sample_tool(prompt: str) -> str:
        """Tool that returns same string
        Args:
            prompt: The string to return
        Returns:
            The same string
        """

        return prompt

    agent = ToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentToolCallError):
        agent.execute_tool_call(_sample_tool.name, arguments)


def test_tool_calling_agents_raises_agent_execution_error_when_tool_raises():
    @tool
    def _sample_tool(_: str) -> float:
        """Tool that fails

        Args:
            _: The pointless string
        Returns:
            Some number
        """

        return 1 / 0

    agent = ToolCallingAgent(model=FakeToolCallModel(), tools=[_sample_tool])
    with pytest.raises(AgentExecutionError):
        agent.execute_tool_call(_sample_tool.name, "sample")
