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
import os
import tempfile
import unittest
import uuid
from pathlib import Path

from transformers.testing_utils import get_tests_dir

from smolagents.agents import (
    AgentMaxStepsError,
    CodeAgent,
    ManagedAgent,
    ToolCall,
    ToolCallingAgent,
)
from smolagents.default_tools import PythonInterpreterTool
from smolagents.tools import tool
from smolagents.types import AgentImage, AgentText
from huggingface_hub import (
    ChatCompletionOutputMessage,
    ChatCompletionOutputToolCall,
    ChatCompletionOutputFunctionDefinition,
)


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


class FakeToolCallModel:
    def __call__(
        self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None
    ):
        if len(messages) < 3:
            return ChatCompletionOutputMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionOutputToolCall(
                        id="call_0",
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            name="python_interpreter", arguments={"code": "2*3.6452"}
                        ),
                    )
                ],
            )
        else:
            return ChatCompletionOutputMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionOutputToolCall(
                        id="call_1",
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            name="final_answer", arguments={"answer": "7.2904"}
                        ),
                    )
                ],
            )


class FakeToolCallModelImage:
    def __call__(
        self, messages, tools_to_call_from=None, stop_sequences=None, grammar=None
    ):
        if len(messages) < 3:
            return ChatCompletionOutputMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionOutputToolCall(
                        id="call_0",
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            name="fake_image_generation_tool",
                            arguments={"prompt": "An image of a cat"},
                        ),
                    )
                ],
            )
        else:
            return ChatCompletionOutputMessage(
                role="assistant",
                content="",
                tool_calls=[
                    ChatCompletionOutputToolCall(
                        id="call_1",
                        type="function",
                        function=ChatCompletionOutputFunctionDefinition(
                            name="final_answer", arguments="image.png"
                        ),
                    )
                ],
            )


def fake_code_model(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
""",
        )


def fake_code_model_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
a = 2
b = a * 2
print = 2
print("Ok, calculation done!")
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
""",
        )


def fake_code_model_syntax_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
a = 2
b = a * 2
    print("Failing due to unexpected indent")
print("Ok, calculation done!")
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
""",
        )


def fake_code_model_import(messages, stop_sequences=None) -> str:
    return ChatCompletionOutputMessage(
        role="assistant",
        content="""
Thought: I can answer the question
Code:
```py
import numpy as np
final_answer("got an error")
```<end_code>
""",
    )


def fake_code_functiondef(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: Let's define the function. special_marker
Code:
```py
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
```<end_code>
""",
        )
    else:  # We're at step 2
        return ChatCompletionOutputMessage(
            role="assistant",
            content="""
Thought: I can now answer the initial question
Code:
```py
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
```<end_code>
""",
        )


def fake_code_model_single_step(messages, stop_sequences=None, grammar=None) -> str:
    return ChatCompletionOutputMessage(
        role="assistant",
        content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
""",
    )


def fake_code_model_no_return(messages, stop_sequences=None, grammar=None) -> str:
    return ChatCompletionOutputMessage(
        role="assistant",
        content="""
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
""",
    )


class AgentTests(unittest.TestCase):
    def test_fake_single_step_code_agent(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()], model=fake_code_model_single_step
        )
        output = agent.run("What is 2 multiplied by 3.6452?", single_step=True)
        assert isinstance(output, str)
        assert "7.2904" in output

    def test_fake_toolcalling_agent(self):
        agent = ToolCallingAgent(
            tools=[PythonInterpreterTool()], model=FakeToolCallModel()
        )
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert "7.2904" in output
        assert agent.logs[1].task == "What is 2 multiplied by 3.6452?"
        assert "7.2904" in agent.logs[2].observations
        assert agent.logs[3].llm_output is None

    def test_toolcalling_agent_handles_image_tool_outputs(self):
        from PIL import Image

        @tool
        def fake_image_generation_tool(prompt: str) -> Image.Image:
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """
            return Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png")

        agent = ToolCallingAgent(
            tools=[fake_image_generation_tool], model=FakeToolCallModelImage()
        )
        output = agent.run("Make me an image.")
        assert isinstance(output, AgentImage)
        assert isinstance(agent.state["image.png"], Image.Image)

    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.logs[1].task == "What is 2 multiplied by 3.6452?"
        assert agent.logs[3].tool_call == ToolCall(
            name="python_interpreter", arguments="final_answer(7.2904)", id="call_3"
        )

    def test_additional_args_added_to_task(self):
        agent = CodeAgent(tools=[], model=fake_code_model)
        agent.run(
            "What is 2 multiplied by 3.6452?",
            additional_args={"instruction": "Remember this."},
        )
        assert "Remember this" in agent.task
        assert "Remember this" in str(agent.input_messages)

    def test_reset_conversations(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model)
        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.logs) == 4

        output = agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.logs) == 6

        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.logs) == 4

    def test_code_agent_code_errors_show_offending_lines(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], model=fake_code_model_error)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Code execution failed at line 'print = 2' because of" in str(agent.logs)

    def test_code_agent_syntax_error_show_offending_lines(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()], model=fake_code_model_syntax_error
        )
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert '    print("Failing due to unexpected indent")' in str(agent.logs)

    def test_setup_agent_with_empty_toolbox(self):
        ToolCallingAgent(model=FakeToolCallModel(), tools=[])

    def test_fails_max_steps(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            model=fake_code_model_no_return,  # use this callable because it never ends
            max_steps=5,
        )
        agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.logs) == 8
        assert type(agent.logs[-1].error) is AgentMaxStepsError

    def test_tool_descriptions_get_baked_in_system_prompt(self):
        tool = PythonInterpreterTool()
        tool.name = "fake_tool_name"
        tool.description = "fake_tool_description"
        agent = CodeAgent(tools=[tool], model=fake_code_model)
        agent.run("Empty task")
        assert tool.name in agent.system_prompt
        assert tool.description in agent.system_prompt

    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = CodeAgent(tools=toolset_1, model=fake_code_model)
        assert (
            len(agent.tools) == 1
        )  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        agent = CodeAgent(tools=toolset_2, model=fake_code_model)
        assert (
            len(agent.tools) == 2
        )  # deduplication of tools, so only one python_interpreter tool is added in addition to final_answer

        # check that python_interpreter base tool does not get added to CodeAgent
        agent = CodeAgent(tools=[], model=fake_code_model, add_base_tools=True)
        assert len(agent.tools) == 3  # added final_answer tool + search + visit_webpage

        # check that python_interpreter base tool gets added to ToolCallingAgent
        agent = ToolCallingAgent(tools=[], model=fake_code_model, add_base_tools=True)
        assert len(agent.tools) == 4  # added final_answer tool + search + visit_webpage

    def test_function_persistence_across_steps(self):
        agent = CodeAgent(
            tools=[],
            model=fake_code_functiondef,
            max_steps=2,
            additional_authorized_imports=["numpy"],
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        agent = CodeAgent(tools=[], model=fake_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        assert managed_agent.name == "managed_agent"
        assert managed_agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        agent = CodeAgent(tools=[], model=fake_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        manager_agent = CodeAgent(
            tools=[],
            model=fake_code_functiondef,
            managed_agents=[managed_agent],
        )
        assert "You can also give requests to team members." not in agent.system_prompt
        print("ok1")
        assert "{{managed_agents_descriptions}}" not in agent.system_prompt
        assert (
            "You can also give requests to team members." in manager_agent.system_prompt
        )

    def test_code_agent_missing_import_triggers_advice_in_error_log(self):
        agent = CodeAgent(tools=[], model=fake_code_model_import)

        from smolagents.agents import console

        with console.capture() as capture:
            agent.run("Count to 3")
        str_output = capture.get()
        assert "import under `additional_authorized_imports`" in str_output

    def test_multiagents(self):
        class FakeModelMultiagentsManagerAgent:
            def __call__(
                self,
                messages,
                stop_sequences=None,
                grammar=None,
                tools_to_call_from=None,
            ):
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatCompletionOutputMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatCompletionOutputFunctionDefinition(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatCompletionOutputMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatCompletionOutputToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatCompletionOutputFunctionDefinition(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatCompletionOutputMessage(
                            role="assistant",
                            content="""
Thought: Let's call our search agent.
Code:
```py
result = search_agent("Who is the current US president?")
```<end_code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatCompletionOutputMessage(
                            role="assistant",
                            content="""
Thought: Let's return the report.
Code:
```py
final_answer("Final report.")
```<end_code>
""",
                        )

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent:
            def __call__(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
                grammar=None,
            ):
                return ChatCompletionOutputMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatCompletionOutputToolCall(
                            id="call_0",
                            type="function",
                            function=ChatCompletionOutputFunctionDefinition(
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
        )

        managed_web_agent = ManagedAgent(
            agent=web_agent,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[managed_web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[managed_web_agent],
        )

        report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."

    def test_code_nontrivial_final_answer_works(self):
        def fake_code_model_final_answer(messages, stop_sequences=None, grammar=None):
            return ChatCompletionOutputMessage(
                role="assistant",
                content="""Code:
```py
def nested_answer():
    final_answer("Correct!")

nested_answer()
```<end_code>""",
            )

        agent = CodeAgent(tools=[], model=fake_code_model_final_answer)

        output = agent.run("Count to 3")
        assert output == "Correct!"
