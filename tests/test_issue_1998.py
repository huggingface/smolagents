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

from smolagents.agent_types import AgentText
from smolagents.agents import CodeAgent
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model


class FakeCodeModelSyntaxErrorAfterPrint(Model):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def generate(self, messages, stop_sequences=None):
        self.call_count += 1
        if self.call_count == 1:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should print the placeholder text first.
<code>
print("This is placeholder text. The real test will happen in step 2.")
</code>
""",
            )
        if self.call_count == 2:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="""
Thought: I should print the starting message and then emit invalid syntax.
<code>
print("Starting Test:")
def bad_syntax(
    pass
</code>
""",
            )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="""
Thought: I can now answer the initial question.
<code>
final_answer("done")
</code>
""",
        )


def test_issue_1998():
    agent = CodeAgent(tools=[], model=FakeCodeModelSyntaxErrorAfterPrint(), max_steps=3)

    output = agent.run(
        "Let's test your ability to handle syntax errors. In two separate steps, do the following:\n"
        "1. print('This is placeholder text. The real test will happen in step 2.')\n"
        "2. print('Starting Test:'), and then write code that has a syntax error. Don't use eval, and do not catch "
        "the exception. Just emit code with wrong syntax in the code block.\n"
        "After you've done both steps and they've raised an exception, stop and return via `final_answer`."
    )

    assert isinstance(output, AgentText)
    assert output == "done"

    action_steps = [step for step in agent.memory.steps if isinstance(step, ActionStep)]
    assert "This is placeholder text. The real test will happen in step 2." in str(action_steps[0].observations)
    assert "Code parsing failed" in str(action_steps[1].error)
    assert "This is placeholder text. The real test will happen in step 2." not in str(action_steps[1].observations)
