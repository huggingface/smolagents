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
import gradio as gr

from .agents import ActionStep, AgentStep, MultiStepAgent
from .types import AgentAudio, AgentImage, AgentText, handle_agent_output_types


def pull_messages_from_step(step_log: AgentStep, test_mode: bool = True):
    """Extract ChatMessage objects from agent steps"""
    if isinstance(step_log, ActionStep):
        yield gr.ChatMessage(role="assistant", content=step_log.llm_output or "")
        if step_log.tool_call is not None:
            used_code = step_log.tool_call.name == "code interpreter"
            content = step_log.tool_call.arguments
            if used_code:
                content = f"```py\n{content}\n```"
            yield gr.ChatMessage(
                role="assistant",
                metadata={"title": f"🛠️ Used tool {step_log.tool_call.name}"},
                content=str(content),
            )
        if step_log.observations is not None:
            yield gr.ChatMessage(role="assistant", content=step_log.observations)
        if step_log.error is not None:
            yield gr.ChatMessage(
                role="assistant",
                content=str(step_log.error),
                metadata={"title": "💥 Error"},
            )


def stream_to_gradio(
    agent,
    task: str,
    test_mode: bool = False,
    reset_agent_memory: bool = False,
    **kwargs,
):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, **kwargs):
        for message in pull_messages_from_step(step_log, test_mode=test_mode):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=f"**Final answer:**\n{final_answer.to_string()}\n",
        )
    elif isinstance(final_answer, AgentImage):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "image/png"},
        )
    elif isinstance(final_answer, AgentAudio):
        yield gr.ChatMessage(
            role="assistant",
            content={"path": final_answer.to_string(), "mime_type": "audio/wav"},
        )
    else:
        yield gr.ChatMessage(role="assistant", content=str(final_answer))


class GradioUI:
    """A one-line interface to launch your agent in Gradio"""

    def __init__(self, agent: MultiStepAgent):
        self.agent = agent

    def interact_with_agent(self, prompt, messages):
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def launch(self):
        with gr.Blocks() as demo:
            stored_message = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
                ),
            )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                lambda s: (s, ""), [text_input], [stored_message, text_input]
            ).then(self.interact_with_agent, [stored_message, chatbot], [chatbot])

        demo.launch()


__all__ = ["stream_to_gradio", "GradioUI"]
