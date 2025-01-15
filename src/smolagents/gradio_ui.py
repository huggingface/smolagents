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
import shutil
import os
import mimetypes
import re

from .agents import ActionStep, AgentStep, MultiStepAgent
from .types import AgentAudio, AgentImage, AgentText, handle_agent_output_types


def pull_messages_from_step(step_log: AgentStep, test_mode: bool = True):
    """Extract ChatMessage objects from agent steps"""
    if isinstance(step_log, ActionStep):
        yield gr.ChatMessage(role="assistant", content=step_log.llm_output or "")
        if step_log.tool_calls is not None:
            first_tool_call = step_log.tool_calls[0]
            used_code = first_tool_call.name == "code interpreter"
            content = first_tool_call.arguments
            if used_code:
                content = f"```py\n{content}\n```"
            yield gr.ChatMessage(
                role="assistant",
                metadata={"title": f"🛠️ Used tool {first_tool_call.name}"},
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

    def __init__(self, agent: MultiStepAgent, file_upload_folder: str | None = None):
        self.agent = agent
        self.file_upload_folder = file_upload_folder
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

    def interact_with_agent(self, prompt, messages):
        messages.append(gr.ChatMessage(role="user", content=prompt))
        yield messages
        for msg in stream_to_gradio(self.agent, task=prompt, reset_agent_memory=False):
            messages.append(msg)
            yield messages
        yield messages

    def upload_file(
        self,
        file,
        file_uploads_log,
        allowed_file_types=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
    ):
        """
        Handle file uploads, default allowed types are .pdf, .docx, and .txt
        """

        if file is None:
            return "No file uploaded"

        try:
            mime_type, _ = mimetypes.guess_type(file.name)
        except Exception as e:
            return f"Error: {e}"

        if mime_type not in allowed_file_types:
            return "File type disallowed"

        # Sanitize file name
        original_name = os.path.basename(file.name)
        sanitized_name = re.sub(
            r"[^\w\-.]", "_", original_name
        )  # Replace any non-alphanumeric, non-dash, or non-dot characters with underscores

        type_to_ext = {}
        for ext, t in mimetypes.types_map.items():
            if t not in type_to_ext:
                type_to_ext[t] = ext

        # Ensure the extension correlates to the mime type
        sanitized_name = sanitized_name.split(".")[:-1]
        sanitized_name.append("" + type_to_ext[mime_type])
        sanitized_name = "".join(sanitized_name)

        # Save the uploaded file to the specified folder
        file_path = os.path.join(
            self.file_upload_folder, os.path.basename(sanitized_name)
        )
        shutil.copy(file.name, file_path)

        return gr.Textbox(
            f"File uploaded: {file_path}", visible=True
        ), file_uploads_log + [file_path]

    def log_user_message(self, text_input, file_uploads_log):
        return (
            text_input
            + (f"\nYou have been provided with these files, which might be helpful or not: {file_uploads_log}"
            if len(file_uploads_log) > 0
            else ""),
            "",
        )

    def launch(self):
        with gr.Blocks() as demo:
            stored_messages = gr.State([])
            file_uploads_log = gr.State([])
            chatbot = gr.Chatbot(
                label="Agent",
                type="messages",
                avatar_images=(
                    None,
                    "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
                ),
            )
            # If an upload folder is provided, enable the upload feature
            if self.file_upload_folder is not None:
                upload_file = gr.File(label="Upload a file", height=1)
                upload_status = gr.Textbox(
                    label="Upload Status", interactive=False, visible=False
                )
                upload_file.change(
                    self.upload_file,
                    [upload_file, file_uploads_log],
                    [upload_status, file_uploads_log],
                )
            text_input = gr.Textbox(lines=1, label="Chat Message")
            text_input.submit(
                self.log_user_message,
                [text_input, file_uploads_log],
                [stored_messages, text_input],
            ).then(self.interact_with_agent, [stored_messages, chatbot], [chatbot])

        demo.launch()


__all__ = ["stream_to_gradio", "GradioUI"]
