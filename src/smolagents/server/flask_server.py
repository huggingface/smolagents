#!/usr/bin/env python
# coding=utf-8
import importlib
import json
import os
import re
import types
from typing import Dict, Optional, Callable, Tuple
import time

import gradio as gr
import httpx
import yaml
from cachetools import TTLCache
from flask import Flask, Response, Blueprint, request, jsonify
from threading import Lock

from smolagents import Model, OpenAIServerModel, LiteLLMModel, MessageRole, StringBuffer
from smolagents.agent_types import AgentAudio, AgentImage, AgentText, AgentRaw, handle_agent_output_types
from smolagents.agents import ActionStep, MultiStepAgent, ChatMessage
from smolagents.memory import MemoryStep


def chat_message_to_json(chat_message):
    """将 gr.ChatMessage 对象转换为 JSON 字符串。"""
    # print(chat_message)
    if isinstance(chat_message, dict):
        return json.dumps(chat_message, ensure_ascii=False)

    message_dict = {
        "role": chat_message.role,
        "type": "text",
        "content": chat_message.content
    }
    metadata = chat_message.metadata
    if metadata and isinstance(metadata, dict):
        message_dict["metadata"] = metadata

    return json.dumps(message_dict, ensure_ascii=False)  # ensure_ascii=False 避免中文乱码


def pull_messages_from_step(step_log: MemoryStep):
    """Extract ChatMessage objects from agent steps with proper nesting"""
    if isinstance(step_log, ActionStep):
        # Output the step number
        yield {"role": "assistant", "type": "progress", "content": {"step": step_log.step_number}}
        step_number = step_log.step_number
        # First yield the thought/reasoning from the LLM
        if hasattr(step_log, "model_output") and step_log.model_output is not None:
            # Clean up the LLM output
            model_output = step_log.model_output.strip()
            pattern = r"Thought:(.*?)\nCode:(\n)?"
            match = re.search(pattern, model_output, re.DOTALL)
            if match:
                model_output = match.group(1).strip()  # 返回捕获组的内容并去除首尾空格
            yield {"role": "assistant", "type": "thinking", "content": model_output}

            # Nesting any errors under the tool call
            if hasattr(step_log, "error") and step_log.error is not None:
                yield gr.ChatMessage(
                    role="assistant",
                    content=str(step_log.error),
                )
        # Handle standalone errors but not from tool calls
        elif hasattr(step_log, "error") and step_log.error is not None:
            yield gr.ChatMessage(role="assistant", content=str(step_log.error), metadata={"title": "💥 Error"})

        # Calculate duration and token information
        step_footnote = f"{step_number}"
        if hasattr(step_log, "input_token_count") and hasattr(step_log, "output_token_count"):
            token_str = (
                f" | Input-tokens:{step_log.input_token_count:,} | Output-tokens:{step_log.output_token_count:,}"
            )
            step_footnote += token_str
        if hasattr(step_log, "duration") and step_log.duration is not None:
            step_duration = f" | Duration: {round(float(step_log.duration), 2)}" if step_log.duration else None
            step_footnote += step_duration
        # step_footnote = f"""*{step_footnote}*"""
        # yield gr.ChatMessage(role="assistant", content=f"{step_footnote}")
        # yield gr.ChatMessage(role="assistant", content="-----")


def stream_to_gradio(agent, task: str, reset_agent_memory: bool = False, additional_args: Optional[dict] = None):
    """Runs an agent with the given task and streams the messages from the agent as gradio ChatMessages."""
    total_input_tokens = 0
    total_output_tokens = 0

    for step_log in agent.run(task, stream=True, reset=reset_agent_memory, additional_args=additional_args):
        # Track tokens if model provides them
        if getattr(agent.model, "last_input_token_count", None) is not None:
            total_input_tokens += agent.model.last_input_token_count
            total_output_tokens += agent.model.last_output_token_count
            if isinstance(step_log, ActionStep):
                step_log.input_token_count = agent.model.last_input_token_count
                step_log.output_token_count = agent.model.last_output_token_count

        for message in pull_messages_from_step(step_log):
            yield message

    final_answer = step_log  # Last log is the run's final_answer
    final_answer = handle_agent_output_types(final_answer)

    if isinstance(final_answer, AgentText):
        yield gr.ChatMessage(
            role="assistant",
            content=final_answer.to_string(),
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
        for res in handle_agent_raw(final_answer):
            yield res


def value_to_gradio(value):
    return gr.ChatMessage(role="assistant", content=value)

def append_value_array(arr, buffer):
    for v in buffer:
        # first merge str content
        if isinstance(v, StringBuffer):
            append_value(arr, v.buffer)
        elif isinstance(v, list):
            append_value_array(arr, v)
        elif hasattr(v, "to_dict"):
            arr.append(value_to_gradio(v.to_dict()))
        elif isinstance(v, types.LambdaType):
            arr.append(value_to_gradio(v()))
        elif len(arr) > 0 and isinstance(arr[-1].content, str):
            arr[-1].content += str(v)
        else:
            arr.append(value_to_gradio(str(v)))


def append_value(arr, value):
    if isinstance(value, StringBuffer):
        append_value_array(arr, value.buffer)
    elif isinstance(value, list):
        append_value_array(arr, value)
    elif isinstance(value, dict):
        append_value_array(arr, [v for _, v in value.items()])
    else:
        append_value_array(arr, [value])

def handle_agent_raw(value):
    arr = []
    if isinstance(value, AgentRaw):
        value = value.to_raw()
    append_value(arr, value)
    return arr


class ModelList:
    def __init__(self, model_dir: Optional[Dict[str, Model]] = None):
        if model_dir is not None:
            self.model_dir = model_dir
        else:
            self.model_dir = {}

        self.parse_yaml()

    def add_model(self, name: str, type: str = None, proxy: str = None, http_client: httpx.Client | None = None,
                  **kwargs):
        if http_client is None and proxy is not None:
            http_client = httpx.Client(proxy=proxy, timeout=60)
        custom_role_conversions = {
            MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
            MessageRole.TOOL_RESPONSE: MessageRole.ASSISTANT,
        }
        if type == 'OpenAI':
            self.model_dir[name] = OpenAIServerModel(custom_role_conversions={}, http_client=http_client, **kwargs)
        else:
            self.model_dir[name] = LiteLLMModel(custom_role_conversions={}, client=http_client, **kwargs)

    def get_model(self, name: str) -> Model:
        if name in self.model_dir:
            return self.model_dir[name]
        else:
            raise ValueError(f"Model {name} not found")

    def parse_yaml(self):
        config = yaml.safe_load(
            importlib.resources.files("smolagents.server").joinpath("model_list.yaml").read_text(encoding='utf-8')
        )
        for model_name, model_config in config.items():
            self.add_model(model_name, **model_config)


class AgentCoordinator:
    def __init__(self,
                 model_list: ModelList,
                 agent_builder: Callable[[str, str, ModelList], ChatMessage],
                 cache: TTLCache):
        self.model_list = model_list
        self.agent_builder = agent_builder
        self.cache = cache
        self.agent_locks = {}
        self.locks_lock = Lock()

    def getAgent(self, chat_id: str, agent_id: str) -> Tuple[MultiStepAgent, Lock]:
        with self.locks_lock:
            # Get or create a lock for this chat_id
            if chat_id not in self.agent_locks:
                self.agent_locks[chat_id] = Lock()
            chat_lock = self.agent_locks[chat_id]
            
            # Get or create the agent
            agent = self.cache.get(chat_id)
            if agent is None:
                agent = self.agent_builder(chat_id, agent_id, self.model_list)
                self.cache[chat_id] = agent
            
            return agent, chat_lock

    def get_existing_agent(self, chat_id: str, agent_id: str) -> Tuple[Optional[MultiStepAgent], Optional[Lock]]:
        with self.locks_lock:
            agent = self.cache.get(chat_id)
            lock = self.agent_locks.get(chat_id)
            return agent, lock

    def get_status(self) -> dict:
        with self.locks_lock:
            return {
                "total_agents": len(self.cache),
                "active_locks": len(self.agent_locks),
                "cached_chat_ids": list(self.cache.keys()),
                "cache_info": {
                    "max_size": self.cache.maxsize,
                    "current_size": len(self.cache),
                    "ttl": self.cache.ttl
                }
            }


class ExecutionManager:
    def __init__(self):
        self.running_tasks = {}  # chat_id -> (lock, cancel_flag)
        self.manager_lock = Lock()
        # Add global statistics as direct attributes
        self.total_tasks = 0
        self.completed_tasks = 0
        self.cancelled_tasks = 0
        self.failed_tasks = 0  # New counter for failed tasks

    def start_task(self, chat_id: str, lock: Lock) -> dict:
        with self.manager_lock:
            cancel_flag = {
                "cancelled": False,
                "start_time": time.time()
            }
            self.running_tasks[chat_id] = (lock, cancel_flag)
            self.total_tasks += 1
            return cancel_flag

    def cancel_task(self, chat_id: str) -> bool:
        with self.manager_lock:
            if chat_id in self.running_tasks:
                lock, cancel_flag = self.running_tasks[chat_id]
                cancel_flag["cancelled"] = True
                self.cancelled_tasks += 1
                return True
            return False

    def end_task(self, chat_id: str, is_success: bool = True):
        with self.manager_lock:
            if chat_id in self.running_tasks:
                lock, cancel_flag = self.running_tasks[chat_id]
                end_time = time.time()
                duration = end_time - cancel_flag["start_time"]
                
                if cancel_flag["cancelled"]:
                    print(f"Task {chat_id} cancelled after {duration:.2f} seconds")
                elif is_success:
                    print(f"Task {chat_id} completed successfully in {duration:.2f} seconds")
                    self.completed_tasks += 1
                else:
                    print(f"Task {chat_id} failed after {duration:.2f} seconds")
                    self.failed_tasks += 1
                
                del self.running_tasks[chat_id]

    def get_status(self) -> dict:
        with self.manager_lock:
            status = {
                "total_tasks": self.total_tasks,
                "completed_tasks": self.completed_tasks,
                "cancelled_tasks": self.cancelled_tasks,
                "failed_tasks": self.failed_tasks,
                "success_rate": f"{(self.completed_tasks / self.total_tasks * 100):.1f}%" if self.total_tasks > 0 else "0%",
                "failure_rate": f"{(self.failed_tasks / self.total_tasks * 100):.1f}%" if self.total_tasks > 0 else "0%",
                "running_tasks_count": len(self.running_tasks),
                "running_tasks_details": []
            }
            
            current_time = time.time()
            for chat_id, (lock, cancel_flag) in self.running_tasks.items():
                task_status = {
                    "chat_id": chat_id,
                    "start_time": time.strftime("%Y-%m-%d %H:%M:%S", 
                                              time.localtime(cancel_flag['start_time'])),
                    "duration": f"{current_time - cancel_flag['start_time']:.2f}s",
                    "is_cancelled": cancel_flag["cancelled"]
                }
                status["running_tasks_details"].append(task_status)
            
            return status

class FlaskServer:
    def __init__(self, app: Flask, coordinator: AgentCoordinator, file_upload_folder: str | None = None):
        self.coordinator = coordinator
        self.file_upload_folder = file_upload_folder
        self.execution_manager = ExecutionManager()
        if self.file_upload_folder is not None:
            if not os.path.exists(file_upload_folder):
                os.mkdir(file_upload_folder)

        self.app = app
        self.register_blueprints()

    @staticmethod
    def interact_with_agent(agent: MultiStepAgent, prompt: str):
        for msg in stream_to_gradio(agent, task=prompt, reset_agent_memory=False):
            yield msg

    def register_blueprints(self):
        app = Blueprint('main', __name__)

        @app.route('/chat/completions', methods=["POST"])
        def api_completion():
            data = request.json
            try:
                model = self.coordinator.model_list.get_model(data.pop('model'))
                # COMPLETION CALL
                response = model(**data)
            except ValueError as e:
                # Handle model not found or invalid model parameters
                return jsonify({"error": str(e)}), 400
            except Exception as e:
                # Handle other unexpected errors
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

            return {'role': response.role, 'content': [{"type": "text", "text": response.content}]}

        @app.route('/agent/chat/memory', methods=["POST"])
        def agent_memory():
            data = request.json
            try:
                chat_id = data["chat_id"]
                agent_id = data["agent_id"]
                agent, lock = self.coordinator.get_existing_agent(chat_id, agent_id)
                if agent is None or lock is None:
                    return jsonify({"error": "Agent not found"}), 404
                
                # Check if the agent is currently being modified
                if lock.locked():
                    return jsonify({
                        "error": "Agent is currently executing a task. Please try again later."
                    }), 429  # 429 Too Many Requests
                
                # COMPLETION CALL - safe to read memory
                response = agent.write_memory_to_messages()
                return response
                
            except KeyError as e:
                # Handle missing required fields in request
                return jsonify({"error": f"Missing required field: {str(e)}"}), 400
            except Exception as e:
                # Handle other unexpected errors
                return jsonify({"error": f"Internal server error: {str(e)}"}), 500

        @app.route('/agent/chat/cancel', methods=['POST'])
        def cancel_chat():
            data = request.json
            if not data or 'chat_id' not in data:
                return jsonify({"error": "Missing chat_id"}), 400
            
            chat_id = data['chat_id']
            success = self.cancel_chat(chat_id)
            return jsonify({"success": success})

        @app.route('/agent/chat/completions', methods=['POST', 'GET'])
        def stream():
            data = request.json
            error_message, is_valid = self.validate_json(data)

            if not is_valid:
                return jsonify({"error": error_message}), 400

            chat_id = data["chat_id"]
            agent_id = data["agent_id"]
            messages = data["messages"]
            text = messages[0]['content'][0]['text']
            
            lock = None
            try:
                agent, lock = self.coordinator.getAgent(chat_id, agent_id)
                
                # Check if the lock is already acquired
                if not lock.acquire(blocking=False):
                    return jsonify({
                        "error": "Another task is currently being processed for this chat. Please wait and try again."
                    }), 429  # 429 Too Many Requests
                
                cancel_flag = self.execution_manager.start_task(chat_id, lock)

                def event_stream():
                    try:
                        res = self.interact_with_agent(agent, text)
                        for message in res:
                            if cancel_flag["cancelled"]:
                                yield f"data: {chat_message_to_json({'role': 'assistant', 'content': 'user cancelled the operation', 'type': 'error'})}\n\n"
                                break
                            yield f"data: {chat_message_to_json(message)}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {chat_message_to_json({'role': 'assistant', 'content': str(e), 'type': 'error'})}\n\n"
                        yield "data: [DONE]\n\n"
                        # Mark task as failed
                        self.execution_manager.end_task(chat_id, is_success=False)
                    else:
                        # Mark task as successful
                        self.execution_manager.end_task(chat_id, is_success=True)
                    finally:
                        if lock and lock.locked():
                            lock.release()

                return Response(event_stream(), content_type='text/event-stream; charset=utf-8')
            except Exception as e:
                # 确保在出现异常时也能释放锁和清理任务
                if lock and lock.locked():
                    lock.release()
                self.execution_manager.end_task(chat_id, is_success=False)
                return jsonify({"error": str(e)}), 500

        @app.route('/system/status', methods=['GET'])
        def system_status():
            try:
                status = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_manager": self.execution_manager.get_status(),
                    "agent_coordinator": self.coordinator.get_status()
                }
                return jsonify(status)
            except Exception as e:
                return jsonify({"error": f"Error getting system status: {str(e)}"}), 500

        self.app.register_blueprint(app, url_prefix='/app')

    def cancel_chat(self, chat_id: str) -> bool:
        """取消指定chat_id的执行任务"""
        return self.execution_manager.cancel_task(chat_id)

    @staticmethod
    def validate_json(data):
        """验证 JSON 数据是否符合预期结构"""
        if not isinstance(data, dict):
            return "Invalid JSON: data must be a dictionary", False

        required_keys = ["chat_id", "agent_id", "messages"]
        for key in required_keys:
            if key not in data:
                return f"Missing required key: {key}", False

        if not isinstance(data["messages"], list):
            return "Invalid type for messages: must be a list", False

        for message in data["messages"]:
            if not isinstance(message, dict):
                return "Invalid type for message: must be a dictionary", False
            if "role" not in message or "content" not in message:
                return "Missing required keys in message: role or content", False
            if not isinstance(message["role"], str):
                return "Invalid type for role: must be a string", False
            if not isinstance(message["content"], list) and not isinstance(message["content"], str):
                return "Invalid type for content: must be a list", False
            content = message["content"]
            message["content"] = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for content_item in message["content"]:
                if not isinstance(content_item, dict):
                    return "Invalid type for content item: must be a dictionary", False
                if "type" not in content_item or "text" not in content_item:
                    return "Missing required keys in content item: type or text", False
                if not isinstance(content_item["type"], str) or not isinstance(content_item["text"], str):
                    return "Invalid type for type or text in content item: must be a string", False

        return None, True  # 返回 None 表示没有错误，True 表示验证成功

    def run(self, **kwargs):
        self.app.run(**kwargs)
