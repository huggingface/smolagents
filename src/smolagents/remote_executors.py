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
import base64
import inspect
import json
import os
import pickle
import re
import secrets
import subprocess
import tempfile
import time
import uuid
from contextlib import closing
from io import BytesIO
from textwrap import dedent
from typing import Any, Optional

import PIL.Image
import requests
from requests.exceptions import RequestException

from .default_tools import FinalAnswerTool
from .local_python_executor import CodeOutput, PythonExecutor
from .monitoring import LogLevel
from .serialization import SafeSerializer, SerializationError
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


__all__ = ["BlaxelExecutor", "E2BExecutor", "ModalExecutor", "DockerExecutor", "WasmExecutor"]


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    """
    Executor of Python code in a remote environment.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
    """

    FINAL_ANSWER_EXCEPTION = "FinalAnswerException"

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
    ):
        self.additional_imports = additional_imports
        self.logger = logger
        self.allow_pickle = allow_pickle
        self.logger.log("Initializing executor, hold on...")
        self.installed_packages = []

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the remote environment and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        raise NotImplementedError

    def send_tools(self, tools: dict[str, Tool]):
        if "final_answer" in tools:
            self._patch_final_answer_with_exception(tools["final_answer"])
        # Install tool packages
        packages_to_install = {
            pkg
            for tool in tools.values()
            for pkg in tool.to_dict()["requirements"]
            if pkg not in self.installed_packages + ["smolagents"]
        }
        if "PIL" in packages_to_install:
            packages_to_install.discard("PIL")
            packages_to_install.add("pillow")
        if packages_to_install:
            self.installed_packages += self.install_packages(list(packages_to_install))
        # Get tool definitions
        code = get_tools_definition_code(tools)
        if code:
            code_output = self.run_code_raise_errors(code)
            self.logger.log(code_output.logs)

    def send_variables(self, variables: dict[str, Any]):
        """Send variables to the kernel namespace using SafeSerializer.

        Uses prefix-based format ("safe:..." or "pickle:...").
        When allow_pickle=False, only safe JSON serialization is allowed.
        When allow_pickle=True, pickle fallback is enabled for complex types.
        """
        if not variables:
            return

        serialized = SafeSerializer.dumps(variables, allow_pickle=self.allow_pickle)
        code = f"""
{SafeSerializer.get_deserializer_code(self.allow_pickle)}
vars_dict = _deserialize({repr(serialized)})
locals().update(vars_dict)
"""
        self.run_code_raise_errors(code)

    def __call__(self, code_action: str) -> CodeOutput:
        """Run the code and determine if it is the final answer."""
        return self.run_code_raise_errors(code_action)

    def install_packages(self, additional_imports: list[str]):
        if additional_imports:
            code_output = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
            self.logger.log(code_output.logs)
        return additional_imports

    def _patch_final_answer_with_exception(self, final_answer_tool: FinalAnswerTool):
        """Patch the FinalAnswerTool to raise an exception.

        This is necessary because the remote executors
        rely on the FinalAnswerTool to detect the final answer.
        It modifies the `forward` method of the FinalAnswerTool to raise
        a `FinalAnswerException` with the final answer as a serialized value.
        This allows the executor to catch this exception and return the final answer.

        Uses prefix-based format ("safe:" or "pickle:") for serialization.

        Args:
            final_answer_tool (`FinalAnswerTool`): FinalAnswerTool instance to patch.
        """

        # Create a new class that inherits from the original FinalAnswerTool
        class _FinalAnswerTool(final_answer_tool.__class__):
            pass

        # Add a new forward method that raises the FinalAnswerException
        # NOTE: Serialization logic is inlined here because this method's source code
        # is extracted and sent to remote environments where external references don't exist
        # Capture settings via closure
        allow_pickle_setting = self.allow_pickle

        def forward(self, *args, **kwargs) -> Any:
            import base64
            import json
            from io import BytesIO

            # Baked in from closure at patch time
            ALLOW_PICKLE = allow_pickle_setting

            class SerializationError(Exception):
                pass

            def _to_json_safe(obj):
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                elif isinstance(obj, dict):
                    # Check if all keys are strings (JSON-compatible)
                    if all(isinstance(k, str) for k in obj.keys()):
                        return {k: _to_json_safe(v) for k, v in obj.items()}
                    else:
                        return {
                            "__type__": "dict_with_complex_keys",
                            "data": [[_to_json_safe(k), _to_json_safe(v)] for k, v in obj.items()],
                        }
                elif isinstance(obj, list):
                    return [_to_json_safe(item) for item in obj]
                elif isinstance(obj, tuple):
                    return {"__type__": "tuple", "data": [_to_json_safe(item) for item in obj]}
                elif isinstance(obj, set):
                    return {"__type__": "set", "data": [_to_json_safe(item) for item in obj]}
                elif isinstance(obj, bytes):
                    return {"__type__": "bytes", "data": base64.b64encode(obj).decode()}
                elif isinstance(obj, complex):
                    return {"__type__": "complex", "real": obj.real, "imag": obj.imag}
                elif isinstance(obj, frozenset):
                    return {"__type__": "frozenset", "data": [_to_json_safe(item) for item in obj]}

                # Try PIL Image
                try:
                    import PIL.Image

                    if isinstance(obj, PIL.Image.Image):
                        buffer = BytesIO()
                        obj.save(buffer, format="PNG")
                        return {"__type__": "PIL.Image", "data": base64.b64encode(buffer.getvalue()).decode()}
                except ImportError:
                    pass

                # Lazy imports for less common types
                from datetime import date, datetime, time, timedelta
                from decimal import Decimal
                from pathlib import Path

                if isinstance(obj, datetime):
                    return {"__type__": "datetime", "data": obj.isoformat()}
                elif isinstance(obj, date):
                    return {"__type__": "date", "data": obj.isoformat()}
                elif isinstance(obj, time):
                    return {"__type__": "time", "data": obj.isoformat()}
                elif isinstance(obj, timedelta):
                    return {"__type__": "timedelta", "total_seconds": obj.total_seconds()}
                elif isinstance(obj, Decimal):
                    return {"__type__": "Decimal", "data": str(obj)}
                elif isinstance(obj, Path):
                    return {"__type__": "Path", "data": str(obj)}

                # Try numpy if available
                try:
                    import numpy as np

                    if isinstance(obj, np.ndarray):
                        return {"__type__": "ndarray", "data": obj.tolist(), "dtype": str(obj.dtype)}
                    elif isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                except ImportError:
                    pass

                # Try dataclass
                import dataclasses

                if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                    return {
                        "__type__": "dataclass",
                        "class_name": type(obj).__name__,
                        "module": type(obj).__module__,
                        "data": {f.name: _to_json_safe(getattr(obj, f.name)) for f in dataclasses.fields(obj)},
                    }

                # Cannot safely serialize - raise error for safe mode
                raise SerializationError(f"Cannot safely serialize object of type {type(obj).__name__}")

            def _serialize_with_fallback(obj):
                """Serialize with safe method, fallback to pickle if allowed."""
                import pickle

                if not ALLOW_PICKLE:
                    # Safe ONLY mode - NO pickle fallback, raise error if can't serialize
                    json_safe = _to_json_safe(obj)  # Will raise SerializationError if fails
                    return "safe:" + json.dumps(json_safe)
                else:
                    # Try safe first, fallback to pickle if allowed
                    try:
                        json_safe = _to_json_safe(obj)
                        return "safe:" + json.dumps(json_safe)
                    except SerializationError:
                        # Fallback to pickle
                        try:
                            return "pickle:" + base64.b64encode(pickle.dumps(obj)).decode()
                        except (pickle.PicklingError, TypeError, AttributeError):
                            # Last resort: string representation
                            return "safe:" + json.dumps(str(obj))

            class FinalAnswerException(BaseException):
                def __init__(self, value):
                    self.value = value

            raise FinalAnswerException(_serialize_with_fallback(self._forward(*args, **kwargs)))

        # - Set the new forward method function to the _FinalAnswerTool class
        _FinalAnswerTool.forward = forward

        # Set __source__ with the actual values baked in (closures don't survive source extraction)
        source = inspect.getsource(forward)
        source = source.replace("ALLOW_PICKLE = allow_pickle_setting", f"ALLOW_PICKLE = {allow_pickle_setting}")
        forward.__source__ = source

        # Rename the original forward method to _forward
        # - Get the original forward method function from the final_answer_tool instance
        original_forward_function = final_answer_tool.forward.__func__
        # - Set the new _forward method function to the _FinalAnswerTool class
        _FinalAnswerTool._forward = original_forward_function
        # - Update the source code of the new forward method to match the original but with the new name
        _FinalAnswerTool._forward.__source__ = inspect.getsource(original_forward_function).replace(
            "def forward(", "def _forward("
        )

        # Set the new class as the class of the final_answer_tool instance
        final_answer_tool.__class__ = _FinalAnswerTool

    @staticmethod
    def _deserialize_final_answer(encoded_value: str, allow_pickle: bool = True) -> Any:
        """Deserialize final answer with format detection.

        Handle both new prefix-based format ("safe:" or "pickle:") and
        legacy no-prefix base64-pickle format for backward compatibility.

        Args:
            encoded_value: Serialized string from FinalAnswerException.
            allow_pickle: Whether to allow pickle deserialization.

        Returns:
            Deserialized Python object.

        Raises:
            SerializationError: If pickle data is rejected.
        """
        if encoded_value.startswith("safe:"):
            json_data = json.loads(encoded_value[5:])
            return SafeSerializer.from_json_safe(json_data)
        elif encoded_value.startswith("pickle:"):
            if not allow_pickle:
                raise SerializationError("Pickle data rejected: allow_pickle=False")
            return pickle.loads(base64.b64decode(encoded_value[7:]))
        else:
            # Legacy format (no prefix) - base64-encoded pickle
            if not allow_pickle:
                raise SerializationError("Pickle data rejected: allow_pickle=False")
            return pickle.loads(base64.b64decode(encoded_value))


class E2BExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in an E2B sandbox.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        **kwargs: Additional keyword arguments to pass to the E2B Sandbox instantiation.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        **kwargs,
    ):
        super().__init__(additional_imports, logger, allow_pickle)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        # Support both e2b v1 and v2 constructors
        # v2 exposes Sandbox.create(...), while v1 uses Sandbox(...)
        if hasattr(Sandbox, "create"):
            self.sandbox = Sandbox.create(**kwargs)
        else:
            self.sandbox = Sandbox(**kwargs)
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running", level=LogLevel.INFO)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the E2B sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        execution = self.sandbox.run_code(code)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])

        # Handle errors
        if execution.error:
            # Check if the error is a FinalAnswerException
            if execution.error.name == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                final_answer = self._deserialize_final_answer(execution.error.value, self.allow_pickle)
                return CodeOutput(output=final_answer, logs=execution_logs, is_final_answer=True)

            # Construct error message
            error_message = (
                f"{execution_logs}\n"
                f"Executing code yielded an error:\n"
                f"{execution.error.name}\n"
                f"{execution.error.value}\n"
                f"{execution.error.traceback}"
            )
            raise AgentError(error_message, self.logger)

        # Handle results
        if not execution.results:
            return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

        for result in execution.results:
            if not result.is_main_result:
                continue
            # Handle image outputs
            for attribute_name in ["jpeg", "png"]:
                img_data = getattr(result, attribute_name, None)
                if img_data is not None:
                    decoded_bytes = base64.b64decode(img_data.encode("utf-8"))
                    return CodeOutput(
                        output=PIL.Image.open(BytesIO(decoded_bytes)), logs=execution_logs, is_final_answer=False
                    )
            # Handle other data formats
            for attribute_name in [
                "chart",
                "data",
                "html",
                "javascript",
                "json",
                "latex",
                "markdown",
                "pdf",
                "svg",
                "text",
            ]:
                data = getattr(result, attribute_name, None)
                if data is not None:
                    return CodeOutput(output=data, logs=execution_logs, is_final_answer=False)
        # If no main result found, return None
        return CodeOutput(output=None, logs=execution_logs, is_final_answer=False)

    def cleanup(self):
        """Clean up the E2B sandbox and resources."""
        try:
            if hasattr(self, "sandbox"):
                self.logger.log("Shutting down sandbox...", level=LogLevel.INFO)
                self.sandbox.kill()
                self.logger.log("Sandbox cleanup completed", level=LogLevel.INFO)
                del self.sandbox
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")


def _websocket_send_execute_request(code: str, ws) -> str:
    """Send code execution request to kernel."""
    import uuid

    # Generate a unique message ID
    msg_id = str(uuid.uuid4())

    # Create execute request
    execute_request = {
        "header": {
            "msg_id": msg_id,
            "username": "anonymous",
            "session": str(uuid.uuid4()),
            "msg_type": "execute_request",
            "version": "5.0",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
        },
    }

    ws.send(json.dumps(execute_request))
    return msg_id


def _websocket_run_code_raise_errors(
    code: str, ws, logger, allow_pickle: bool = True, safe_serialization: bool = False
) -> CodeOutput:
    """Run code over a websocket."""
    try:
        # Send execute request
        msg_id = _websocket_send_execute_request(code, ws)

        # Collect output and results
        outputs = []
        result = None
        is_final_answer = False

        while True:
            msg = json.loads(ws.recv())
            parent_msg_id = msg.get("parent_header", {}).get("msg_id")
            # Skip unrelated messages
            if parent_msg_id != msg_id:
                continue
            msg_type = msg.get("msg_type", "")
            msg_content = msg.get("content", {})
            if msg_type == "stream":
                outputs.append(msg_content["text"])
            elif msg_type == "execute_result":
                result = msg_content["data"].get("text/plain", None)
            elif msg_type == "error":
                if msg_content.get("ename", "") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                    result = RemotePythonExecutor._deserialize_final_answer(
                        msg_content.get("evalue", ""), allow_pickle
                    )
                    is_final_answer = True
                else:
                    raise AgentError("\n".join(msg_content.get("traceback", [])), logger)
            elif msg_type == "status" and msg_content["execution_state"] == "idle":
                break

        return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

    except Exception as e:
        logger.log_error(f"Code execution failed: {e}")
        raise


def _create_kernel_http(crate_kernel_endpoint: str, logger, headers: Optional[dict] = None) -> str:
    """Create kernel using http."""

    r = requests.post(crate_kernel_endpoint, headers=headers)
    if r.status_code != 201:
        error_details = {
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "url": r.url,
            "body": r.text,
            "request_method": r.request.method,
            "request_headers": dict(r.request.headers),
            "request_body": r.request.body,
        }
        logger.log_error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
        raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None
    return r.json()["id"]


class DockerExecutor(RemotePythonExecutor):
    """
    Remote Python code executor using Jupyter Kernel Gateway in a Docker container.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        host (`str`, default `"127.0.0.1"`): Host to bind to.
        port (`int`, default `8888`): Port to bind to.
        image_name (`str`, default `"jupyter-kernel"`): Name of the Docker image to use. If the image doesn't exist, it will be built.
        build_new_image (`bool`, default `True`): Whether to rebuild a new image even if it already exists.
        container_run_kwargs (`dict`, *optional*): Additional keyword arguments to pass to the Docker container run command.
        dockerfile_content (`str`, *optional*): Custom Dockerfile content. If `None`, uses default.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        host: str = "127.0.0.1",
        port: int = 8888,
        image_name: str = "jupyter-kernel",
        build_new_image: bool = True,
        container_run_kwargs: dict[str, Any] | None = None,
        dockerfile_content: str | None = None,
    ):
        super().__init__(additional_imports, logger, allow_pickle)
        try:
            import docker
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )
        self.host = host
        self.port = port
        self.image_name = image_name

        self.dockerfile_content = dockerfile_content or dedent(
            """\
            FROM python:3.12-bullseye

            RUN pip install jupyter_kernel_gateway jupyter_client ipykernel

            EXPOSE 8888
            CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
            """
        )

        # Initialize Docker
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

        # Build and start container
        try:
            # Check if image exists, unless forced to rebuild
            if not build_new_image:
                try:
                    self.client.images.get(self.image_name)
                    self.logger.log(f"Using existing Docker image: {self.image_name}", level=LogLevel.INFO)
                except docker.errors.ImageNotFound:
                    self.logger.log(f"Image {self.image_name} not found, building...", level=LogLevel.INFO)
                    build_new_image = True

            if build_new_image:
                self.logger.log(f"Building Docker image {self.image_name}...", level=LogLevel.INFO)
                dockerfile_obj = BytesIO(self.dockerfile_content.encode("utf-8"))
                _, build_logs = self.client.images.build(fileobj=dockerfile_obj, tag=self.image_name)
                for log_chunk in build_logs:
                    # Only log non-empty messages
                    if log_message := log_chunk.get("stream", "").rstrip():
                        self.logger.log(log_message, level=LogLevel.DEBUG)

            self.logger.log(f"Starting container on {host}:{port}...", level=LogLevel.INFO)
            # Create base container parameters
            container_kwargs = {}
            if container_run_kwargs:
                container_kwargs.update(container_run_kwargs)

            # Ensure required port mapping and background running
            if not isinstance(container_kwargs.get("ports"), dict):
                container_kwargs["ports"] = {}
            container_kwargs["ports"]["8888/tcp"] = (host, port)
            container_kwargs["detach"] = True

            self.container = self.client.containers.run(self.image_name, **container_kwargs)

            retries = 0
            while self.container.status != "running" and retries < 5:
                self.logger.log(f"Container status: {self.container.status}, waiting...", level=LogLevel.INFO)
                time.sleep(1)
                self.container.reload()
                retries += 1

            self.base_url = f"http://{host}:{port}"

            # Wait for Jupyter to start
            self._wait_for_server()

            # Create new kernel via HTTP
            self.kernel_id = _create_kernel_http(f"{self.base_url}/api/kernels", self.logger)
            self.ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"

            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with kernel {self.kernel_id}", level=LogLevel.INFO
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the Docker container and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def cleanup(self):
        """Clean up the Docker container and resources."""
        try:
            if hasattr(self, "container"):
                self.logger.log(f"Stopping and removing container {self.container.short_id}...", level=LogLevel.INFO)
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed", level=LogLevel.INFO)
                del self.container
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self):
        retries = 0
        jupyter_ready = False
        while not jupyter_ready and retries < 10:
            try:
                if requests.get(f"{self.base_url}/api/kernelspecs", timeout=2).status_code == 200:
                    jupyter_ready = True
                else:
                    self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            except requests.RequestException:
                self.logger.log("Jupyter not ready, waiting...", level=LogLevel.INFO)
            if not jupyter_ready:
                time.sleep(1)
                retries += 1


class ModalExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in a Modal sandbox.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        app_name (`str`, default `"smolagent-executor"`): App name.
        port (`int`, default `8888`): Port for jupyter to bind to.
        create_kwargs (`dict`, *optional*): Additional keyword arguments to pass to the Modal Sandbox create command. See
            `modal.Sandbox.create` [docs](https://modal.com/docs/reference/modal.Sandbox#create) for all the
            keyword arguments.
    """

    _ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        app_name: str = "smolagent-executor",
        port: int = 8888,
        create_kwargs: Optional[dict] = None,
    ):
        super().__init__(additional_imports, logger, allow_pickle)
        self.port = port
        try:
            import modal
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'modal' extra to use ModalExecutor: `pip install 'smolagents[modal]'`"""
            )

        if create_kwargs is None:
            create_kwargs = {}

        create_kwargs = {
            "image": modal.Image.debian_slim().uv_pip_install("jupyter_kernel_gateway", "ipykernel"),
            "timeout": 60 * 5,
            **create_kwargs,
        }

        if "app" not in create_kwargs:
            create_kwargs["app"] = modal.App.lookup(app_name, create_if_missing=True)

        if "encrypted_ports" not in create_kwargs:
            create_kwargs["encrypted_ports"] = [port]
        else:
            create_kwargs["encrypted_ports"] = create_kwargs["encrypted_ports"] + [port]

        token = secrets.token_urlsafe(16)
        default_secrets = [modal.Secret.from_dict({"KG_AUTH_TOKEN": token})]

        if "secrets" not in create_kwargs:
            create_kwargs["secrets"] = default_secrets
        else:
            create_kwargs["secrets"] = create_kwargs["secrets"] + default_secrets

        entrypoint = [
            "jupyter",
            "kernelgateway",
            "--KernelGatewayApp.ip='0.0.0.0'",
            f"--KernelGatewayApp.port={port}",
            "--KernelGatewayApp.allow_origin='*'",
        ]

        self.logger.log("Starting Modal sandbox", level=LogLevel.INFO)
        self.sandbox = modal.Sandbox.create(
            *entrypoint,
            **create_kwargs,
        )

        tunnel = self.sandbox.tunnels()[port]
        self.logger.log(f"Waiting for Modal sandbox on {tunnel.host}:{port}", level=LogLevel.INFO)
        self._wait_for_server(tunnel.host, token)

        self.logger.log("Starting Jupyter kernel", level=LogLevel.INFO)
        kernel_id = _create_kernel_http(f"https://{tunnel.host}/api/kernels?token={token}", logger)
        self.ws_url = f"wss://{tunnel.host}/api/kernels/{kernel_id}/channels?token={token}"
        self.installed_packages = self.install_packages(additional_imports)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the Modal sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from websocket import create_connection

        with closing(create_connection(self.ws_url)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def cleanup(self):
        """Clean up the Modal sandbox by terminating it."""
        if hasattr(self, "sandbox"):
            self.sandbox.terminate()

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def _wait_for_server(self, host: str, token: str):
        """Wait for server to start up."""
        n_retries = 0
        while True:
            try:
                resp = requests.get(f"https://{host}/api/kernelspecs?token={token}")
                if resp.status_code == 200:
                    break
            except RequestException:
                n_retries += 1
                if n_retries % 10 == 0:
                    self.logger.log("Waiting for server to startup, retrying...", level=LogLevel.INFO)
                if n_retries > 60:
                    raise RuntimeError("Unable to connect to sandbox")
                time.sleep(1.0)

    @classmethod
    def _strip_ansi_colors(cls, text: str) -> str:
        """Remove ansi colors from text."""
        return cls._ANSI_ESCAPE.sub("", text)


class BlaxelExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in a Blaxel sandbox.

    Blaxel provides fast-launching virtual machines that start from hibernation in under 25ms
    and scale back to zero after inactivity while maintaining memory state.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        sandbox_name (`str`, *optional*): Name for the sandbox. Defaults to "smolagent-executor".
        image (`str`, default `"blaxel/jupyter-notebook"`): Docker image to use.
        memory (`int`, default `4096`): Memory allocation in MB.
        ttl (`str`, *optional*): Time to live in seconds.
        region (`str`, *optional*): Deployment region. If not specified, Blaxel chooses default.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        sandbox_name: str | None = None,
        image: str = "blaxel/jupyter-notebook",
        memory: int = 4096,
        ttl: str | None = None,
        region: Optional[str] = None,
    ):
        super().__init__(additional_imports, logger, allow_pickle=allow_pickle)

        try:
            import blaxel  # noqa: F401
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'blaxel' extra to use BlaxelExecutor: `pip install 'smolagents[blaxel]'`"
            )

        self.sandbox_name = sandbox_name or f"smolagent-executor-{uuid.uuid4().hex[:8]}"
        self.image = image
        self.memory = memory
        self.region = region
        self.port = 8888
        self._cleaned_up = False  # Flag to prevent double cleanup

        # Prepare sandbox creation parameters
        token = secrets.token_urlsafe(16)
        sandbox_config = {
            "metadata": {
                "name": self.sandbox_name,
            },
            "spec": {
                "runtime": {"image": image, "memory": memory, "ports": [{"target": self.port}]},
            },
        }

        if region:
            sandbox_config["spec"]["region"] = region

        if ttl:
            sandbox_config["spec"]["runtime"]["ttl"] = ttl

        # Create the sandbox
        try:
            # Create sandbox environment on Blaxel
            self.sandbox = BlaxelExecutor._create_sandbox(sandbox_config)

            # Create kernel via HTTP
            from blaxel.core import settings

            kernel_id = _create_kernel_http(
                f"{self.sandbox.metadata.url}/port/{self.port}/api/kernels?token={token}",
                self.logger,
                headers=settings.headers,
            )

            # Set up websocket URL
            # Convert http/https to ws/wss
            ws_scheme = "wss" if self.sandbox.metadata.url.startswith("https") else "ws"
            ws_base = self.sandbox.metadata.url.replace("https://", "").replace("http://", "")
            self.ws_url = f"{ws_scheme}://{ws_base}/port/{self.port}/api/kernels/{kernel_id}/channels?token={token}"

            # Install additional packages
            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log("Blaxel is running", level=LogLevel.INFO)
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Blaxel sandbox: {e}") from e

    @staticmethod
    def _create_sandbox(config):
        """Helper method to create sandbox asynchronously."""
        from blaxel.core import SandboxInstance
        from blaxel.core.client import client
        from blaxel.core.client.api.compute import create_sandbox

        response = create_sandbox.sync(client=client, body=config)
        return SandboxInstance(response)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the Blaxel sandbox and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        from blaxel.core import settings
        from websocket import create_connection

        headers = []
        for key, value in settings.headers.items():
            headers.append(f"{key}: {value}")
        with closing(create_connection(self.ws_url, header=headers)) as ws:
            return _websocket_run_code_raise_errors(code, ws, self.logger, self.allow_pickle)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        """Helper method to install packages asynchronously."""
        if not additional_imports:
            return []

        from blaxel.core import settings
        from blaxel.core.sandbox.client import client
        from blaxel.core.sandbox.client.api.process import get_process_identifier, post_process
        from blaxel.core.sandbox.client.models import ErrorResponse, ProcessResponse

        try:
            client.with_base_url(self.sandbox.metadata.url)
            client.with_headers(settings.headers)

            # Install packages using pip via run_code
            self.logger.log(f"Installing packages: {', '.join(additional_imports)}", level=LogLevel.INFO)
            pip_install_code = f"pip install --root-user-action=ignore {' '.join(additional_imports)}"

            identifier = "install-packages"
            body = {
                "name": identifier,
                "command": pip_install_code,
            }
            post_process.sync(client=client, body=body)

            status = "running"
            interval = 1000
            max_wait = 600000
            start_time = time.time() * 1000
            logs = ""
            exit_code = 0

            while status == "running":
                if (time.time() * 1000) - start_time > max_wait:
                    raise Exception("Process did not finish in time")
                data = get_process_identifier.sync(identifier, client=client)
                if isinstance(data, ProcessResponse):
                    status = data.status or "running"
                    exit_code = data.exit_code
                    logs = data.logs
                elif isinstance(data, ErrorResponse):
                    raise Exception(f"Failed to install packages: {data.message}")
                else:
                    raise Exception(f"Unknown response: {data}")

                if status == "running":
                    time.sleep(interval / 1000)  # Convert to seconds

            if exit_code != 0:
                self.logger.log_error(f"Failed to install packages (exit code {exit_code}): {logs}")
                return []

            self.logger.log(f"Successfully installed packages: {', '.join(additional_imports)}", level=LogLevel.INFO)
            return additional_imports

        except Exception as e:
            self.logger.log_error(f"Error installing packages: {e}")
            return []

    def _delete_sandbox(self):
        """Delete sandbox using Blaxel's sync API and wait for completion."""
        from blaxel.core.client import client
        from blaxel.core.client.api.compute import delete_sandbox

        self.logger.log(f"Requesting sandbox {self.sandbox_name} deletion...", level=LogLevel.INFO)
        delete_sandbox.sync(client=client, sandbox_name=self.sandbox_name)

    def cleanup(self):
        """Sync wrapper to clean up sandbox and resources."""
        # Prevent double cleanup
        if self._cleaned_up:
            return
        self.logger.log("Shutting down sandbox...", level=LogLevel.INFO)
        self._cleaned_up = True
        try:
            self._delete_sandbox()
        except Exception as e:
            # Log cleanup errors but don't raise - cleanup should be best-effort
            self.logger.log(f"Error during cleanup: : {e}", level=LogLevel.INFO)
        finally:
            # Always clean up local references
            if hasattr(self, "sandbox"):
                del self.sandbox
            self.logger.log("Sandbox cleanup completed", level=LogLevel.INFO)

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Silently ignore errors during cleanup


class WasmExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in a sandboxed WebAssembly environment powered by Pyodide and Deno.

    This executor combines Deno's secure runtime with Pyodide's WebAssembly‑compiled Python interpreter to deliver s
    trong isolation guarantees while enabling full Python execution.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install in the Pyodide environment.
        logger (`Logger`): Logger to use for output and errors.
        allow_pickle (`bool`, default `False`): Whether to allow pickle serialization for objects that cannot be safely serialized to JSON.
            - `False` (default, recommended): Only safe JSON serialization is used. Raises error if object cannot be safely serialized.
            - `True` (legacy mode): Tries safe JSON serialization first, falls back to pickle with warning if needed.

            **Security Warning:** Pickle deserialization can execute arbitrary code. Only set `allow_pickle=True`
            if you fully trust the execution environment and need backward compatibility with custom types.
        deno_path (`str`, default `"deno"`): Path to the Deno executable. If not provided, will use "deno" from PATH.
        deno_permissions (`list[str]`, *optional*): List of permissions to grant to the Deno runtime.
            Default is minimal permissions needed for execution.
        timeout (`int`, default `60`): Timeout in seconds for code execution
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        deno_path: str = "deno",
        deno_permissions: list[str] | None = None,
        timeout: int = 60,
    ):
        super().__init__(additional_imports, logger, allow_pickle=allow_pickle)

        # Check if Deno is installed
        try:
            subprocess.run([deno_path, "--version"], capture_output=True, check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError(
                "Deno is not installed or not found in PATH. Please install Deno from https://deno.land/"
            )

        self.deno_path = deno_path
        self.timeout = timeout

        # Default minimal permissions needed
        if deno_permissions is None:
            # Use minimal permissions for Deno execution
            home_dir = os.getenv("HOME")
            deno_permissions = [
                "allow-net="
                + ",".join(
                    [
                        "0.0.0.0:8000",  # allow requests to the local server
                        "cdn.jsdelivr.net:443",  # allow loading pyodide packages
                        "pypi.org:443,files.pythonhosted.org:443",  # allow pyodide install packages from PyPI
                    ]
                ),
                f"allow-read={home_dir}/.cache/deno",
                f"allow-write={home_dir}/.cache/deno",
            ]
        self.deno_permissions = [f"--{perm}" for perm in deno_permissions]

        # Create the Deno JavaScript runner file
        self._create_deno_runner()

        # Install additional packages
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("WasmExecutor is running", level=LogLevel.INFO)

    def _create_deno_runner(self):
        """Create the Deno JavaScript file that will run Pyodide and execute Python code."""
        self.runner_dir = tempfile.mkdtemp(prefix="pyodide_deno_")
        self.runner_path = os.path.join(self.runner_dir, "pyodide_runner.js")

        # Create the JavaScript runner file
        with open(self.runner_path, "w") as f:
            f.write(self.JS_CODE)

        # Start the Deno server
        self._start_deno_server()

    def _start_deno_server(self):
        """Start the Deno server that will run our JavaScript code."""
        cmd = [self.deno_path, "run"] + self.deno_permissions + [self.runner_path]

        # Start the server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the server to start
        time.sleep(2)  # Give the server time to start

        # Check if the server started successfully
        if self.server_process.poll() is not None:
            stderr = self.server_process.stderr.read()
            raise RuntimeError(f"Failed to start Deno server: {stderr}")

        self.server_url = "http://localhost:8000"  # TODO: Another port?

        # Test the connection
        try:
            response = requests.get(self.server_url)
            if response.status_code != 200:
                raise RuntimeError(f"Server responded with status code {response.status_code}: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to Deno server: {e}")

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute Python code in the Pyodide environment and return the result.

        Args:
            code (`str`): Python code to execute.

        Returns:
            `CodeOutput`: Code output containing the result, logs, and whether it is the final answer.
        """
        try:
            # Prepare the request payload
            payload = {
                "code": code,
                "packages": self.installed_packages,
            }

            # Send the request to the Deno server
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                raise AgentError(f"Server error: {response.text}", self.logger)

            result = None
            is_final_answer = False

            # Parse the response
            result_data = response.json()

            # Process the result
            if result_data.get("result"):
                result = result_data.get("result")
            # Check for execution errors
            elif result_data.get("error"):
                error = result_data["error"]
                if (
                    error.get("pythonExceptionType") == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION
                    and "pythonExceptionValue" in error
                ):
                    result = self._deserialize_final_answer(error["pythonExceptionValue"], self.allow_pickle)
                    is_final_answer = True
                else:
                    error_message = f"{error.get('name', 'Error')}: {error.get('message', 'Unknown error')}"
                    if "stack" in error:
                        error_message += f"\n{error['stack']}"
                    raise AgentError(error_message, self.logger)

            # Get the execution logs
            execution_logs = result_data.get("stdout", "")

            # Handle image results
            if isinstance(result, dict) and result.get("type") == "image":
                image_data = result.get("data", "")
                decoded_bytes = base64.b64decode(image_data.encode("utf-8"))
                return PIL.Image.open(BytesIO(decoded_bytes)), execution_logs

            return CodeOutput(output=result, logs=execution_logs, is_final_answer=is_final_answer)

        except requests.RequestException as e:
            raise AgentError(f"Failed to communicate with Deno server: {e}", self.logger)

    def install_packages(self, additional_imports: list[str]) -> list[str]:
        """
        Install additional Python packages in the Pyodide environment.

        Args:
            additional_imports (`list[str]`): Package names to install.

        Returns:
            list[str]: Installed packages.
        """
        # In Pyodide, we don't actually install packages here, but we keep track of them
        # to load them when executing code
        # TODO: Install  here instead?
        self.logger.log(f"Adding packages to load: {', '.join(additional_imports)}", level=LogLevel.INFO)
        return additional_imports

    def cleanup(self):
        """Clean up resources used by the executor."""
        if hasattr(self, "server_process") and self.server_process:
            self.logger.log("Stopping Deno server...", level=LogLevel.INFO)
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

        # Remove the temporary directory
        if hasattr(self, "runner_dir") and os.path.exists(self.runner_dir):
            import shutil

            shutil.rmtree(self.runner_dir)

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    JS_CODE = dedent("""\
        // pyodide_runner.js - Runs Python code in Pyodide within Deno
        import { serve } from "https://deno.land/std/http/server.ts";
        import { loadPyodide } from "npm:pyodide";

        // Initialize Pyodide instance
        const pyodidePromise = loadPyodide();

        // Function to execute Python code and return the result
        async function executePythonCode(code) {
          const pyodide = await pyodidePromise;

          // Create a capture for stdout
          pyodide.runPython(`
            import sys
            import io
            sys.stdout = io.StringIO()
          `);

          // Execute the code and capture any errors
          let result = null;
          let error = null;
          let stdout = "";

          try {
            // Execute the code
            result = await pyodide.runPythonAsync(code);

            // Get captured stdout
            stdout = pyodide.runPython("sys.stdout.getvalue()");
          } catch (e) {
            error = {
              name: e.constructor.name,
              message: e.message,
              stack: e.stack
            };

            // Extract Python exception details
            if (e.constructor.name === "PythonError") {
              // Get the Python exception type from the error message: at the end of the traceback
              const errorMatch = e.message.match(/\\n([^:]+Exception): /);
              if (errorMatch) {
                error.pythonExceptionType = errorMatch[1].split(".").pop();
              }

              // If the error is a FinalAnswerException, extract its the encoded value
              if (error.pythonExceptionType === "FinalAnswerException") {
                // Extract the base64 encoded value from the error message
                const valueMatch = e.message.match(/FinalAnswerException: (.*?)(?:\\n|$)/);
                if (valueMatch) {
                  error.pythonExceptionValue = valueMatch[1];
                }
              }
            }
          }

          return {
            result,
            stdout,
            error
          };
        }

        // Start a simple HTTP server to receive code execution requests
        //const port = 8765;
        //console.log(`Starting Pyodide server on port ${port}`);

        serve(async (req) => {
          if (req.method === "POST") {
            try {
              const body = await req.json();
              const { code, packages = [] } = body;

              // Load any requested packages
              if (packages && packages.length > 0) {
                const pyodide = await pyodidePromise;
                //await pyodide.loadPackagesFromImports(code);
                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                try {
                  await micropip.install(packages);
                } catch (e) {
                  console.error(`Failed to load package ${packages}: ${e.message}`);
                }
              }

              const result = await executePythonCode(code);
              return new Response(JSON.stringify(result), {
                headers: { "Content-Type": "application/json" }
              });
            } catch (e) {
              return new Response(JSON.stringify({ error: e.message }), {
                status: 500,
                headers: { "Content-Type": "application/json" }
              });
            }
          }

          return new Response("Pyodide-Deno Executor is running. Send POST requests with code to execute.", {
            headers: { "Content-Type": "text/plain" }
          });
        });
        """)
