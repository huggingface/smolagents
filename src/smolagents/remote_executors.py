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
import pickle
import time
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import PIL.Image
import requests

from .default_tools import FinalAnswerTool
from .local_python_executor import CodeOutput, PythonExecutor
from .monitoring import LogLevel
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    FINAL_ANSWER_EXCEPTION = "FinalAnswerException"

    def __init__(self, additional_imports: list[str], logger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.installed_packages = []

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        """
        Execute code, return the result and output, also determining if
        the result is the final answer.
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
        if packages_to_install:
            self.installed_packages += self.install_packages(list(packages_to_install))
        # Get tool definitions
        code = get_tools_definition_code(tools)
        if code:
            code_output = self.run_code_raise_errors(code)
            self.logger.log(code_output.logs)

    def send_variables(self, variables: dict):
        """
        Send variables to the kernel namespace using pickle.
        """
        pickled_vars = base64.b64encode(pickle.dumps(variables)).decode()
        code = f"""
import pickle, base64
vars_dict = pickle.loads(base64.b64decode('{pickled_vars}'))
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
        a `FinalAnswerException` with the final answer as a pickled value.
        This allows the executor to catch this exception and return the final answer.

        Args:
            final_answer_tool (`FinalAnswerTool`): FinalAnswerTool instance to patch.
        """

        # Create a new class that inherits from the original FinalAnswerTool
        class _FinalAnswerTool(final_answer_tool.__class__):
            pass

        # Add a new forward method that raises the FinalAnswerException
        # - Define the new forward method function
        def forward(self, *args, **kwargs) -> Any:
            import base64
            import pickle

            class FinalAnswerException(Exception):
                def __init__(self, value):
                    self.value = value

            raise FinalAnswerException(base64.b64encode(pickle.dumps(self._forward(*args, **kwargs))).decode())

        # - Set the new forward method function to the _FinalAnswerTool class
        _FinalAnswerTool.forward = forward

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


class E2BExecutor(RemotePythonExecutor):
    """
    Executes Python code using E2B.

    Args:
        additional_imports (`list[str]`): Additional imports to install.
        logger (`Logger`): Logger to use.
        **kwargs: Additional arguments to pass to the E2B Sandbox.
    """

    def __init__(self, additional_imports: list[str], logger, **kwargs):
        super().__init__(additional_imports, logger)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        self.sandbox = Sandbox(**kwargs)
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running", level=LogLevel.INFO)

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        execution = self.sandbox.run_code(code)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])

        # Handle errors
        if execution.error:
            # Check if the error is a FinalAnswerException
            if execution.error.name == RemotePythonExecutor.FINAL_ANSWER_EXCEPTION:
                final_answer = pickle.loads(base64.b64decode(execution.error.value))
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


class DockerExecutor(RemotePythonExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        host: str = "127.0.0.1",
        port: int = 8888,
        image_name: str = "jupyter-kernel",
        build_new_image: bool = True,
        container_run_kwargs: dict[str, Any] | None = None,
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.

        Args:
            additional_imports: Additional imports to install.
            logger: Logger to use.
            host: Host to bind to.
            port: Port to bind to.
            image_name: Name of the Docker image to use. If the image doesn't exist, it will be built.
            build_new_image: If True, the image will be rebuilt even if it already exists.
            container_run_kwargs: Additional keyword arguments to pass to the Docker container run command.
        """
        super().__init__(additional_imports, logger)
        try:
            import docker
            from websocket import create_connection
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )
        self.host = host
        self.port = port
        self.image_name = image_name

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
                dockerfile_path = Path(__file__).parent / "Dockerfile"
                if not dockerfile_path.exists():
                    with open(dockerfile_path, "w") as f:
                        f.write(
                            dedent(
                                """\
                                FROM python:3.12-slim

                                RUN pip install jupyter_kernel_gateway jupyter_client

                                EXPOSE 8888
                                CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
                                """
                            )
                        )
                _, build_logs = self.client.images.build(
                    path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag=self.image_name
                )
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

            # Create new kernel via HTTP
            r = requests.post(f"{self.base_url}/api/kernels")
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
                self.logger.log_error(f"Failed to create kernel. Details: {json.dumps(error_details, indent=2)}")
                raise RuntimeError(f"Failed to create kernel: Status {r.status_code}\nResponse: {r.text}") from None

            self.kernel_id = r.json()["id"]

            ws_url = f"ws://{host}:{port}/api/kernels/{self.kernel_id}/channels"
            self.ws = create_connection(ws_url)

            self.installed_packages = self.install_packages(additional_imports)
            self.logger.log(
                f"Container {self.container.short_id} is running with kernel {self.kernel_id}", level=LogLevel.INFO
            )

        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def run_code_raise_errors(self, code_action: str) -> CodeOutput:
        try:
            # Send execute request
            msg_id = self._send_execute_request(code_action)

            # Collect output and results
            outputs = []
            result = None
            is_final_answer = False

            while True:
                msg = json.loads(self.ws.recv())
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
                        result = pickle.loads(base64.b64decode(msg_content.get("evalue", "")))
                        is_final_answer = True
                    else:
                        raise AgentError("\n".join(msg_content.get("traceback", [])), self.logger)
                elif msg_type == "status" and msg_content["execution_state"] == "idle":
                    break

            return CodeOutput(output=result, logs="".join(outputs), is_final_answer=is_final_answer)

        except Exception as e:
            self.logger.log_error(f"Code execution failed: {e}")
            raise

    def _send_execute_request(self, code: str) -> str:
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

        self.ws.send(json.dumps(execute_request))
        return msg_id

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


__all__ = ["E2BExecutor", "DockerExecutor"]
