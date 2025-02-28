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
import json
import pickle
import re
import time
import uuid
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple, Optional

import requests
from PIL import Image

from .local_python_executor import PythonExecutor
from .monitoring import AgentLogger, LogLevel
from .tools import Tool, get_tools_definition_code


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    def __init__(self, additional_imports: List[str], logger: AgentLogger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$", re.M)
        self.installed_packages = []

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        raise NotImplementedError

    def send_tools(self, tools: Dict[str, Tool]):
        tool_definition_code = get_tools_definition_code(tools)

        packages_to_install = set()
        for tool in tools.values():
            for package in tool.to_dict()["requirements"]:
                if package not in self.installed_packages:
                    packages_to_install.add(package)
                    self.installed_packages.append(package)

        execution = self.run_code_raise_errors(
            f"!pip install {' '.join(packages_to_install)}\n" + tool_definition_code
        )
        self.logger.log(execution[1])

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

    def __call__(self, code_action: str) -> Tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        is_final_answer = bool(self.final_answer_pattern.search(code_action))
        output = self.run_code_raise_errors(code_action, return_final_answer=is_final_answer)
        return output[0], output[1], is_final_answer

    def install_packages(self, additional_imports: List[str]):
        additional_imports = additional_imports + ["smolagents"]
        self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
        return additional_imports


class E2BExecutor(RemotePythonExecutor):
    def __init__(self, additional_imports: List[str], logger: AgentLogger):
        super().__init__(additional_imports, logger)
        try:
            from e2b_code_interpreter import Sandbox
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                """Please install 'e2b' extra to use E2BExecutor: `pip install 'smolagents[e2b]'`"""
            )
        self.sandbox = Sandbox()
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("E2B is running")

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        execution = self.sandbox.run_code(
            code,
        )
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs
            logs += "Executing code yielded an error:"
            logs += execution.error.name
            logs += execution.error.value
            logs += execution.error.traceback
            raise ValueError(logs)
        self.logger.log(execution.logs)
        execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
        if not execution.results:
            return None, execution_logs
        else:
            for result in execution.results:
                if result.is_main_result:
                    for attribute_name in ["jpeg", "png"]:
                        if getattr(result, attribute_name) is not None:
                            image_output = getattr(result, attribute_name)
                            decoded_bytes = base64.b64decode(image_output.encode("utf-8"))
                            return Image.open(BytesIO(decoded_bytes)), execution_logs
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
                        if getattr(result, attribute_name) is not None:
                            return getattr(result, attribute_name), execution_logs
            if return_final_answer:
                raise ValueError("No main result returned by executor!")
            return None, execution_logs


class ContainerAbstractExecutor(RemotePythonExecutor):
    """
    Abstract base class for container-based executors (Docker/Podman).
    Implements common functionality for container management and code execution.
    """

    def __init__(
        self,
        additional_imports: List[str],
        logger: AgentLogger,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the container-based Jupyter Kernel Gateway executor.
        """
        super().__init__(additional_imports, logger)
        try:
            from websocket import create_connection
            self.websocket_module = create_connection
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install the required dependencies: `pip install 'smolagents[docker]'`"
            )
        
        self.host = host
        self.port = port
        self.client = None
        self.container = None
        self.kernel_id = None
        self.ws = None
        self.base_url = f"http://{host}:{port}"

    def _initialize_container(self):
        """To be implemented by subclasses to initialize container client."""
        raise NotImplementedError("Subclasses must implement _initialize_container method")

    def _build_image(self):
        """To be implemented by subclasses to build container image."""
        raise NotImplementedError("Subclasses must implement _build_image method")

    def _start_container(self):
        """To be implemented by subclasses to start container."""
        raise NotImplementedError("Subclasses must implement _start_container method")

    def _setup_dockerfile(self):
        """Create Dockerfile if it doesn't exist."""
        dockerfile_path = Path(__file__).parent / "Dockerfile"
        if not dockerfile_path.exists():
            with open(dockerfile_path, "w") as f:
                f.write("""FROM python:3.12-slim

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
""")
        return dockerfile_path

    def wait_for_service(self, max_retries=15, initial_delay=2.0):
        """Wait for the Jupyter Kernel Gateway service to be ready."""
        self.logger.log("Waiting for Jupyter Kernel Gateway service to start...")
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                # Try to access the API endpoint to check if service is running
                response = requests.get(f"{self.base_url}/api/kernelspecs", timeout=5)
                if response.status_code == 200:
                    self.logger.log(f"Service is ready after {attempt + 1} attempts")
                    return True
                else:
                    self.logger.log(f"Service not ready yet (status code: {response.status_code}). Retrying...")
            except requests.exceptions.RequestException as e:
                self.logger.log(f"Attempt {attempt + 1}/{max_retries}: Service not ready yet: {str(e)[:100]}...")

            # Exponential backoff with jitter
            import random

            delay = min(delay * 1.5, 10.0)  # Cap max delay at 10 seconds
            jitter = random.uniform(0, 0.1 * delay)  # Add up to 10% jitter
            time_to_sleep = delay + jitter
            self.logger.log(f"Waiting {time_to_sleep:.2f} seconds before next attempt...")
            time.sleep(time_to_sleep)

        # Log container logs to help diagnose issues
        try:
            logs = self.container.logs().decode("utf-8")
            self.logger.log(f"Container logs:\n{logs}")
        except Exception as e:
            self.logger.log(f"Failed to get container logs: {e}")

        raise RuntimeError(f"Service failed to become ready after {max_retries} attempts")

    def create_kernel(self, max_retries=5):
        """Create a kernel with retry mechanism."""
        self.logger.log("Creating a new kernel...")
        delay = 1.0

        for attempt in range(max_retries):
            try:
                r = requests.post(f"{self.base_url}/api/kernels", timeout=10)
                if r.status_code == 201:
                    self.kernel_id = r.json()["id"]
                    self.logger.log(f"Kernel created successfully with ID: {self.kernel_id}")
                    return
                else:
                    error_details = {"status_code": r.status_code, "body": r.text}
                    self.logger.log(
                        f"Attempt {attempt + 1}/{max_retries}: Failed to create kernel: {json.dumps(error_details)}"
                    )
            except requests.exceptions.RequestException as e:
                self.logger.log(f"Attempt {attempt + 1}/{max_retries}: Request failed: {str(e)[:100]}...")

            # Exponential backoff with jitter
            import random

            delay = min(delay * 1.5, 10.0)
            jitter = random.uniform(0, 0.1 * delay)
            time_to_sleep = delay + jitter
            self.logger.log(f"Waiting {time_to_sleep:.2f} seconds before next attempt...")
            time.sleep(time_to_sleep)

        raise RuntimeError(f"Failed to create kernel after {max_retries} attempts")

    def _send_execute_request(self, code: str) -> str:
        """Send code execution request to kernel."""
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

    def run_code_raise_errors(self, code_action: str, return_final_answer: bool = False) -> Tuple[Any, str]:
        """
        Execute code and return result based on whether it's a final answer.
        """
        try:
            if return_final_answer:
                match = self.final_answer_pattern.search(code_action)
                if match:
                    pre_final_answer_code = self.final_answer_pattern.sub("", code_action)
                    result_expr = match.group(1)
                    wrapped_code = pre_final_answer_code + dedent(f"""
                        import pickle, base64
                        _result = {result_expr}
                        print("RESULT_PICKLE:" + base64.b64encode(pickle.dumps(_result)).decode())
                        """)
            else:
                wrapped_code = code_action

            # Send execute request
            msg_id = self._send_execute_request(wrapped_code)

            # Collect output and results
            outputs = []
            result = None
            waiting_for_idle = False

            while True:
                msg = json.loads(self.ws.recv())
                msg_type = msg.get("msg_type", "")
                parent_msg_id = msg.get("parent_header", {}).get("msg_id")

                # Only process messages related to our execute request
                if parent_msg_id != msg_id:
                    continue

                if msg_type == "stream":
                    text = msg["content"]["text"]
                    if return_final_answer and text.startswith("RESULT_PICKLE:"):
                        pickle_data = text[len("RESULT_PICKLE:") :].strip()
                        result = pickle.loads(base64.b64decode(pickle_data))
                        waiting_for_idle = True
                    else:
                        outputs.append(text)
                elif msg_type == "error":
                    traceback = msg["content"].get("traceback", [])
                    raise RuntimeError("\n".join(traceback)) from None
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    if not return_final_answer or waiting_for_idle:
                        break

            return result, "".join(outputs)

        except Exception as e:
            self.logger.log_error(f"Code execution failed: {e}")
            raise
            
    def setup(self):
        """Initialize and start the container environment."""
        try:
            self._initialize_container()
            dockerfile_path = self._setup_dockerfile()
            self._build_image(dockerfile_path)
            self._start_container()
            
            # Wait for services and setup connection
            self.wait_for_service()
            self.create_kernel()
            
            # Initialize WebSocket connection
            ws_url = f"ws://{self.host}:{self.port}/api/kernels/{self.kernel_id}/channels"
            self.ws = self.websocket_module(ws_url)
            
            # Install packages
            self.installed_packages = self.install_packages(self.additional_imports)
            self.logger.log(f"Container {self.container.short_id} is running with kernel {self.kernel_id}")
            
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to initialize Jupyter kernel: {e}") from e

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "container") and self.container:
                self.logger.log(f"Stopping and removing container {self.container.short_id}...")
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed")
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class DockerExecutor(ContainerAbstractExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Docker container.
    """

    def __init__(
        self,
        additional_imports: List[str],
        logger: AgentLogger,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the Docker-based Jupyter Kernel Gateway executor.
        """
        super().__init__(additional_imports, logger, host, port)
        self.setup()

    def _initialize_container(self):
        """Initialize Docker client."""
        try:
            import docker
            self.client = docker.from_env()
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'docker' extra to use DockerExecutor: `pip install 'smolagents[docker]'`"
            )
        except Exception as e:
            raise RuntimeError("Could not connect to Docker daemon: make sure Docker is running.") from e

    def _build_image(self, dockerfile_path):
        """Build Docker image."""
        self.logger.log("Building Docker image...")
        try:
            _, build_logs = self.client.images.build(
                path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag="jupyter-kernel"
            )
            self.logger.log(build_logs, LogLevel.DEBUG)
        except Exception as e:
            raise RuntimeError(f"Failed to build Docker image: {e}") from e

    def _start_container(self):
        """Start Docker container."""
        self.logger.log(f"Starting container on {self.host}:{self.port}...")
        try:
            self.container = self.client.containers.run(
                "jupyter-kernel", ports={"8888/tcp": (self.host, self.port)}, detach=True
            )
            
            retries = 0
            while retries < 5:
                self.container.reload()
                if self.container.status == "running":
                    break
                self.logger.log(f"Container status: {self.container.status}, waiting...")
                time.sleep(1)
                retries += 1
                
            if self.container.status != "running":
                raise RuntimeError(f"Container failed to start. Status: {self.container.status}")
        except Exception as e:
            raise RuntimeError(f"Failed to start Docker container: {e}") from e


class PodmanExecutor(ContainerAbstractExecutor):
    """
    Executes Python code using Jupyter Kernel Gateway in a Podman container.
    """

    def __init__(
        self,
        additional_imports: List[str],
        logger: AgentLogger,
        host: str = "127.0.0.1",
        port: int = 8888,
    ):
        """
        Initialize the Podman-based Jupyter Kernel Gateway executor.
        """
        super().__init__(additional_imports, logger, host, port)
        self.setup()

    def _initialize_container(self):
        """Initialize Podman client with multiple socket URL attempts."""
        try:
            import os
            import podman
            import podman.errors
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'podman' extra to use PodmanExecutor: `pip install 'smolagents[docker]'`"
            )

        # Try different socket URLs
        socket_urls = [
            "unix:///run/podman/podman.sock",  # Default
            "unix:///run/user/{}/podman/podman.sock".format(os.getuid()),  # Rootless
            "http://localhost:8080",  # TCP if configured
        ]

        connection_error = None
        for url in socket_urls:
            try:
                self.client = podman.PodmanClient(base_url=url)
                # Test connection
                self.client.ping()
                self.logger.log(f"Successfully connected to Podman at {url}")
                return
            except Exception as e:
                connection_error = e
                continue

        if self.client is None:
            raise RuntimeError(
                f"Failed to connect to Podman. Tried URLs: {socket_urls}. Last error: {connection_error}"
            )

    def _build_image(self, dockerfile_path):
        """Build Podman image if it doesn't exist."""
        self.logger.log("Building Podman image...")
        try:
            if self.client.images.exists("jupyter-kernel"):
                self.logger.log("Image already exists, skipping creation")
            else:
                _, build_logs = self.client.images.build(
                    path=str(dockerfile_path.parent),
                    dockerfile=str(dockerfile_path),
                    tag="jupyter-kernel",
                    rm=True,
                    pull=True,
                    forcerm=True,
                    buildargs={},
                )
                self.logger.log(build_logs, LogLevel.DEBUG)
        except Exception as e:
            raise RuntimeError(f"Failed to build Podman image: {e}") from e

    def _start_container(self):
        """Start Podman container."""
        self.logger.log(f"Starting container on {self.host}:{self.port}...")
        try:
            self.container = self.client.containers.run(
                "jupyter-kernel", 
                ports={"8888/tcp": (self.host, self.port)}, 
                detach=True, 
                cap_drop=["ALL"]
            )
            self.logger.log(f"Container started with ID: {self.container.short_id}")
            
            # Wait for container to be in running state
            retries = 0
            while retries < 5:
                self.container.reload()
                if self.container.status == "running":
                    break
                self.logger.log(f"Container status: {self.container.status}, waiting...")
                time.sleep(1)
                retries += 1
                
            if self.container.status != "running":
                raise RuntimeError(f"Container failed to start. Status: {self.container.status}")
        except Exception as e:
            raise RuntimeError(f"Failed to start Podman container: {e}") from e


__all__ = ["E2BExecutor", "DockerExecutor", "PodmanExecutor", "ContainerAbstractExecutor"]