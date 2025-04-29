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
import os
import pickle
import re
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import Any

import PIL.Image
import requests

from .local_python_executor import PythonExecutor
from .monitoring import LogLevel
from .tools import Tool, get_tools_definition_code
from .utils import AgentError


__all__ = ["E2BExecutor", "DockerExecutor"]


try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


class RemotePythonExecutor(PythonExecutor):
    def __init__(self, additional_imports: list[str], logger):
        self.additional_imports = additional_imports
        self.logger = logger
        self.logger.log("Initializing executor, hold on...")
        self.final_answer_pattern = re.compile(r"^final_answer\((.*)\)$", re.M)
        self.installed_packages = []

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> tuple[Any, str]:
        raise NotImplementedError

    def send_tools(self, tools: dict[str, Tool]):
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

    def __call__(self, code_action: str) -> tuple[Any, str, bool]:
        """Check if code is a final answer and run it accordingly"""
        is_final_answer = bool(self.final_answer_pattern.search(code_action))
        output = self.run_code_raise_errors(code_action, return_final_answer=is_final_answer)
        return output[0], output[1], is_final_answer

    def install_packages(self, additional_imports: list[str]):
        additional_imports = additional_imports + ["smolagents"]
        _, execution_logs = self.run_code_raise_errors(f"!pip install {' '.join(additional_imports)}")
        self.logger.log(execution_logs)
        return additional_imports


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

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> tuple[Any, str]:
        execution = self.sandbox.run_code(
            code,
        )
        if execution.error:
            execution_logs = "\n".join([str(log) for log in execution.logs.stdout])
            logs = execution_logs
            logs += "Executing code yielded an error:"
            logs += execution.error.name + "\n"
            logs += execution.error.value
            logs += execution.error.traceback
            raise AgentError(logs, self.logger)
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
                            return PIL.Image.open(BytesIO(decoded_bytes)), execution_logs
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
                raise AgentError("No main result returned by executor!", self.logger)
            return None, execution_logs


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
                        f.write("""FROM python:3.12-slim

RUN pip install jupyter_kernel_gateway requests numpy pandas
RUN pip install jupyter_client notebook

EXPOSE 8888
CMD ["jupyter", "kernelgateway", "--KernelGatewayApp.ip='0.0.0.0'", "--KernelGatewayApp.port=8888", "--KernelGatewayApp.allow_origin='*'"]
""")
                _, build_logs = self.client.images.build(
                    path=str(dockerfile_path.parent), dockerfile=str(dockerfile_path), tag=self.image_name
                )
                self.logger.log(build_logs, level=LogLevel.DEBUG)

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

    def run_code_raise_errors(self, code_action: str, return_final_answer: bool = False) -> tuple[Any, str]:
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
                    raise AgentError("\n".join(traceback), self.logger)
                elif msg_type == "status" and msg["content"]["execution_state"] == "idle":
                    if not return_final_answer or waiting_for_idle:
                        break

            return result, "".join(outputs)

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
        """Clean up resources."""
        try:
            if hasattr(self, "container"):
                self.logger.log(f"Stopping and removing container {self.container.short_id}...", level=LogLevel.INFO)
                self.container.stop()
                self.container.remove()
                self.logger.log("Container cleanup completed", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log_error(f"Error during cleanup: {e}")

    def delete(self):
        """Ensure cleanup on deletion."""
        self.cleanup()


class WebAssemblyExecutor(RemotePythonExecutor):
    """
    Remote Python code executor in a sandboxed WebAssembly environment powered by Pyodide and Deno.

    This executor combines Deno's secure runtime with Pyodide's WebAssembly‑compiled Python interpreter to deliver s
    trong isolation guarantees while enabling full Python execution.

    Args:
        additional_imports (`list[str]`): Additional Python packages to install in the Pyodide environment.
        logger (`Logger`): Logger to use for output and errors.
        deno_path (`str`, optional): Path to the Deno executable. If not provided, will use "deno" from PATH.
        deno_permissions (`list[str]`, optional): List of permissions to grant to the Deno runtime.
            Default is minimal permissions needed for execution.
        timeout (`int`, optional): Timeout in seconds for code execution. Default is 60 seconds.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        deno_path: str = "deno",
        deno_permissions: list[str] | None = None,
        timeout: int = 60,
    ):
        super().__init__(additional_imports, logger)

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
            # TODO: Set minimal permissions
            self.deno_permissions = [
                "--allow-net=cdn.jsdelivr.net,0.0.0.0:8000",  # allow fetch pyodide packages & server
                "--allow-read",  # grant read access for pyodide packages
                "--allow-write",  # grant write access for pyodide packages
            ]
        else:
            self.deno_permissions = [f"--{perm}" for perm in deno_permissions]

        # Create the Deno JavaScript runner file
        self._create_deno_runner()

        # Install additional packages
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log("WebAssemblyExecutor is running", level=LogLevel.INFO)

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

    def run_code_raise_errors(self, code: str, return_final_answer: bool = False) -> tuple[Any, str]:
        """
        Execute Python code in the Pyodide environment and return the result.

        Args:
            code (`str`): Python code to execute.
            return_final_answer (`bool`, default `False`): Whether to extract and return the final answer.

        Returns:
            tuple[Any, str]: A tuple containing the result and execution logs.
        """
        try:
            # Prepare the request payload
            payload = {
                "code": code,
                "returnFinalAnswer": return_final_answer,
                "packages": self.installed_packages,
            }

            # Send the request to the Deno server
            response = requests.post(self.server_url, json=payload, timeout=self.timeout)

            if response.status_code != 200:
                raise AgentError(f"Server error: {response.text}", self.logger)

            # Parse the response
            result_data = response.json()

            # Check for execution errors
            if result_data.get("error"):
                error = result_data["error"]
                error_message = f"{error.get('name', 'Error')}: {error.get('message', 'Unknown error')}"
                if "stack" in error:
                    error_message += f"\n{error['stack']}"
                raise AgentError(error_message, self.logger)

            # Get the execution logs
            execution_logs = result_data.get("stdout", "")

            # Process the result
            result = result_data.get("result")

            # Handle image results
            if isinstance(result, dict) and result.get("type") == "image":
                image_data = result.get("data", "")
                decoded_bytes = base64.b64decode(image_data.encode("utf-8"))
                return PIL.Image.open(BytesIO(decoded_bytes)), execution_logs

            return result, execution_logs

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

    def send_tools(self, tools: dict[str, Tool]):
        for tool in tools.values():
            for package in tool.to_dict()["requirements"]:
                if package not in self.installed_packages:
                    self.installed_packages.append(package)
        tool_definition_code = get_tools_definition_code(tools)
        # TODO: Run it now or later, preceding code?
        execution = self.run_code_raise_errors(tool_definition_code)
        self.logger.log(execution[1])

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
        async function executePythonCode(code, returnFinalAnswer = false) {
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
            if (returnFinalAnswer) {
              // Extract the final_answer call if present
              const finalAnswerMatch = code.match(/final_answer\\s*\\((.*)\\)/);
              if (finalAnswerMatch) {
                // Execute the code up to the final_answer call
                const preCode = code.replace(/final_answer\\s*\\(.*\\)/, "");
                pyodide.runPython(preCode);

                // Execute the final_answer expression and get the result
                const finalAnswerExpr = finalAnswerMatch[1];
                result = pyodide.runPython(`${finalAnswerExpr}`);

                // Handle image results
                if (result && result.constructor.name === "Image") {
                  // Convert PIL Image to base64
                  const pngBytes = pyodide.runPython(`
                    import io
                    import base64
                    buf = io.BytesIO()
                    _result.save(buf, format='PNG')
                    base64.b64encode(buf.getvalue()).decode('utf-8')
                  `);
                  result = { type: "image", data: pngBytes };
                }
              }
            } else {
              // Just run the code without expecting a final answer
              result = pyodide.runPython(code);
            }

            // Get captured stdout
            stdout = pyodide.runPython("sys.stdout.getvalue()");
          } catch (e) {
            error = {
              name: e.constructor.name,
              message: e.message,
              stack: e.stack
            };
          }

          return {
            result: result,
            stdout: stdout,
            error: error
          };
        }

        // Start a simple HTTP server to receive code execution requests
        //const port = 8765;
        //console.log(`Starting Pyodide server on port ${port}`);

        serve(async (req) => {
          if (req.method === "POST") {
            try {
              const body = await req.json();
              const { code, returnFinalAnswer = false, packages = [] } = body;

              // Load any requested packages
              if (packages && packages.length > 0) {
                const pyodide = await pyodidePromise;
                //await pyodide.loadPackagesFromImports(code);
                await pyodide.loadPackage("micropip");
                const micropip = pyodide.pyimport("micropip");
                for (const pkg of packages) {
                  try {
                    // await pyodide.loadPackage(pkg);
                    await micropip.install(pkg);
                  } catch (e) {
                    console.error(`Failed to load package ${pkg}: ${e.message}`);
                  }
                }
              }

              const result = await executePythonCode(code, returnFinalAnswer);
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
