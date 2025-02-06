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
import textwrap
import hashlib
import time
from typing import Any, List, Tuple

from dotenv import load_dotenv
from rizaio import Riza

from .tool_validation import validate_tool_attributes
from .tools import Tool
from .utils import BASE_BUILTIN_MODULES, instance_to_source
from .monitoring import AgentLogger, LogLevel

load_dotenv()


class RizaExecutor:
    def __init__(self, additional_imports: List[str], tools: dict[str, Tool], logger: AgentLogger):
        self.logger = logger
        self.riza_client = Riza()

        tool_codes = []
        for tool in tools.values():
            validate_tool_attributes(tool.__class__, check_imports=False)
            tool_code = instance_to_source(tool, base_cls=Tool)
            tool_code = tool_code.replace("from smolagents.tools import Tool", "")
            tool_code += f"\n{tool.name} = {tool.__class__.__name__}()\n"
            tool_codes.append(tool_code)

        tool_definition_code = "\n".join(
            [f"import {module}" for module in BASE_BUILTIN_MODULES]
        )
        tool_definition_code += textwrap.dedent("""
        class Tool:
            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def forward(self, *args, **kwargs):
                pass # to be implemented in child class
        """)
        tool_definition_code += "\n\n".join(tool_codes)

        self.tool_definition_code = tool_definition_code

        if len(additional_imports) > 0:
            additional_imports.sort()
            requirements = "\n".join(additional_imports)
            requirements_hash = hashlib.md5(requirements.encode()).hexdigest()
            runtime_name = f"smolagents-{requirements_hash}"

            self.logger.log("Looking for a custom Riza runtime with the requested additional packages", LogLevel.INFO)
            runtimes_resp = self.riza_client.runtimes.list()
            for runtime in runtimes_resp.runtimes:
                if runtime.name == runtime_name:
                    self.runtime_revision_id = runtime.revision_id
                    return

            self.logger.log("Building a custom Riza runtime with the requested additional packages (this may take a bit, but you only need to do it once)", LogLevel.INFO)
            create_runtime_resp = self.riza_client.runtimes.create(
                name=runtime_name,
                language="python",
                manifest_file={
                    "name": "requirements.txt",
                    "contents": requirements,
                },
            )

            self.runtime_revision_id = create_runtime_resp.revision_id

            while True:
                get_runtime_resp = self.riza_client.runtimes.get(id=create_runtime_resp.id)
                status = dict(get_runtime_resp)["status"]

                if status in ["succeeded", "failed"]:
                    self.logger.log(f"Riza custom runtime build \"{status}\"")
                    return
                
                self.logger.log(f"Riza custom runtime build status is \"{status}\"...")

                time.sleep(3)

    def run_code_raise_errors(self, code: str, input: dict = None):
        munged_code = code.replace("final_answer(", "return final_answer(")
        wrapped_code = f"""
{self.tool_definition_code}
def execute(input):
{textwrap.indent(munged_code, "    ")}
        """

        response = self.riza_client.command.exec_func(
            runtime_revision_id=self.runtime_revision_id,
            language="python",
            code=wrapped_code,
            input=input,
            http={"allow": [{"host": "*"}]}
        )
        if response.output_status != "valid":
            if response.output_status == "json_serialization_error":
                raise ValueError("Executing code returned an invalid object") # TODO better message
            logs = "\n".join([
                response.execution.stdout,
                "Executing code yielded an error:",
                response.execution.stderr
            ])
            raise ValueError(logs)
        return response

    def __call__(self, code: str, additional_args: dict) -> Tuple[Any, Any]:
        response = self.run_code_raise_errors(code, additional_args)
        is_final_answer = "final_answer(" in code
        return response.output, response.execution.stdout, is_final_answer


__all__ = ["RizaExecutor"]
