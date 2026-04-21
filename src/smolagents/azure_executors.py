#!/usr/bin/env python
# coding=utf-8

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from __future__ import annotations

import base64
import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import requests

from .local_python_executor import CodeOutput
from .monitoring import LogLevel
from .remote_executors import RemotePythonExecutor
from .utils import AgentError


__all__ = ["AzureDynamicSessionsExecutor"]


API_VERSION = "2024-10-02-preview"
SESSION_VOLUME_PATH = "/mnt/data"


def _default_token_provider_factory() -> Callable[[], str]:
    try:
        from azure.core.credentials import AccessToken
        from azure.identity import DefaultAzureCredential
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Please run: `pip install 'smolagents[azure]'` to use AzureDynamicSessionsExecutor"
        ) from error

    cached_token: AccessToken | None = None

    def _provide() -> str:
        nonlocal cached_token
        if cached_token is None or datetime.fromtimestamp(cached_token.expires_on, timezone.utc) < datetime.now(
            timezone.utc
        ) + timedelta(minutes=5):
            cached_token = DefaultAzureCredential().get_token("https://dynamicsessions.io/.default")
        return cached_token.token

    return _provide


class AzureDynamicSessionsExecutor(RemotePythonExecutor):
    """Execute Python code in Azure Container Apps Dynamic Sessions.

    Works with both the built-in code interpreter pool and custom container
    session pools.  Requires ``azure-identity`` at runtime and the calling
    identity to hold the *Azure ContainerApps Session Executor* role on the
    session pool resource.

    Args:
        additional_imports: Packages to ``pip install`` inside the session.
        logger: smolagents logger instance.
        pool_management_endpoint: Base URL of the session pool
            (e.g. ``https://pool.env.region.azurecontainerapps.io``).
        session_id: Reuse a specific session.  A random UUID is generated
            when omitted.
        access_token_provider: ``() -> str`` callable that returns a valid
            Entra bearer token.  Defaults to ``DefaultAzureCredential``.
    """

    def __init__(
        self,
        additional_imports: list[str],
        logger,
        allow_pickle: bool = False,
        *,
        pool_management_endpoint: str,
        session_id: str | None = None,
        access_token_provider: Callable[[], str] | None = None,
        api_version: str = API_VERSION,
    ):
        super().__init__(additional_imports, logger, allow_pickle)
        if not pool_management_endpoint:
            raise ValueError(
                "pool_management_endpoint is required. Set AZURE_SESSIONS_POOL_ENDPOINT in your environment."
            )
        self._endpoint = pool_management_endpoint.rstrip("/")
        self._session_id = session_id or str(uuid4())
        self._api_version = api_version
        self._token_provider = access_token_provider or _default_token_provider_factory()
        self.installed_packages = self.install_packages(additional_imports)
        self.logger.log(
            f"Azure Dynamic Sessions executor ready (session={self._session_id})",
            level=LogLevel.INFO,
        )

    def _build_url(self, path: str) -> str:
        qs = urllib.parse.urlencode(
            {
                "identifier": self._session_id,
                "api-version": self._api_version,
            }
        )
        return f"{self._endpoint}/{path}?{qs}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token_provider()}",
            "Content-Type": "application/json",
        }

    def run_code_raise_errors(self, code: str) -> CodeOutput:
        url = self._build_url("code/execute")
        body: dict[str, Any] = {
            "properties": {
                "codeInputType": "inline",
                "executionType": "synchronous",
                "code": code,
            }
        }
        resp = requests.post(url, headers=self._headers(), json=body, timeout=230)
        resp.raise_for_status()

        props = resp.json().get("properties", {})
        stdout = props.get("stdout", "")
        stderr = props.get("stderr", "")
        result = props.get("result")
        logs = stdout
        if stderr:
            logs = f"{stdout}\n{stderr}" if stdout else stderr

        if stderr and self.FINAL_ANSWER_EXCEPTION in stderr:
            try:
                value_line = stderr.split(self.FINAL_ANSWER_EXCEPTION)[-1].strip()
                final_answer = self._deserialize_final_answer(
                    value_line,
                    allow_pickle=self.allow_pickle,
                )
                return CodeOutput(output=final_answer, logs=logs, is_final_answer=True)
            except Exception:
                pass

        if stderr and "Traceback" in stderr:
            raise AgentError(f"{logs}\nExecuting code yielded an error:\n{stderr}", self.logger)

        output = self._parse_result(result) if result is not None else None
        return CodeOutput(output=output, logs=logs, is_final_answer=False)

    @staticmethod
    def _parse_result(result: Any) -> Any:
        if isinstance(result, dict):
            rtype = result.get("type")
            if rtype == "image" and "base64_data" in result:
                from io import BytesIO

                import PIL.Image

                return PIL.Image.open(BytesIO(base64.b64decode(result["base64_data"])))
            if "value" in result:
                return result["value"]
        if isinstance(result, str):
            try:
                return json.loads(result)
            except (json.JSONDecodeError, ValueError):
                pass
        return result

    def upload_file(self, local_path: str, remote_path: str | None = None) -> dict:
        remote_name = remote_path or Path(local_path).name
        url = self._build_url("files") + f"&path={urllib.parse.quote(SESSION_VOLUME_PATH)}"
        headers = {k: v for k, v in self._headers().items() if k != "Content-Type"}
        with open(local_path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                files=[("file", (remote_name, f, "application/octet-stream"))],
                timeout=120,
            )
        resp.raise_for_status()
        return resp.json()

    def download_file(self, remote_path: str) -> bytes:
        encoded = urllib.parse.quote(remote_path)
        url = self._build_url(f"files/{encoded}/content")
        resp = requests.get(url, headers=self._headers(), timeout=120)
        resp.raise_for_status()
        return resp.content

    def list_files(self) -> list[dict]:
        url = self._build_url("files")
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        return resp.json().get("value", [])

    def cleanup(self):
        try:
            self.logger.log(
                f"Deleting Azure Dynamic Session {self._session_id}...",
                level=LogLevel.INFO,
            )
            response = requests.delete(
                self._build_url("sessions"),
                headers=self._headers(),
                timeout=10,
            )
            if response.status_code not in {200, 202, 204, 404}:
                response.raise_for_status()
            self.logger.log("Azure session cleanup completed", level=LogLevel.INFO)
        except Exception as e:
            self.logger.log_error(
                f"Error during Azure session cleanup: {e}. Session will auto-expire if still active."
            )
