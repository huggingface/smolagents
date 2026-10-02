# coding=utf-8
# Copyright 2024 HuggingFace Inc.
from typing import Any

from smolagents.tools import Tool, get_tools_definition_code


class TestGetToolsDefinitionCodeInitParams:
    def _make_api_tool_class(self):
        class ApiClientTool(Tool):
            name = "api_client_tool"
            description = "Calls a remote API with a configured base URL."
            inputs = {"query": {"type": "string", "description": "Lookup query"}}
            output_type = "string"

            def __init__(self, api_url: str = "https://example.com", timeout: int = 30, **kwargs):
                super().__init__()
                self.api_url = api_url
                self.timeout = timeout
                self.extra = kwargs

            def forward(self, query: str) -> str:
                return f"{self.api_url}:{self.timeout}:{query}"

        return ApiClientTool

    def test_default_init_still_emits_empty_constructor(self):
        tool_cls = self._make_api_tool_class()
        code = get_tools_definition_code({"api_client_tool": tool_cls()})
        assert "api_client_tool = ApiClientTool()" in code
        assert "api_url=" not in code.split("api_client_tool = ", 1)[1].split("\n", 1)[0]

    def test_custom_init_params_are_rebuilt_in_remote_source(self):
        tool_cls = self._make_api_tool_class()
        tool = tool_cls(api_url="https://soc.example", timeout=5, region="eu")
        code = get_tools_definition_code({"api_client_tool": tool})

        assert "api_client_tool = ApiClientTool(api_url='https://soc.example', timeout=5, region='eu')" in code

        remote_scope: dict[str, Any] = {}
        exec(code, remote_scope, remote_scope)
        rebuilt = remote_scope["api_client_tool"]
        assert rebuilt.api_url == "https://soc.example"
        assert rebuilt.timeout == 5
        assert rebuilt.extra == {"region": "eu"}
        assert rebuilt.forward("alert") == "https://soc.example:5:alert"

    def test_non_serializable_init_values_are_omitted(self):
        class ClientHolderTool(Tool):
            name = "client_holder_tool"
            description = "Holds a live client plus serializable config."
            inputs = {"query": {"type": "string", "description": "Lookup query"}}
            output_type = "string"

            def __init__(self, api_url: str = "https://example.com", client: object | None = None):
                super().__init__()
                self.api_url = api_url
                self.client = client

            def forward(self, query: str) -> str:
                return f"{self.api_url}:{query}"

        tool = ClientHolderTool(api_url="https://nvd.example", client=object())
        code = get_tools_definition_code({"client_holder_tool": tool})
        instantiation = [line for line in code.splitlines() if line.startswith("client_holder_tool = ")][0]
        assert "api_url='https://nvd.example'" in instantiation
        assert "client=" not in instantiation
