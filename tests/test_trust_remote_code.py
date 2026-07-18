import pytest
from smolagents.tools import Tool
from smolagents.agents import MultiStepAgent


class TestTrustRemoteCode:
    """Tests verifying trust_remote_code is required for code execution paths."""

    def test_from_code_requires_trust_remote_code(self):
        """Tool.from_code() should reject code execution without explicit opt-in."""
        with pytest.raises(ValueError, match="trust_remote_code"):
            Tool.from_code("print('pwned')")

    def test_from_dict_requires_trust_remote_code(self):
        """Tool.from_dict() should reject code execution without explicit opt-in."""
        tool_dict = {
            "name": "test_tool",
            "code": "print('pwned')",
            "description": "test",
            "inputs": {},
            "output_type": "string",
        }
        with pytest.raises(ValueError, match="trust_remote_code"):
            Tool.from_dict(tool_dict)

    def test_from_code_allows_execution_with_flag(self):
        """Tool.from_code() should work when trust_remote_code=True is passed."""
        code = '''
from smolagents import Tool
class TestTool(Tool):
    name = "test_tool"
    description = "test"
    inputs = {}
    output_type = "string"
    def forward(self):
        return "ok"
'''
        tool = Tool.from_code(code, trust_remote_code=True)
        assert tool.name == "test_tool"
