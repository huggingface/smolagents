import unittest
from smolagents.state import AgentState, AgentStep, ToolCall, ToolResult
import json
from datetime import datetime

class TestAgentState(unittest.TestCase):
    def test_state_serialization(self):
        # Create a ToolCall and ToolResult
        call = ToolCall(tool_name="QRCodeTool", arguments={"data": "https://arxiv.org/pdf/1706.03762"})
        result = ToolResult(call_id=call.id, output="/tmp/qrcode.png", duration_ms=12.5)

        # Create an AgentStep
        step = AgentStep(
            step_number=1,
            thought="Generate a QR code for the Arxiv paper.",
            tool_calls=[call],
            tool_results=[result],
            token_usage=42,
            latency_ms=100.0
        )

        # Create AgentState
        state = AgentState(
            task="Share Arxiv paper as QR code",
            plan=["Find paper", "Generate QR code", "Share"],
            history=[step],
            working_memory={"last_url": "https://arxiv.org/pdf/1706.03762"},
            total_tokens=42,
            total_cost=0.01
        )

        # Serialize to JSON (Pydantic v2)
        state_json = state.model_dump_json()
        self.assertIn('Share Arxiv paper as QR code', state_json)
        self.assertIn('Generate a QR code for the Arxiv paper.', state_json)
        self.assertIn('QRCodeTool', state_json)
        # Deserialize and check
        loaded = AgentState.model_validate_json(state_json)
        self.assertEqual(loaded.task, state.task)
        self.assertEqual(loaded.history[0].thought, step.thought)

if __name__ == "__main__":
    unittest.main()
