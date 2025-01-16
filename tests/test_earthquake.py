import unittest
from unittest.mock import Mock
import json
from datetime import datetime
from smolagents import LiteLLMModel, ToolCallingAgent
from smolagents.extra_tools.ip import GetPublicIPTool
from smolagents.extra_tools.location import GetIPLocationTool
from smolagents.extra_tools.earthquake import GetEarthquakesTool
from smolagents.types import AgentText

class MockPublicIPTool(GetPublicIPTool):
    def forward(self) -> str:
        return "8.8.8.8"

class MockIPLocationTool(GetIPLocationTool):
    def forward(self, ip: str) -> str:
        return "34.0522, -118.2437"

class MockEarthquakesTool(GetEarthquakesTool):
    def forward(self) -> str:
        return json.dumps([{
            "magnitude": 2.8,
            "location": "3 km S of Mentone, CA",
            "coordinates": [34.0456667, -117.13],
            "depth_km": 7.67,
            "minutes_ago": 5
        }])

class TestEarthquakeAgent(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock(spec=LiteLLMModel)
        self.mock_model.get_tool_call.side_effect = [
            ("get_public_ip", {}, "call_1"),
            ("get_ip_location", {"ip": "8.8.8.8"}, "call_2"),
            ("get_earthquakes", {}, "call_3"),
            ("final_answer", {
                "answer": "The nearest earthquake was 2.8 magnitude, 3 km S of Mentone, CA"
            }, "call_4"),
        ]

    def test_agent_flow(self):
        # Create agent with mock tools
        agent = ToolCallingAgent(
            tools=[MockPublicIPTool(), MockIPLocationTool(), MockEarthquakesTool()],
            model=self.mock_model,
            verbose=True,
            add_base_tools=False
        )

        result = agent.run("What's the nearest earthquake to me right now?")
        
        # Verify results
        self.assertIsInstance(result, AgentText)
        self.assertIn("Mentone", str(result))
        self.assertEqual(self.mock_model.get_tool_call.call_count, 4)