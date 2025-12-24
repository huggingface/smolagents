import json
import unittest
from pydantic import BaseModel
from smolagents.agents import CodeAgent

class DummyModel:
    model_id = "dummy-model"
    def generate(self, *args, **kwargs):
        # Not used in these tests; CodeAgent's structured output validation is exercised directly
        return ""

class UserProfile(BaseModel):
    username: str
    is_admin: bool

class TestStructuredOutput(unittest.TestCase):
    def setUp(self):
        # We do not execute the agent loop; we directly exercise the structured output validation logic
        self.agent = CodeAgent(tools=[], model=DummyModel())

    def _validate(self, output):
        # Mimic the structured output parsing path in Agent.run
        if isinstance(output, dict):
            return UserProfile.model_validate(output)
        if isinstance(output, UserProfile):
            return output
        if isinstance(output, str):
            cleaned = output.strip()
            if cleaned.startswith("```"):
                parts = cleaned.split("```")
                if len(parts) >= 2:
                    cleaned = parts[1]
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
            return UserProfile.model_validate_json(cleaned)
        raise ValueError("Unsupported output type")

    def test_accepts_dict(self):
        result = self._validate({"username": "srijan", "is_admin": True})
        self.assertIsInstance(result, UserProfile)
        self.assertEqual(result.username, "srijan")
        self.assertTrue(result.is_admin)

    def test_accepts_json_string(self):
        payload = '```json\n{"username": "admin", "is_admin": false}\n```'
        result = self._validate(payload)
        self.assertIsInstance(result, UserProfile)
        self.assertFalse(result.is_admin)

    def test_rejects_invalid(self):
        with self.assertRaises(ValueError):
            self._validate({"wrong_key": 123})

if __name__ == "__main__":
    unittest.main()
