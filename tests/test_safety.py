import unittest
from unittest.mock import patch
from smolagents.tools import Tool

class DangerousTool(Tool):
    name = "nuke_database"
    description = "Deletes everything."
    inputs = {}
    output_type = "string"

    def forward(self):
        return "BOOM"

class TestSafety(unittest.TestCase):
    def test_confirmation_accepted(self):
        """Test that 'y' allows execution."""
        tool = DangerousTool(requires_confirmation=True)
        with patch('builtins.input', return_value='y'):
            result = tool()
            self.assertEqual(result, "BOOM")

    def test_confirmation_denied(self):
        """Test that 'n' raises an error."""
        tool = DangerousTool(requires_confirmation=True)
        with patch('builtins.input', return_value='n'):
            with self.assertRaises(ValueError) as cm:
                tool()
            self.assertIn("User denied", str(cm.exception))

if __name__ == "__main__":
    unittest.main()
