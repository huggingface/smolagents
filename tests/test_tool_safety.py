import unittest
from unittest.mock import patch
from smolagents.tools import Tool

# 1. Define a dummy tool for testing
class DangerousTool(Tool):
    name = "launch_missiles"
    description = "A dangerous test tool."
    inputs = {
        "target": {
            "type": "string",
            "description": "Target to launch missiles at.",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, target="moon"):
        return f"Missiles launched at {target}"

class TestToolSafety(unittest.TestCase):
    
    def test_confirmation_allowed(self):
        """Test that user input 'y' allows execution."""
        tool = DangerousTool(requires_confirmation=True)
        
        # Mock 'input' to return 'y' automatically
        with patch('builtins.input', return_value='y'):
            result = tool(target="mars")
            self.assertEqual(result, "Missiles launched at mars")

    def test_confirmation_denied(self):
        """Test that user input 'n' raises ValueError."""
        tool = DangerousTool(requires_confirmation=True)
        
        # Mock 'input' to return 'n'
        with patch('builtins.input', return_value='n'):
            with self.assertRaises(ValueError) as cm:
                tool(target="mars")
            
            self.assertIn("User denied execution", str(cm.exception))

    def test_no_confirmation_needed(self):
        """Test that normal tools run without nagging."""
        tool = DangerousTool(requires_confirmation=False)
        result = tool(target="mars")
        self.assertEqual(result, "Missiles launched at mars")

if __name__ == "__main__":
    unittest.main()
