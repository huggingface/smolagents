"""Tests for Tool.from_space (Issue #2775)."""
from unittest.mock import MagicMock, patch
import pytest

from smolagents.tools import Tool


def test_from_space_single_element_output():
    """Test that from_space handles single-element list/tuple outputs correctly."""
    with patch('gradio_client.Client') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Mock the API view
        mock_client.view_api.return_value = {
            "named_endpoints": {
                "predict": {
                    "parameters": [],
                    "returns": [{"component": "Text"}]
                }
            }
        }
        
        # Test case 1: single-element list -> should return the element
        mock_client.predict.return_value = ["hello"]
        tool = Tool.from_space("test/space", "test_tool", "Test tool")
        result = tool.forward()
        assert result == "hello", f"Expected 'hello', got {result}"
        
        # Test case 2: empty list -> should not raise, return empty list
        mock_client.predict.return_value = []
        result = tool.forward()
        assert result == [], f"Expected [], got {result}"
        
        # Test case 3: scalar output -> should work unchanged
        mock_client.predict.return_value = "hello"
        result = tool.forward()
        assert result == "hello", f"Expected 'hello', got {result}"
        
        # Test case 4: tuple with (result, seed) -> should return result
        mock_client.predict.return_value = ("res", 42)
        result = tool.forward()
        assert result == "res", f"Expected 'res', got {result}"
        
        # Test case 5: tuple with (result, error_message) -> should raise ValueError
        mock_client.predict.return_value = ("res", "error message")
        with pytest.raises(ValueError, match="error message"):
            tool.forward()
