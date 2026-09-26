"""Tests for parse_tool_calls_from_text function."""
import json
import pytest
from unittest.mock import patch
from smolagents.models import parse_tool_calls_from_text


class TestParseToolCallsFromText:
    """Test cases for parallel tool call parsing."""

    def test_single_tool_call(self):
        """Test backward compatibility: single tool call in text."""
        text = '{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
        
        tool_calls = parse_tool_calls_from_text(
            text,
            tool_name_key="name",
            tool_arguments_key="arguments"
        )
        
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == {"city": "Tokyo"}
        assert tool_calls[0].type == "function"
        assert tool_calls[0].id is not None

    def test_multiple_parallel_tool_calls(self):
        """Test multiple tool calls extracted from text."""
        text = '{"name": "get_weather", "arguments": {"city": "Tokyo"}} {"name": "get_weather", "arguments": {"city": "New York"}}'
        
        tool_calls = parse_tool_calls_from_text(
            text,
            tool_name_key="name",
            tool_arguments_key="arguments"
        )
        
        assert len(tool_calls) == 2
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == {"city": "Tokyo"}
        assert tool_calls[1].function.name == "get_weather"
        assert tool_calls[1].function.arguments == {"city": "New York"}

    def test_mixed_content_with_tool_calls(self):
        """Test tool calls mixed with other text content."""
        text = 'Getting weather. {"name": "get_weather", "arguments": {"city": "London"}} Also need: {"name": "get_weather", "arguments": {"city": "Paris"}}'
        
        tool_calls = parse_tool_calls_from_text(
            text,
            tool_name_key="name",
            tool_arguments_key="arguments"
        )
        
        assert len(tool_calls) == 2
        assert tool_calls[0].function.arguments == {"city": "London"}
        assert tool_calls[1].function.arguments == {"city": "Paris"}

    def test_error_handling_invalid_tool_calls(self):
        """Test error handling when no valid tool calls found."""
        text = "This text has no tool calls at all."
        
        with pytest.raises(ValueError, match="No valid tool calls found"):
            parse_tool_calls_from_text(
                text,
                tool_name_key="name",
                tool_arguments_key="arguments"
            )
