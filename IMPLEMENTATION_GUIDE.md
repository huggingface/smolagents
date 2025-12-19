# Parallel Tool Calls Support - Implementation Guide

## Issue Reference
Fixes: huggingface/smolagents#1899

## Problem Statement
TransformersModel only extracts ONE tool call per response, even when models like Qwen natively support parallel tool calls. This limits local model performance on tasks requiring independent parallel operations.

## Root Cause
The `parse_tool_calls()` method wraps a single `get_tool_call_from_text()` call in a list, and `parse_json_blob()` explicitly rejects multiple JSON objects with error: "PROVIDE ONLY ONE TOOL CALL"

## Solution Summary
Modify text-based tool call parsing to extract ALL tool calls from text, not just one.

## Code Changes Required

### 1. Add new function in src/smolagents/models.py (before Model class):

def parse_tool_calls_from_text(text, tool_name_key, tool_arguments_key):
    """Extract ALL tool calls from text with multiple JSON objects."""
    tool_calls = []
    text_to_parse = text
    while text_to_parse:
        try:
            tool_call_dict, rest = parse_json_blob(text_to_parse)
            tool_call = ChatMessageToolCall(
                id=str(uuid.uuid4()),
                type="function",
                function=ChatMessageToolCallFunction(
                    name=tool_call_dict[tool_name_key],
                    arguments=tool_call_dict.get(tool_arguments_key),
                ),
            )
            tool_calls.append(tool_call)
            text_to_parse = rest.strip()
            if not text_to_parse or "{" not in text_to_parse:
                break
        except ValueError:
            if tool_calls:
                break
            raise ValueError("No valid tool calls found")
    return tool_calls

### 2. Update Model.parse_tool_calls() method:

if not message.tool_calls:
    message.tool_calls = parse_tool_calls_from_text(
        message.content,
        self.tool_name_key,
        self.tool_arguments_key
    )

## Benefits
- Backward compatible with single tool calls
- Leverages existing ThreadPoolExecutor infrastructure
- Enables Qwen, Llama parallel support
- ~40 lines of code

## Testing Scope
1. Single tool call (backward compat)
2. Multiple parallel tool calls
3. Mixed content with tool calls
4. Error handling

## References
- Issue: huggingface/smolagents#1899
- PR #1412: Added parallel execution support
- agents.py lines 1300-1350: ToolCallingAgent.process_tool_calls()
