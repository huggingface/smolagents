"""ChatMessage.from_dict must not mutate the caller's dict."""

from smolagents.models import ChatMessage, ChatMessageToolCall


def test_from_dict_is_repeatable():
    message_dict = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"location": "Paris"}},
            }
        ],
    }

    first = ChatMessage.from_dict(message_dict)
    second = ChatMessage.from_dict(message_dict)

    assert isinstance(message_dict["tool_calls"][0], dict)
    assert first.tool_calls[0].function.name == "get_weather"
    assert second.tool_calls[0].function.name == "get_weather"
    assert isinstance(first.tool_calls[0], ChatMessageToolCall)
