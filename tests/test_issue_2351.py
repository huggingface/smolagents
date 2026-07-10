from unittest.mock import MagicMock

from smolagents.models import AmazonBedrockModel, ChatMessage, MessageRole


def test_issue_2351():
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "The 118th Fibonacci number is 1264937032042997393488322."}],
            }
        },
        "usage": {
            "inputTokens": 12,
            "outputTokens": 8,
        },
    }

    model = AmazonBedrockModel(
        model_id="us.anthropic.claude-sonnet-4-6",
        client=client,
    )
    messages = [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Hello"}])]

    result = model.generate(messages)

    assert result.role == MessageRole.ASSISTANT
    assert result.content == "The 118th Fibonacci number is 1264937032042997393488322."
    assert result.tool_calls is None
    assert result.token_usage.input_tokens == 12
    assert result.token_usage.output_tokens == 8
