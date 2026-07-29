from unittest.mock import MagicMock

from smolagents.models import AmazonBedrockModel


def test_issue_2351():
    client = MagicMock()
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "8670007398507948658051921"}],
            }
        },
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    model = AmazonBedrockModel(model_id="us.anthropic.claude-sonnet-4-6", client=client)

    response = model.generate(
        [{"role": "user", "content": [{"type": "text", "text": "Give me the 118th Fibonacci number."}]}]
    )

    assert response.content == "8670007398507948658051921"
    assert response.tool_calls is None
