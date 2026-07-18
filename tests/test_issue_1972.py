from unittest.mock import MagicMock, patch

from smolagents.models import LiteLLMModel


def test_issue_1972():
    mock_litellm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "FOO.BAR"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_litellm.completion.return_value = mock_response

    messages = [
        {"role": "system", "content": "When you say anything Start with 'FOO'"},
        {"role": "system", "content": "When you say anything End with 'BAR'"},
        {"role": "user", "content": "Just say '.'"},
    ]

    with patch("smolagents.models.LiteLLMModel.create_client", return_value=mock_litellm):
        model = LiteLLMModel(model_id="gemini/gemini-2.5-flash", api_key="test_api_key")
        response = model(messages)

    assert response.content == "FOO.BAR"
    assert mock_litellm.completion.call_count == 1
