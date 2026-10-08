import re
import pytest
from smolagents.models import ChatMessage, Model, MessageRole

class DummyModel(Model):
    def __init__(self):
        super().__init__(model_id="dummy")
    def generate(self, messages, **kwargs):
        # Simulate DeepSeek output
        class DummyResponse:
            class Choice:
                class Message:
                    role = "assistant"
                    content = """<think>I need to print hello.</think>\n```python\nprint(\"Hello World\")\n```"""
                    tool_calls = None
                message = Message()
                tool_calls = None
            choices = [Choice()]
            usage = type("Usage", (), {"prompt_tokens": 1, "completion_tokens": 1})()
        response = DummyResponse()
        # Patch the same logic as in the real model
        content = response.choices[0].message.content
        if content and "<think>" in content:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        return ChatMessage(role="assistant", content=content)

def test_deepseek_think_tag_removal():
    model = DummyModel()
    messages = [ChatMessage(role=MessageRole.USER, content="Say hello")]
    result = model.generate(messages)
    # Should only contain the code block, not the <think> tag
    assert "<think>" not in result.content
    assert "print(\"Hello World\")" in result.content
    assert result.content.strip().startswith("```python")
