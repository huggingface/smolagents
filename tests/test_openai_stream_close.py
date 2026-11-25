"""Closing a model stream must release the owned SDK stream."""

from unittest.mock import MagicMock

from smolagents.models import ChatMessage, MessageRole, OpenAIModel


def _event():
    event = MagicMock()
    event.usage = None
    choice = MagicMock()
    choice.delta.content = "hi"
    choice.delta.tool_calls = None
    event.choices = [choice]
    return event


def _model_with_stream(stream):
    model = OpenAIModel(model_id="gpt-4o-mini", api_key="k")
    model.client = MagicMock()
    model.client.chat.completions.create.return_value = stream
    return model


def test_closing_generator_closes_sdk_stream():
    stream = MagicMock()
    stream.__iter__.return_value = iter([_event(), _event()])
    model = _model_with_stream(stream)

    generator = model.generate_stream([ChatMessage(role=MessageRole.USER, content="hi")])
    next(generator)
    generator.close()

    stream.close.assert_called_once()


def test_full_consumption_closes_sdk_stream():
    stream = MagicMock()
    stream.__iter__.return_value = iter([_event()])
    model = _model_with_stream(stream)

    list(model.generate_stream([ChatMessage(role=MessageRole.USER, content="hi")]))

    stream.close.assert_called_once()
