"""Tests that ``Model.to_dict()`` never leaks credentials through ``**kwargs``."""

from unittest.mock import patch

from smolagents.models import Model


def test_to_dict_strips_token_and_api_key_kwargs():
    """Secrets passed as kwargs must not appear in the serialized dict."""
    model = Model(
        model_id="test-model",
        token="super-secret-token",
        api_key="super-secret-key",
    )
    with patch("builtins.print"):
        d = model.to_dict()
    assert "token" not in d
    assert "api_key" not in d
    assert d["model_id"] == "test-model"


def test_to_dict_still_exports_regular_kwargs():
    """Non-sensitive kwargs passed through **self.kwargs must remain in the dict."""
    model = Model(model_id="test-model", temperature=0.7, max_tokens=1024)
    with patch("builtins.print"):
        d = model.to_dict()
    assert d["temperature"] == 0.7
    assert d["max_tokens"] == 1024
    assert d["model_id"] == "test-model"


def test_to_dict_no_secret_when_nothing_passed():
    """A plain model without credentials exports cleanly."""
    model = Model(model_id="test-model")
    with patch("builtins.print"):
        d = model.to_dict()
    assert "token" not in d
    assert "api_key" not in d
