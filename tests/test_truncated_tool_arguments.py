"""A truncated JSON tool-call argument must not be used as a tool value."""

import pytest

from smolagents.models import parse_json_if_needed


def test_incomplete_json_object_raises():
    with pytest.raises(ValueError, match="Invalid JSON tool arguments"):
        parse_json_if_needed('{"answer": "Order 42 is currently in')


def test_plain_string_still_allowed():
    assert parse_json_if_needed("abc") == "abc"
