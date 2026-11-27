"""Markdown fence fallback must consume the whole Python info string."""

import pytest

from smolagents.utils import parse_code_blobs

TAGS = ("<code>", "</code>")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Thought: x\n```python3\nprint(1)\n```", "print(1)"),
        ("Thought: x\n```Python\nprint(1)\n```", "print(1)"),
        ("Thought: x\n```PY\nprint(1)\n```", "print(1)"),
        ("Thought: x\n```python\nprint(1)\n```", "print(1)"),
    ],
)
def test_python_fence_tags(text, expected):
    assert parse_code_blobs(text, TAGS) == expected


def test_pycon_fence_is_not_treated_as_python():
    with pytest.raises(ValueError):
        parse_code_blobs("Thought: x\n```pycon\n>>> print(1)\n```", TAGS)
