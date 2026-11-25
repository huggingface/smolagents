import locale

import pytest

from smolagents import Tool


def test_from_hub_reads_utf8_tool_with_non_utf8_locale(tmp_path, monkeypatch):
    if locale.getpreferredencoding(False).lower().replace("-", "") == "utf8":
        pytest.skip("Run with a non-UTF-8 locale, e.g. Windows Python -X utf8=0")
    text = "\u65e5\u672c\u8a9e - na\u00efve caf\u00e9"
    source = (
        "from smolagents import Tool\n"
        "class Greeting(Tool):\n"
        "    name = 'greeting'\n"
        f"    description = {text!r}\n"
        "    inputs = {}\n"
        "    output_type = 'string'\n"
        "    def forward(self) -> str:\n"
        f"        return {text!r}\n"
    )
    tool_file = tmp_path / "tool.py"
    tool_file.write_text(source, encoding="utf-8")
    monkeypatch.setattr("smolagents.tools.hf_hub_download", lambda *args, **kwargs: str(tool_file))
    loaded = Tool.from_hub("test/local-fixture", trust_remote_code=True)
    assert loaded.description == text
    assert loaded() == text
