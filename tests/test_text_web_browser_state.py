"""Regression test for text-browser find state reset."""

from pathlib import Path
import importlib.util


MODULE_PATH = Path(__file__).parents[1] / "examples" / "open_deep_research" / "scripts" / "text_web_browser.py"


def _load_browser_module():
    package_dir = MODULE_PATH.parent
    spec = importlib.util.spec_from_file_location("text_web_browser_test_module", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "examples.open_deep_research.scripts"
    module.__path__ = [str(package_dir)]
    raise RuntimeError("The example uses relative imports; run this regression from its example package context.")


def test_source_resets_actual_find_state_attributes():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "self._find_on_page_query = None" in source
    assert "self._find_on_page_last_result = None" in source
    assert "self.find_on_page_query = None" not in source
    assert "self.find_on_page_viewport = None" not in source
