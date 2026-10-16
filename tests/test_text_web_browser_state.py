"""Regression tests for text-browser find state."""

from examples.open_deep_research.scripts.text_web_browser import SimpleTextBrowser


class FakeBrowser(SimpleTextBrowser):
    pages = {
        "https://a.example": "alpha one two three four",
        "https://b.example": "alpha " + ("x" * 40) + " alpha second",
    }

    def _fetch_page(self, url: str) -> None:
        self._set_page_content(self.pages[url])


def test_find_state_is_reset_after_navigation():
    browser = FakeBrowser(viewport_size=10, request_kwargs={})

    browser.visit_page("https://a.example")
    assert browser.find_on_page("alpha") is not None
    assert browser._find_on_page_query == "alpha"

    browser.visit_page("https://b.example")

    assert browser._find_on_page_query is None
    assert browser._find_on_page_last_result is None
    assert browser.find_next() is None
    assert browser.find_on_page("alpha") is not None
    assert browser._find_on_page_last_result == 0
