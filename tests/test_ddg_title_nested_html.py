"""DuckDuckGo parser must keep title text outside nested tags."""

from smolagents.default_tools import WebSearchTool


def _parse(html):
    tool = object.__new__(WebSearchTool)
    parser = tool._create_duckduckgo_parser()
    parser.feed(html)
    parser.close()
    return parser.results


HTML = (
    "<table><tr>"
    "<td><a class='result-link'><b>Local</b> inventory &amp; count</a></td>"
    "<td class='result-snippet'>There are <strong>17</strong> entries.</td>"
    "<td><span class='link-text'>catalog.invalid/local</span></td>"
    "</tr></table>"
)


def test_title_keeps_text_before_nested_tag():
    results = _parse(HTML)

    assert results[0]["title"] == "Local inventory & count"
    assert results[0]["description"] == "There are 17 entries."
