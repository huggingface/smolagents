from unittest.mock import MagicMock, patch

import pytest

from smolagents.default_tools import ApiWebSearchTool, DuckDuckGoSearchTool


@pytest.fixture(params=[DuckDuckGoSearchTool, ApiWebSearchTool])
def rate_limited_search_tool(request):
    with patch.dict("sys.modules", {"ddgs": MagicMock()}):
        return request.param(rate_limit=2.0)


def test_search_rate_limit_uses_monotonic_clock(rate_limited_search_tool):
    with (
        patch("time.time", side_effect=AssertionError("Wall clock must not control throttling")),
        patch("time.monotonic", side_effect=[0.0, 0.0, 0.1, 0.5]),
        patch("time.sleep") as sleep,
    ):
        rate_limited_search_tool._enforce_rate_limit()
        sleep.assert_not_called()
        rate_limited_search_tool._enforce_rate_limit()
        sleep.assert_called_once_with(pytest.approx(0.4))


def test_search_rate_limit_elapsed_interval(rate_limited_search_tool):
    with (
        patch("time.monotonic", side_effect=[10.0, 10.0, 10.8, 10.8]),
        patch("time.sleep") as sleep,
    ):
        rate_limited_search_tool._enforce_rate_limit()
        rate_limited_search_tool._enforce_rate_limit()
        sleep.assert_not_called()


@pytest.mark.parametrize("rate_limit", [None, 0])
def test_search_rate_limit_disabled(rate_limited_search_tool, rate_limit):
    rate_limited_search_tool.rate_limit = rate_limit
    with patch("time.monotonic") as clock, patch("time.sleep") as sleep:
        rate_limited_search_tool._enforce_rate_limit()
        clock.assert_not_called()
        sleep.assert_not_called()
