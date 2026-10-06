"""Test vision_web_browser module, including lazy imports and XPath escaping."""

import importlib
import re
import sys
from unittest.mock import Mock, patch

import pytest

from smolagents.utils import _is_package_available
from smolagents.vision_web_browser import (
    _escape_xpath_string,
    parse_arguments,
    run_webagent,
    search_item_ctrl_f,
)


@pytest.fixture
def mock_driver():
    """Mock Selenium WebDriver and selenium modules."""
    driver = Mock()
    driver.find_elements.return_value = [Mock()]  # Mock found elements
    driver.execute_script.return_value = None
    mock_by = Mock()
    mock_by.XPATH = "xpath"
    modules = {
        "selenium": Mock(),
        "selenium.webdriver": Mock(),
        "selenium.webdriver.common": Mock(),
        "selenium.webdriver.common.by": Mock(By=mock_by),
        "selenium.webdriver.common.keys": Mock(),
    }
    with patch.dict(sys.modules, modules):
        yield driver


class TestXPathEscaping:
    """Test XPath string escaping functionality"""

    @pytest.mark.parametrize(
        "input_text,expected_pattern",
        [
            ("normal text", "'normal text'"),
            ("text with 'quote'", "\"text with 'quote'\""),
            ('text with "quote"', "'text with \"quote\"'"),
            ("text with one single'quote", '"text with one single\'quote"'),
            ('text with one double"quote', "'text with one double\"quote'"),
            (
                "text with both 'single' and \"double\" quotes",
                "concat('text with both ', \"'\", 'single', \"'\", ' and \"double\" quotes')",
            ),
            ("", "''"),
            ("'", '"\'"'),
            ('"', "'\"'"),
        ],
    )
    def test_escape_xpath_string_basic(self, input_text, expected_pattern):
        """Test basic XPath escaping cases"""
        result = _escape_xpath_string(input_text)
        assert result == expected_pattern

    @pytest.mark.parametrize(
        "input_text",
        [
            "text with both 'single' and \"double\" quotes",
            'it\'s a "test" case',
            "'mixed\" quotes'",
        ],
    )
    def test_escape_xpath_string_mixed_quotes(self, input_text):
        """Test XPath escaping with mixed quotes uses concat()"""
        result = _escape_xpath_string(input_text)
        assert result.startswith("concat(")
        assert result.endswith(")")

    @pytest.mark.parametrize(
        "malicious_input",
        [
            "')] | //script[@src='evil.js'] | foo[contains(text(), '",
            "') or 1=1 or ('",
            "')] | //user[contains(@role,'admin')] | foo[contains(text(), '",
            "') and substring(//user[1]/password,1,1)='a",
        ],
    )
    def test_escape_prevents_injection(self, malicious_input):
        """Test that malicious XPath injection attempts are safely escaped"""
        result = _escape_xpath_string(malicious_input)
        # Should either be wrapped in quotes or use concat()
        assert (
            (result.startswith("'") and result.endswith("'"))
            or (result.startswith('"') and result.endswith('"'))
            or result.startswith("concat(")
        )


class TestSearchItemCtrlF:
    """Test the search_item_ctrl_f function with XPath injection protection"""

    @pytest.mark.parametrize(
        "search_text",
        [
            "normal search",
            "search with 'quotes'",
            'search with "quotes"',
            "')] | //script[@src='evil.js'] | foo[contains(text(), '",
            "') or 1=1 or ('",
        ],
    )
    def test_search_item_prevents_injection(self, search_text, mock_driver):
        """Test that search_item_ctrl_f prevents XPath injection"""
        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            # Call the function
            result = search_item_ctrl_f(search_text)

            # Verify driver.find_elements was called
            mock_driver.find_elements.assert_called_once()

            # Get the actual XPath query that was generated
            call_args = mock_driver.find_elements.call_args
            xpath_query = call_args[0][1]  # Second positional argument

            # Verify the query doesn't contain unescaped injection
            if "')] | //" in search_text:
                # For injection attempts, verify they're properly escaped
                # The query should either use concat() or be properly quoted
                is_concat = "concat(" in xpath_query
                is_properly_quoted = xpath_query.count('"') >= 2 or xpath_query.count("'") >= 2
                assert is_concat or is_properly_quoted, f"XPath injection not prevented: {xpath_query}"

            # Verify we got a result
            assert "Found" in result

    def test_search_item_nth_result(self, mock_driver):
        """Test nth_result parameter works correctly"""
        mock_driver.find_elements.return_value = [Mock(), Mock(), Mock()]  # 3 elements

        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            result = search_item_ctrl_f("test", nth_result=2)

            # Should find 3 matches and focus on element 2
            assert "Found 3 matches" in result
            assert "Focused on element 2 of 3" in result

    def test_search_item_not_found(self, mock_driver):
        """Test exception when nth_result exceeds available matches"""
        mock_driver.find_elements.return_value = [Mock()]  # Only 1 element

        with patch("smolagents.vision_web_browser.driver", mock_driver, create=True):
            with pytest.raises(Exception, match="Match n°3 not found"):
                search_item_ctrl_f("test", nth_result=3)


def test_import_vision_web_browser_without_vision_extra():
    """Test that import smolagents.vision_web_browser succeeds even when helium and selenium are blocked from sys.modules."""
    blocked_modules = {
        "helium": None,
        "selenium": None,
        "selenium.webdriver": None,
        "selenium.webdriver.common": None,
        "selenium.webdriver.common.by": None,
        "selenium.webdriver.common.keys": None,
    }
    with patch.dict(sys.modules, blocked_modules):
        saved_module = sys.modules.pop("smolagents.vision_web_browser", None)
        try:
            mod = importlib.import_module("smolagents.vision_web_browser")
            assert mod is not None
            assert hasattr(mod, "run_webagent")
            assert hasattr(mod, "parse_arguments")
        finally:
            if saved_module is not None:
                sys.modules["smolagents.vision_web_browser"] = saved_module


def test_run_webagent_missing_vision_extra_raises_helpful_error():
    """Test that calling run_webagent("test", "InferenceClientModel", "test-model") without the vision extra raises ModuleNotFoundError with the exact helpful error message."""
    expected_message = "Please install 'vision' extra to use the web agent: `pip install 'smolagents[vision]'`"
    with patch("smolagents.vision_web_browser._is_package_available", return_value=False):
        with pytest.raises(ModuleNotFoundError, match=re.escape(expected_message)) as exc_info:
            run_webagent("test", "InferenceClientModel", "test-model")
        assert str(exc_info.value) == expected_message

    if not _is_package_available("helium") or not _is_package_available("selenium"):
        with pytest.raises(ModuleNotFoundError, match=re.escape(expected_message)) as exc_info:
            run_webagent("test", "InferenceClientModel", "test-model")
        assert str(exc_info.value) == expected_message


def test_run_webagent_missing_only_helium():
    """Test when selenium is available but helium is missing."""
    expected_message = "Please install 'vision' extra to use the web agent: `pip install 'smolagents[vision]'`"

    def mock_is_pkg(pkg_name: str) -> bool:
        if pkg_name == "selenium":
            return True
        if pkg_name == "helium":
            return False
        return False

    with patch("smolagents.vision_web_browser._is_package_available", side_effect=mock_is_pkg):
        with pytest.raises(ModuleNotFoundError, match=re.escape(expected_message)) as exc_info:
            run_webagent("test", "InferenceClientModel", "test-model")
        assert str(exc_info.value) == expected_message


def test_run_webagent_missing_only_selenium():
    """Test when helium is available but selenium is missing."""
    expected_message = "Please install 'vision' extra to use the web agent: `pip install 'smolagents[vision]'`"

    def mock_is_pkg(pkg_name: str) -> bool:
        if pkg_name == "helium":
            return True
        if pkg_name == "selenium":
            return False
        return False

    with patch("smolagents.vision_web_browser._is_package_available", side_effect=mock_is_pkg):
        with pytest.raises(ModuleNotFoundError, match=re.escape(expected_message)) as exc_info:
            run_webagent("test", "InferenceClientModel", "test-model")
        assert str(exc_info.value) == expected_message


def test_webagent_cli_help(capsys):
    """Test that running --help through parse_arguments succeeds without error."""
    with patch.object(sys, "argv", ["webagent", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()
        assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Run a web browser automation script with a specified model." in captured.out
    assert "--model-type" in captured.out
    assert "--model-id" in captured.out
