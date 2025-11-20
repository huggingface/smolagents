"""Test XPath injection vulnerability fix in vision_web_browser.py"""

from unittest.mock import Mock, patch

import pytest

from smolagents.vision_web_browser import _escape_xpath_string, search_item_ctrl_f


@pytest.fixture
def mock_driver():
    """Mock Selenium WebDriver"""
    driver = Mock()
    driver.find_elements.return_value = [Mock()]  # Mock found elements
    driver.execute_script.return_value = None
    return driver


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
