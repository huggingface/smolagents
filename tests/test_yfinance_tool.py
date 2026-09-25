import sys
import unittest
from unittest.mock import MagicMock, patch
from smolagents.yfinance_tool import YahooFinanceTool

class TestYFinanceTool(unittest.TestCase):
    def test_stock_lookup(self):
        """
        Test stock data retrieval with mocked yfinance.
        """
        # 1. Mock the Ticker Object
        mock_ticker = MagicMock()
        mock_ticker.info = {
            'regularMarketPrice': 150.00,
            'currency': 'USD',
            'marketCap': 2000000000000,
            'trailingPE': 35.5,
            'sector': 'Technology',
            'longBusinessSummary': 'Nvidia Corporation provides graphics, computing and networking solutions...'
        }
        
        # 2. Mock the Library
        mock_yf = MagicMock()
        mock_yf.Ticker.return_value = mock_ticker
        
        # 3. Patch and Run
        with patch.dict(sys.modules, {'yfinance': mock_yf}):
            tool = YahooFinanceTool()
            result = tool.forward("NVDA")
            
            # 4. Verify
            self.assertIn("NVDA Analysis", result)
            self.assertIn("150.0", result)
            self.assertIn("Technology", result)

    def test_missing_dependency(self):
        """Test graceful failure."""
        with patch.dict(sys.modules):
            if 'yfinance' in sys.modules:
                del sys.modules['yfinance']
            
            tool = YahooFinanceTool()
            result = tool.forward("AAPL")
            self.assertIn("Please install 'yfinance'", result)

if __name__ == "__main__":
    unittest.main()
