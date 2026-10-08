from .tools import Tool

class YahooFinanceTool(Tool):
    name = "yahoo_finance_tool"
    description = "Retrieves real-time stock data and company fundamentals for a given ticker symbol using yfinance. Input: ticker (str). Output: formatted string with price, sector, market cap, P/E ratio, and summary."
    inputs = {
        "ticker": {
            "type": "string",
            "description": "The stock symbol (e.g., 'NVDA', 'AAPL', 'BTC-USD')."
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        """
        Initialize the Yahoo Finance tool.
        """
        super().__init__(**kwargs)

    def forward(self, ticker: str) -> str:
        """
        Retrieves real-time stock data and company fundamentals for a given ticker symbol.
        
        Args:
            ticker: The stock symbol (e.g., 'NVDA', 'AAPL', 'BTC-USD').
        """
        try:
            # Lazy Import
            import yfinance as yf
        except ImportError:
            return "Error: Please install 'yfinance' package to use this tool."

        try:
            # 1. Clean the ticker
            symbol = ticker.strip().upper()
            stock = yf.Ticker(symbol)
            
            # 2. Fetch Info (Fastest way to get current state)
            info = stock.info
            
            # 3. Handle Invalid Tickers
            # yfinance doesn't always raise errors, sometimes it returns empty info
            if not info or 'regularMarketPrice' not in info:
                # Fallback: Try fetching history to see if it exists
                hist = stock.history(period="1d")
                if hist.empty:
                    return f"Error: Could not find data for symbol '{symbol}'."
                current_price = hist['Close'].iloc[-1]
            else:
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')

            # 4. Extract Key Metrics
            currency = info.get('currency', 'USD')
            market_cap = info.get('marketCap', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            sector = info.get('sector', 'Unknown')
            summary = info.get('longBusinessSummary', '')[:200] # Truncate

            # 5. Format Output
            report = (
                f"\U0001F4C8 **{symbol} Analysis**\n"
                f"\U0001F4B0 Price: {current_price} {currency}\n"
                f"\U0001F3E2 Sector: {sector}\n"
                f"\U0001F4CA Market Cap: {market_cap}\n"
                f"\U0001F4C9 P/E Ratio: {pe_ratio}\n"
                f"\U0001F4DD Summary: {summary}..."
            )
            return report

        except Exception as e:
            return f"Error fetching data for {ticker}: {str(e)}"
