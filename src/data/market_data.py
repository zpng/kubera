"""
Market Data Module - yfinance integration
Provides stock price data, historical data, and basic market information
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MarketDataProvider:
    """
    Market data provider using yfinance.
    Handles stock price queries, historical data, and basic company info.
    """

    def __init__(self, cache_dir: str = "./data/cache"):
        """
        Initialize market data provider.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        logger.info(f"MarketDataProvider initialized with cache_dir: {cache_dir}")

    def get_stock_price(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Get historical stock price data (OHLCV).

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "NVDA")
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume

        Example:
            >>> provider = MarketDataProvider()
            >>> data = provider.get_stock_price("AAPL", "2024-01-01", "2024-01-31")
            >>> print(data.head())
        """
        try:
            # Validate date format
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")

            # Create ticker object
            ticker = yf.Ticker(symbol.upper())

            # Fetch historical data
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                logger.warning(f"No data found for {symbol} between {start_date} and {end_date}")
                return pd.DataFrame()

            # Remove timezone info for consistency
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            # Round to 2 decimal places
            numeric_columns = ["Open", "High", "Low", "Close"]
            for col in numeric_columns:
                if col in data.columns:
                    data[col] = data[col].round(2)

            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            raise

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get the latest available price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest closing price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            data = ticker.history(period="1d")

            if data.empty:
                logger.warning(f"No latest price data for {symbol}")
                return None

            latest_price = data['Close'].iloc[-1]
            logger.info(f"Latest price for {symbol}: ${latest_price:.2f}")
            return round(latest_price, 2)

        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get basic company information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company info including:
            - name, sector, industry, market_cap, etc.
        """
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info

            # Extract key fields
            company_data = {
                "symbol": symbol.upper(),
                "name": info.get("longName", "N/A"),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "dividend_yield": info.get("dividendYield", 0),
                "52_week_high": info.get("fiftyTwoWeekHigh", 0),
                "52_week_low": info.get("fiftyTwoWeekLow", 0),
                "description": info.get("longBusinessSummary", "N/A")
            }

            logger.info(f"Retrieved company info for {symbol}")
            return company_data

        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {}

    def get_balance_sheet(
        self,
        symbol: str,
        freq: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Get balance sheet data.

        Args:
            symbol: Stock ticker symbol
            freq: "quarterly" or "annual"

        Returns:
            DataFrame with balance sheet data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            if freq.lower() == "quarterly":
                data = ticker.quarterly_balance_sheet
            else:
                data = ticker.balance_sheet

            if data.empty:
                logger.warning(f"No balance sheet data for {symbol}")
                return pd.DataFrame()

            logger.info(f"Retrieved {freq} balance sheet for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching balance sheet for {symbol}: {e}")
            return pd.DataFrame()

    def get_income_statement(
        self,
        symbol: str,
        freq: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Get income statement data.

        Args:
            symbol: Stock ticker symbol
            freq: "quarterly" or "annual"

        Returns:
            DataFrame with income statement data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            if freq.lower() == "quarterly":
                data = ticker.quarterly_income_stmt
            else:
                data = ticker.income_stmt

            if data.empty:
                logger.warning(f"No income statement data for {symbol}")
                return pd.DataFrame()

            logger.info(f"Retrieved {freq} income statement for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching income statement for {symbol}: {e}")
            return pd.DataFrame()

    def get_cash_flow(
        self,
        symbol: str,
        freq: str = "quarterly"
    ) -> pd.DataFrame:
        """
        Get cash flow data.

        Args:
            symbol: Stock ticker symbol
            freq: "quarterly" or "annual"

        Returns:
            DataFrame with cash flow data
        """
        try:
            ticker = yf.Ticker(symbol.upper())

            if freq.lower() == "quarterly":
                data = ticker.quarterly_cashflow
            else:
                data = ticker.cashflow

            if data.empty:
                logger.warning(f"No cash flow data for {symbol}")
                return pd.DataFrame()

            logger.info(f"Retrieved {freq} cash flow for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error fetching cash flow for {symbol}: {e}")
            return pd.DataFrame()

    def get_multiple_stocks(
        self,
        symbols: list,
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks at once.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in "YYYY-MM-DD" format
            end_date: End date in "YYYY-MM-DD" format

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}

        for symbol in symbols:
            try:
                data = self.get_stock_price(symbol, start_date, end_date)
                results[symbol] = data
            except Exception as e:
                logger.error(f"Failed to get data for {symbol}: {e}")
                results[symbol] = pd.DataFrame()

        logger.info(f"Retrieved data for {len(results)} stocks")
        return results


# Convenience function for quick access
def get_stock_data(symbol: str, days_back: int = 30) -> pd.DataFrame:
    """
    Quick function to get recent stock data.

    Args:
        symbol: Stock ticker symbol
        days_back: Number of days to look back

    Returns:
        DataFrame with recent stock data
    """
    provider = MarketDataProvider()
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    return provider.get_stock_price(symbol, start_date, end_date)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    provider = MarketDataProvider()

    # Test getting stock data
    print("\n=== Testing Stock Price Data ===")
    data = provider.get_stock_price("AAPL", "2024-01-01", "2024-01-31")
    print(data.head())

    # Test getting latest price
    print("\n=== Testing Latest Price ===")
    price = provider.get_latest_price("NVDA")
    print(f"NVDA latest price: ${price}")

    # Test company info
    print("\n=== Testing Company Info ===")
    info = provider.get_company_info("MSFT")
    print(f"Company: {info.get('name')}")
    print(f"Sector: {info.get('sector')}")
    print(f"Market Cap: ${info.get('market_cap'):,}")
