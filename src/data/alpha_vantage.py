"""
Alpha Vantage Data Provider
Provides news, fundamentals, and additional market data
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import StringIO
import logging

logger = logging.getLogger(__name__)


class AlphaVantageRateLimitError(Exception):
    """Raised when Alpha Vantage API rate limit is exceeded."""
    pass


class AlphaVantageProvider:
    """
    Alpha Vantage data provider for news and fundamental data.
    Free tier: 25 API calls per day.
    """

    API_BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage provider.

        Args:
            api_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY not found in environment")

        logger.info("AlphaVantageProvider initialized")

    def _make_request(self, function: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request to Alpha Vantage.

        Args:
            function: API function name
            params: Additional parameters

        Returns:
            JSON response as dictionary

        Raises:
            AlphaVantageRateLimitError: If rate limit exceeded
        """
        request_params = {
            "function": function,
            "apikey": self.api_key,
            **params
        }

        try:
            response = requests.get(self.API_BASE_URL, params=request_params)
            response.raise_for_status()

            # Try to parse as JSON
            data = response.json()

            # Check for rate limit
            if "Information" in data:
                info_msg = data["Information"]
                if "rate limit" in info_msg.lower() or "call frequency" in info_msg.lower():
                    raise AlphaVantageRateLimitError(f"Rate limit exceeded: {info_msg}")

            # Check for error message
            if "Error Message" in data:
                raise ValueError(f"API Error: {data['Error Message']}")

            return data

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_news_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get news and sentiment data.

        Args:
            tickers: List of stock tickers to filter by (e.g., ["AAPL", "MSFT"])
            topics: List of topics (e.g., ["technology", "finance"])
            limit: Max number of articles (1-1000, default 50)

        Returns:
            List of news articles with sentiment scores

        Example:
            >>> provider = AlphaVantageProvider()
            >>> news = provider.get_news_sentiment(tickers=["NVDA"], limit=10)
            >>> for article in news:
            >>>     print(article['title'], article['overall_sentiment_score'])
        """
        params = {"limit": limit}

        if tickers:
            params["tickers"] = ",".join(tickers)

        if topics:
            params["topics"] = ",".join(topics)

        try:
            data = self._make_request("NEWS_SENTIMENT", params)

            feed = data.get("feed", [])
            logger.info(f"Retrieved {len(feed)} news articles")

            # Extract relevant fields
            articles = []
            for item in feed:
                article = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "time_published": item.get("time_published", ""),
                    "authors": item.get("authors", []),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "overall_sentiment_score": item.get("overall_sentiment_score", 0),
                    "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                    "ticker_sentiment": item.get("ticker_sentiment", [])
                }
                articles.append(article)

            return articles

        except Exception as e:
            logger.error(f"Error fetching news sentiment: {e}")
            raise

    def get_company_overview(self, symbol: str) -> Dict[str, Any]:
        """
        Get fundamental company data.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with company fundamentals including:
            - Market cap, P/E ratio, EPS, dividend yield, etc.

        Example:
            >>> provider = AlphaVantageProvider()
            >>> overview = provider.get_company_overview("AAPL")
            >>> print(f"PE Ratio: {overview['PERatio']}")
        """
        params = {"symbol": symbol.upper()}

        try:
            data = self._make_request("OVERVIEW", params)

            if not data or "Symbol" not in data:
                logger.warning(f"No overview data for {symbol}")
                return {}

            # Extract key fundamentals
            overview = {
                "symbol": data.get("Symbol", ""),
                "name": data.get("Name", ""),
                "description": data.get("Description", ""),
                "sector": data.get("Sector", ""),
                "industry": data.get("Industry", ""),
                "market_cap": data.get("MarketCapitalization", "0"),
                "pe_ratio": data.get("PERatio", "0"),
                "peg_ratio": data.get("PEGRatio", "0"),
                "book_value": data.get("BookValue", "0"),
                "dividend_per_share": data.get("DividendPerShare", "0"),
                "dividend_yield": data.get("DividendYield", "0"),
                "eps": data.get("EPS", "0"),
                "revenue_per_share": data.get("RevenuePerShareTTM", "0"),
                "profit_margin": data.get("ProfitMargin", "0"),
                "operating_margin": data.get("OperatingMarginTTM", "0"),
                "return_on_assets": data.get("ReturnOnAssetsTTM", "0"),
                "return_on_equity": data.get("ReturnOnEquityTTM", "0"),
                "revenue": data.get("RevenueTTM", "0"),
                "gross_profit": data.get("GrossProfitTTM", "0"),
                "diluted_eps": data.get("DilutedEPSTTM", "0"),
                "quarterly_earnings_growth": data.get("QuarterlyEarningsGrowthYOY", "0"),
                "quarterly_revenue_growth": data.get("QuarterlyRevenueGrowthYOY", "0"),
                "analyst_target_price": data.get("AnalystTargetPrice", "0"),
                "52_week_high": data.get("52WeekHigh", "0"),
                "52_week_low": data.get("52WeekLow", "0"),
                "50_day_ma": data.get("50DayMovingAverage", "0"),
                "200_day_ma": data.get("200DayMovingAverage", "0")
            }

            logger.info(f"Retrieved company overview for {symbol}")
            return overview

        except Exception as e:
            logger.error(f"Error fetching company overview for {symbol}: {e}")
            raise

    def get_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get earnings data (annual and quarterly).

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary with annual and quarterly earnings

        Example:
            >>> provider = AlphaVantageProvider()
            >>> earnings = provider.get_earnings("MSFT")
            >>> print(earnings['quarterlyEarnings'][:2])
        """
        params = {"symbol": symbol.upper()}

        try:
            data = self._make_request("EARNINGS", params)

            if not data or "symbol" not in data:
                logger.warning(f"No earnings data for {symbol}")
                return {}

            earnings = {
                "symbol": data.get("symbol", ""),
                "annual_earnings": data.get("annualEarnings", []),
                "quarterly_earnings": data.get("quarterlyEarnings", [])
            }

            logger.info(f"Retrieved earnings data for {symbol}")
            return earnings

        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            raise

    def get_top_gainers_losers(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top gainers, losers, and most actively traded stocks.

        Returns:
            Dictionary with three lists:
            - top_gainers: Stocks with biggest gains
            - top_losers: Stocks with biggest losses
            - most_actively_traded: Highest volume stocks

        Example:
            >>> provider = AlphaVantageProvider()
            >>> market_movers = provider.get_top_gainers_losers()
            >>> print("Top Gainer:", market_movers['top_gainers'][0])
        """
        try:
            data = self._make_request("TOP_GAINERS_LOSERS", {})

            result = {
                "top_gainers": data.get("top_gainers", []),
                "top_losers": data.get("top_losers", []),
                "most_actively_traded": data.get("most_actively_traded", [])
            }

            logger.info("Retrieved top gainers/losers")
            return result

        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
            raise

    def search_symbol(self, keywords: str) -> List[Dict[str, Any]]:
        """
        Search for stock symbols by keywords.

        Args:
            keywords: Search query (company name, ticker, etc.)

        Returns:
            List of matching symbols with info

        Example:
            >>> provider = AlphaVantageProvider()
            >>> results = provider.search_symbol("Apple")
            >>> for match in results:
            >>>     print(f"{match['symbol']}: {match['name']}")
        """
        params = {"keywords": keywords}

        try:
            data = self._make_request("SYMBOL_SEARCH", params)

            matches = data.get("bestMatches", [])

            results = []
            for match in matches:
                results.append({
                    "symbol": match.get("1. symbol", ""),
                    "name": match.get("2. name", ""),
                    "type": match.get("3. type", ""),
                    "region": match.get("4. region", ""),
                    "currency": match.get("8. currency", "")
                })

            logger.info(f"Found {len(results)} matches for '{keywords}'")
            return results

        except Exception as e:
            logger.error(f"Error searching symbol '{keywords}': {e}")
            raise


# Convenience functions
def get_news(ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Quick function to get news for a ticker.

    Args:
        ticker: Stock ticker symbol
        limit: Number of articles to retrieve

    Returns:
        List of news articles with sentiment
    """
    provider = AlphaVantageProvider()
    return provider.get_news_sentiment(tickers=[ticker], limit=limit)


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    """
    Quick function to get company fundamentals.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dictionary with company overview and fundamentals
    """
    provider = AlphaVantageProvider()
    return provider.get_company_overview(ticker)


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    provider = AlphaVantageProvider()

    # Test news sentiment
    print("\n=== Testing News Sentiment ===")
    try:
        news = provider.get_news_sentiment(tickers=["NVDA"], limit=5)
        for article in news[:2]:
            print(f"\nTitle: {article['title']}")
            print(f"Sentiment: {article['overall_sentiment_label']} ({article['overall_sentiment_score']})")
            print(f"Source: {article['source']}")
    except Exception as e:
        print(f"Error: {e}")

    # Test company overview
    print("\n=== Testing Company Overview ===")
    try:
        overview = provider.get_company_overview("AAPL")
        print(f"Company: {overview.get('name')}")
        print(f"Sector: {overview.get('sector')}")
        print(f"PE Ratio: {overview.get('pe_ratio')}")
        print(f"Market Cap: {overview.get('market_cap')}")
    except Exception as e:
        print(f"Error: {e}")

    # Test symbol search
    print("\n=== Testing Symbol Search ===")
    try:
        results = provider.search_symbol("Tesla")
        for match in results[:3]:
            print(f"{match['symbol']}: {match['name']} ({match['region']})")
    except Exception as e:
        print(f"Error: {e}")
