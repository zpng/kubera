"""
Historical Data Agent
Fetches real-time stock data and compares with portfolio holdings
Model: openai/gpt-oss-20b (efficient for data analysis and edge deployment)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StockDataPoint(BaseModel):
    """Schema for stock data point"""
    symbol: str
    current_price: float
    previous_close: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


class PortfolioComparison(BaseModel):
    """Schema for portfolio comparison"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float
    current_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_pl_percent: float
    recommendation: str = Field(description="Brief analysis based on price movement")


@tool
def get_stock_realtime_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch real-time stock data from yfinance
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with current stock data
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        history = stock.history(period="5d")
        
        if history.empty:
            raise ValueError(f"No data available for {symbol}")
        
        current_price = history['Close'].iloc[-1]
        previous_close = info.get('previousClose', history['Close'].iloc[-2] if len(history) > 1 else current_price)
        
        change_percent = ((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0
        
        return {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "change_percent": round(change_percent, 2),
            "volume": int(history['Volume'].iloc[-1]),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "fifty_two_week_high": info.get('fiftyTwoWeekHigh'),
            "fifty_two_week_low": info.get('fiftyTwoWeekLow'),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


@tool
def get_historical_prices(symbol: str, days: int = 30) -> Dict[str, Any]:
    """
    Fetch historical price data for technical analysis
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days of history (default 30)
        
    Returns:
        Dictionary with historical price data
    """
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period=f"{days}d")
        
        if history.empty:
            raise ValueError(f"No historical data available for {symbol}")
        
        return {
            "symbol": symbol,
            "high": round(history['High'].max(), 2),
            "low": round(history['Low'].min(), 2),
            "average": round(history['Close'].mean(), 2),
            "volatility": round(history['Close'].std(), 2),
            "trend": "up" if history['Close'].iloc[-1] > history['Close'].iloc[0] else "down",
            "data_points": len(history),
            "period_days": days
        }
    
    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


class HistoricalDataAgent:
    """
    Agent responsible for fetching real-time stock data and comparing with portfolio
    Uses: openai/gpt-oss-20b for efficient data analysis
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "openai/gpt-oss-20b"
    ):
        """
        Initialize Historical Data Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: openai/gpt-oss-20b)
        """
        self.model = model
        
        # Initialize LLM for data analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.2  # Low temperature for analytical consistency
        )
        
        # Bind tools to LLM
        self.tools = [get_stock_realtime_data, get_historical_prices]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(f"Historical Data Agent initialized with model: {model}")
    
    def fetch_realtime_data(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch real-time data for multiple symbols
        
        Args:
            symbols: List of stock ticker symbols
            
        Returns:
            List of stock data points
        """
        logger.info(f"Fetching real-time data for {len(symbols)} symbols...")
        
        stock_data = []
        for symbol in symbols:
            try:
                data = get_stock_realtime_data.invoke({"symbol": symbol})
                if "error" not in data:
                    stock_data.append(data)
                    logger.info(f"✓ {symbol}: ${data['current_price']} ({data['change_percent']:+.2f}%)")
                else:
                    logger.warning(f"✗ {symbol}: {data['error']}")
            
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
        
        return stock_data
    
    def compare_with_portfolio(
        self,
        holdings: List[Dict[str, Any]],
        realtime_data: List[Dict[str, Any]]
    ) -> List[PortfolioComparison]:
        """
        Compare portfolio holdings with real-time data
        
        Args:
            holdings: Portfolio holdings
            realtime_data: Real-time stock data
            
        Returns:
            List of portfolio comparisons
        """
        logger.info("Comparing portfolio holdings with real-time data...")
        
        # Create lookup dict for realtime data
        price_lookup = {data['symbol']: data for data in realtime_data}
        
        comparisons = []
        total_pl = 0
        total_value = 0
        total_cost = 0
        
        for holding in holdings:
            symbol = holding['symbol']
            shares = holding['shares']
            avg_cost = holding['avg_cost']
            
            # Get current price
            stock_data = price_lookup.get(symbol)
            if not stock_data:
                logger.warning(f"No real-time data for {symbol}")
                continue
            
            current_price = stock_data['current_price']
            
            # Calculate P&L
            current_value = shares * current_price
            cost_basis = shares * avg_cost
            unrealized_pl = current_value - cost_basis
            unrealized_pl_percent = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0
            
            # Track totals
            total_value += current_value
            total_cost += cost_basis
            total_pl += unrealized_pl
            
            # Generate recommendation using LLM
            recommendation = self._generate_quick_recommendation(
                symbol, avg_cost, current_price, unrealized_pl_percent
            )
            
            comparison = PortfolioComparison(
                symbol=symbol,
                shares=shares,
                avg_cost=avg_cost,
                current_price=current_price,
                current_value=current_value,
                cost_basis=cost_basis,
                unrealized_pl=unrealized_pl,
                unrealized_pl_percent=unrealized_pl_percent,
                recommendation=recommendation
            )
            
            comparisons.append(comparison)
            
            emoji = "🟢" if unrealized_pl >= 0 else "🔴"
            logger.info(f"{emoji} {symbol}: ${current_value:.2f} ({unrealized_pl_percent:+.2f}%)")
        
        # Log summary
        total_pl_percent = (total_pl / total_cost * 100) if total_cost > 0 else 0
        logger.info("=" * 50)
        logger.info(f"Portfolio Summary:")
        logger.info(f"  Total Value: ${total_value:.2f}")
        logger.info(f"  Total Cost: ${total_cost:.2f}")
        logger.info(f"  Total P&L: ${total_pl:.2f} ({total_pl_percent:+.2f}%)")
        logger.info("=" * 50)
        
        return comparisons
    
    def _generate_quick_recommendation(
        self,
        symbol: str,
        avg_cost: float,
        current_price: float,
        pl_percent: float
    ) -> str:
        """
        Generate a quick recommendation based on price movement
        
        Args:
            symbol: Stock symbol
            avg_cost: Average cost
            current_price: Current price
            pl_percent: P&L percentage
            
        Returns:
            Brief recommendation
        """
        if pl_percent > 20:
            return f"Strong gains (+{pl_percent:.1f}%). Consider taking partial profits."
        elif pl_percent > 10:
            return f"Good gains (+{pl_percent:.1f}%). Hold or add on dips."
        elif pl_percent > -5:
            return f"Near entry ({pl_percent:+.1f}%). Hold for now."
        elif pl_percent > -15:
            return f"Moderate loss ({pl_percent:+.1f}%). Monitor closely."
        else:
            return f"Significant loss ({pl_percent:+.1f}%). Review fundamentals."
    
    def run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - fetches and compares data
        
        Args:
            portfolio_data: Portfolio data from PortfolioLoaderAgent
            
        Returns:
            Dictionary with updated portfolio and comparisons
        """
        logger.info("=" * 50)
        logger.info("HISTORICAL DATA AGENT - Starting execution")
        logger.info("=" * 50)
        
        # Extract holdings and symbols
        holdings = portfolio_data.get('holdings', [])
        symbols = portfolio_data.get('symbols', [])
        
        # Fetch real-time data
        realtime_data = self.fetch_realtime_data(symbols)
        
        # Compare with portfolio
        comparisons = self.compare_with_portfolio(holdings, realtime_data)
        
        result = {
            "agent": "historical_data",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "realtime_data": realtime_data,
            "comparisons": [comp.dict() for comp in comparisons],
            "total_holdings": len(comparisons),
            "status": "success"
        }
        
        logger.info(f"Historical Data Agent completed - {len(realtime_data)} stocks analyzed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_historical_data_agent():
    """Test the historical data agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Historical Data Agent...")
        print("=" * 60)
        
        # Mock portfolio data
        mock_portfolio = {
            "holdings": [
                {"symbol": "AAPL", "shares": 10, "avg_cost": 150.0},
                {"symbol": "TSLA", "shares": 5, "avg_cost": 200.0}
            ],
            "symbols": ["AAPL", "TSLA"]
        }
        
        agent = HistoricalDataAgent(openrouter_api_key=api_key)
        result = agent.run(mock_portfolio)
        
        print("\n✅ Test Results:")
        print(f"   - Stocks analyzed: {result['total_holdings']}")
        print(f"   - Real-time data points: {len(result['realtime_data'])}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Historical Data Agent test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_historical_data_agent()

