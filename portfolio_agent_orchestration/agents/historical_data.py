"""
Historical Data Agent
Pulls historical stock data using yfinance
Model: deepseek/deepseek-chat-v3.1 (fast, good with tools)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from .base_agent import BaseAgent
from ..config import AGENT_MODELS, CACHE_DIR

logger = logging.getLogger(__name__)


class HistoricalDataAgent(BaseAgent):
    """
    Agent responsible for fetching historical stock data
    """
    
    def __init__(self):
        super().__init__(
            name="HistoricalData",
            model=AGENT_MODELS["historical_data"],
            role="Historical stock data fetcher and analyzer",
            temperature=0.3
        )
        self.cache_dir = CACHE_DIR / "historical"
        self.cache_dir.mkdir(exist_ok=True)
    
    def fetch_historical_data(
        self,
        symbol: str,
        period: str = "1y",  # Changed from 6mo to 1y for accurate 52-week range
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a symbol

        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various return metrics"""
        if df.empty:
            return {}
        
        try:
            latest_close = df['Close'].iloc[-1]
            oldest_close = df['Close'].iloc[0]
            
            total_return = ((latest_close - oldest_close) / oldest_close) * 100
            
            # Daily returns
            daily_returns = df['Close'].pct_change().dropna()
            
            metrics = {
                "current_price": round(float(latest_close), 2),
                "period_return_pct": round(float(total_return), 2),
                "volatility": round(float(daily_returns.std() * 100), 2),
                "avg_daily_return_pct": round(float(daily_returns.mean() * 100), 4),
                "max_price": round(float(df['Close'].max()), 2),
                "min_price": round(float(df['Close'].min()), 2),
                "avg_volume": int(df['Volume'].mean()),
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return {}
    
    def identify_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify price trends"""
        if df.empty or len(df) < 20:
            return {}
        
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean() if len(df) >= 50 else None
            
            current_price = df['Close'].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1]
            
            trend = "bullish" if current_price > sma_20 else "bearish"
            
            # Price momentum (last 30 days)
            if len(df) >= 30:
                price_30d_ago = df['Close'].iloc[-30]
                momentum_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
            else:
                momentum_30d = 0
            
            trends = {
                "trend": trend,
                "sma_20": round(float(sma_20), 2),
                "distance_from_sma20_pct": round(((current_price - sma_20) / sma_20) * 100, 2),
                "momentum_30d_pct": round(float(momentum_30d), 2),
            }
            
            if len(df) >= 50 and df['SMA_50'].iloc[-1] is not None:
                sma_50 = df['SMA_50'].iloc[-1]
                trends["sma_50"] = round(float(sma_50), 2)
                trends["golden_cross"] = sma_20 > sma_50
            
            return trends
            
        except Exception as e:
            logger.error(f"Error identifying trends: {e}")
            return {}
    
    def process_symbol(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Process a single stock symbol"""
        logger.info(f"Processing historical data for {symbol}")
        
        # Fetch data
        df = self.fetch_historical_data(symbol, period=period)

        # Also get ticker info for 52-week range and YTD
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except:
            info = {}
        
        if df.empty:
            return {
                "symbol": symbol,
                "status": "error",
                "error": "No data available"
            }
        
        # Calculate metrics
        returns = self.calculate_returns(df)
        trends = self.identify_trends(df)
        
        # Add accurate 52-week range from yfinance info
        if info:
            returns["fifty_two_week_high"] = info.get("fiftyTwoWeekHigh", returns.get("max_price"))
            returns["fifty_two_week_low"] = info.get("fiftyTwoWeekLow", returns.get("min_price"))

        # Calculate YTD return (Year-to-Date: from Jan 1 to present)
        if not df.empty:
            try:
                # Get the current year start (timezone-aware if df.index is timezone-aware)
                current_year = datetime.now().year
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    # DataFrame has timezone-aware index
                    year_start = pd.Timestamp(f'{current_year}-01-01', tz=df.index.tz)
                else:
                    # DataFrame has naive index
                    year_start = pd.Timestamp(f'{current_year}-01-01')

                # Filter data from year start to present
                df_ytd = df[df.index >= year_start]

                if len(df_ytd) > 1:  # Need at least 2 data points
                    ytd_return = ((df_ytd['Close'].iloc[-1] / df_ytd['Close'].iloc[0] - 1) * 100)
                    returns["ytd_return_pct"] = round(float(ytd_return), 2)
                    logger.info(f"Calculated YTD return for {symbol}: {returns['ytd_return_pct']}% (from {df_ytd.index[0].strftime('%Y-%m-%d')} to {df_ytd.index[-1].strftime('%Y-%m-%d')})")
                else:
                    # If no data from Jan 1, use period return
                    logger.warning(f"Insufficient YTD data for {symbol}, using period return")
                    returns["ytd_return_pct"] = returns.get("period_return_pct", 0)
            except Exception as e:
                logger.error(f"Error calculating YTD return for {symbol}: {e}")
                # Fallback to period return
                returns["ytd_return_pct"] = returns.get("period_return_pct", 0)
        
        result = {
            "symbol": symbol,
            "status": "success",
            "data_points": len(df),
            "period": period,
            "metrics": returns,
            "trends": trends,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with stock_symbols
            
        Returns:
            Updated state with historical data
        """
        logger.info(f"[{self.name}] Starting historical data collection...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        historical_data = {}
        
        for symbol in symbols:
            data = self.process_symbol(symbol)
            historical_data[symbol] = data
        
        # Update state
        state["historical_data"] = historical_data
        state["historical_data_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Collected data for {len(historical_data)} symbols")
        return state


# Test function
def test_historical_data_agent():
    """Test the HistoricalDataAgent"""
    import json
    
    print("Testing Historical Data Agent...")
    
    agent = HistoricalDataAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["AAPL", "MSFT", "NVDA"]
    }
    
    result_state = agent.process(state)
    
    print(f"\nHistorical Data Results:")
    for symbol, data in result_state.get("historical_data", {}).items():
        print(f"\n{symbol}:")
        print(f"  Status: {data['status']}")
        if data['status'] == 'success':
            print(f"  Current Price: ${data['metrics']['current_price']}")
            print(f"  Period Return: {data['metrics']['period_return_pct']}%")
            print(f"  Volatility: {data['metrics']['volatility']}%")
            print(f"  Trend: {data['trends']['trend']}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_historical_data_agent()

