"""
Market Analyst Agent
Analyzes technical indicators, price action, and market trends
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from utils.prompts import (
    MARKET_ANALYST_SYSTEM,
    MARKET_ANALYST_USER,
    format_indicators_summary
)

logger = logging.getLogger(__name__)


class MarketAnalyst(BaseAgent):
    """
    Market Analyst agent specializing in technical analysis.
    Uses GPT-4o-mini for fast, cost-effective analysis.
    """

    def __init__(self):
        super().__init__(
            name="Market Analyst",
            role="Technical analysis and price action expert",
            model="gpt-4o-mini",  # Fast and cheap for data collection
            temperature=0.3,  # Lower temp for more consistent technical analysis
            max_tokens=1500
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and technical indicators.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - price_data: DataFrame with OHLCV data
                - indicators: Dict with technical indicator values
                - indicator_summary: Dict with signals and analysis

        Returns:
            Analysis report with technical outlook
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            price_data = data.get('price_data')
            indicators = data.get('indicators', {})
            summary = data.get('indicator_summary', {})

            if price_data is None or price_data.empty:
                raise ValueError("No price data provided")

            # Extract current price and recent action
            latest_price = price_data['Close'].iloc[-1]
            prev_price = price_data['Close'].iloc[-2] if len(price_data) > 1 else latest_price
            price_change_pct = ((latest_price - prev_price) / prev_price) * 100

            # Format price action
            price_action = self._format_price_action(price_data)

            # Format indicators
            indicators_text = format_indicators_summary(indicators)

            # Create user prompt
            user_prompt = MARKET_ANALYST_USER.format(
                symbol=symbol,
                current_price=f"{latest_price:.2f}",
                price_change_pct=f"{price_change_pct:+.2f}",
                indicators_summary=indicators_text,
                price_action=price_action
            )

            # Get analysis from LLM
            logger.info(f"Analyzing technical outlook for {symbol}")
            analysis = self._call_llm(
                system_prompt=MARKET_ANALYST_SYSTEM,
                user_prompt=user_prompt
            )

            # Create report
            report = self._create_report(
                analysis=analysis,
                metadata={
                    'symbol': symbol,
                    'current_price': float(latest_price),
                    'price_change_pct': float(price_change_pct),
                    'signals': summary.get('signals', []),
                    'analysis_type': 'technical'
                }
            )

            logger.info(f"✅ Market analysis complete for {symbol}")
            return report

        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            raise

    def _format_price_action(self, price_data: pd.DataFrame, days: int = 10) -> str:
        """Format recent price action for prompt."""
        recent = price_data.tail(days)

        lines = []
        lines.append(f"Last {days} trading days:")

        for idx, row in recent.iterrows():
            date = idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx)
            close = row['Close']
            change = ((row['Close'] - row['Open']) / row['Open']) * 100
            lines.append(f"  {date}: ${close:.2f} ({change:+.2f}%)")

        # Add price range
        high = recent['High'].max()
        low = recent['Low'].min()
        lines.append(f"\n{days}-day range: ${low:.2f} - ${high:.2f}")

        return "\n".join(lines)


if __name__ == "__main__":
    # Test the Market Analyst
    logging.basicConfig(level=logging.INFO)

    from data import get_stock_data, TechnicalIndicators

    print("\n=== Testing Market Analyst ===")

    # Get data
    symbol = "AAPL"
    print(f"\nFetching data for {symbol}...")
    price_data = get_stock_data(symbol, days_back=100)

    # Calculate indicators
    calc = TechnicalIndicators()
    data_with_indicators = calc.calculate_all(price_data)
    indicators = calc.get_latest_indicators(price_data)
    summary = calc.get_indicator_summary(price_data)

    # Create analyst
    analyst = MarketAnalyst()

    # Run analysis
    print(f"\nRunning technical analysis...")
    result = analyst.analyze({
        'symbol': symbol,
        'price_data': price_data,
        'indicators': indicators,
        'indicator_summary': summary
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nCurrent Price: ${result['metadata']['current_price']:.2f}")
    print(f"Price Change: {result['metadata']['price_change_pct']:+.2f}%")
    print(f"\nSignals Detected: {len(result['metadata']['signals'])}")
    for signal in result['metadata']['signals']:
        print(f"  - {signal}")
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
