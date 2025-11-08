"""
Sentiment Analyst Agent
Analyzes market sentiment and investor psychology
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompts import SENTIMENT_ANALYST_SYSTEM, SENTIMENT_ANALYST_USER

logger = logging.getLogger(__name__)


class SentimentAnalyst(BaseAgent):
    """
    Sentiment Analyst agent specializing in market psychology and investor behavior.
    Uses GPT-4o-mini for sentiment analysis.
    """

    def __init__(self):
        super().__init__(
            name="Sentiment Analyst",
            role="Market sentiment and investor psychology expert",
            model="gpt-4o-mini",
            temperature=0.5,
            max_tokens=1500
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market sentiment and investor psychology.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - indicators: Dict with technical indicators
                - price_data: DataFrame with price data
                - news_sentiment: Overall news sentiment

        Returns:
            Analysis report with sentiment assessment
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            indicators = data.get('indicators', {})
            price_data = data.get('price_data')
            news_sentiment = data.get('news_sentiment', 'Unknown')

            # Extract sentiment indicators
            rsi = indicators.get('rsi', 'N/A')

            # Price vs moving averages
            if price_data is not None and not price_data.empty:
                current_price = price_data['Close'].iloc[-1]
                sma_50 = indicators.get('sma_50', 0)
                sma_200 = indicators.get('sma_200', 0)

                price_vs_sma50 = "Above" if current_price > sma_50 and sma_50 > 0 else "Below"
                price_vs_sma200 = "Above" if current_price > sma_200 and sma_200 > 0 else "Below"

                # Calculate recent volatility
                recent_data = price_data.tail(20)
                volatility = recent_data['Close'].std() / recent_data['Close'].mean() * 100

                # Price momentum
                price_change_5d = ((current_price - price_data['Close'].iloc[-6]) / price_data['Close'].iloc[-6] * 100) if len(price_data) > 5 else 0
                price_change_20d = ((current_price - price_data['Close'].iloc[-21]) / price_data['Close'].iloc[-21] * 100) if len(price_data) > 20 else 0

                momentum = f"5-day: {price_change_5d:+.2f}%, 20-day: {price_change_20d:+.2f}%"
            else:
                price_vs_sma50 = "Unknown"
                price_vs_sma200 = "Unknown"
                volatility = "Unknown"
                momentum = "Data not available"

            # Determine sentiment level
            sentiment_level = self._calculate_sentiment_level(
                rsi if isinstance(rsi, (int, float)) else 50,
                price_vs_sma50,
                price_vs_sma200
            )

            # Create user prompt
            user_prompt = SENTIMENT_ANALYST_USER.format(
                symbol=symbol,
                rsi=rsi,
                price_vs_sma50=price_vs_sma50,
                price_vs_sma200=price_vs_sma200,
                volatility=f"{volatility:.2f}%" if isinstance(volatility, (int, float)) else volatility,
                news_sentiment=news_sentiment,
                price_momentum=momentum
            )

            # Get analysis from LLM
            logger.info(f"Analyzing market sentiment for {symbol}")
            analysis = self._call_llm(
                system_prompt=SENTIMENT_ANALYST_SYSTEM,
                user_prompt=user_prompt
            )

            # Create report
            report = self._create_report(
                analysis=analysis,
                metadata={
                    'symbol': symbol,
                    'sentiment_level': sentiment_level,
                    'rsi': float(rsi) if isinstance(rsi, (int, float)) else None,
                    'price_vs_sma50': price_vs_sma50,
                    'price_vs_sma200': price_vs_sma200,
                    'news_sentiment': news_sentiment,
                    'analysis_type': 'sentiment'
                }
            )

            logger.info(f"✅ Sentiment analysis complete for {symbol} ({sentiment_level})")
            return report

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise

    def _calculate_sentiment_level(
        self,
        rsi: float,
        price_vs_sma50: str,
        price_vs_sma200: str
    ) -> str:
        """Calculate overall sentiment level from indicators."""
        score = 0

        # RSI sentiment
        if rsi > 70:
            score -= 2  # Extreme greed
        elif rsi > 60:
            score -= 1  # Greed
        elif rsi < 30:
            score += 2  # Extreme fear
        elif rsi < 40:
            score += 1  # Fear

        # Moving average sentiment
        if price_vs_sma50 == "Below":
            score += 1
        else:
            score -= 1

        if price_vs_sma200 == "Below":
            score += 1
        else:
            score -= 1

        # Classify sentiment
        if score >= 3:
            return "Extreme Fear"
        elif score >= 1:
            return "Fear"
        elif score <= -3:
            return "Extreme Greed"
        elif score <= -1:
            return "Greed"
        else:
            return "Neutral"


if __name__ == "__main__":
    # Test the Sentiment Analyst
    logging.basicConfig(level=logging.INFO)

    from data import get_stock_data, TechnicalIndicators

    print("\n=== Testing Sentiment Analyst ===")

    # Get data
    symbol = "AAPL"
    print(f"\nFetching data for {symbol}...")
    price_data = get_stock_data(symbol, days_back=100)

    # Calculate indicators
    calc = TechnicalIndicators()
    indicators = calc.get_latest_indicators(price_data)

    # Create analyst
    analyst = SentimentAnalyst()

    # Run analysis
    print(f"\nRunning sentiment analysis...")
    result = analyst.analyze({
        'symbol': symbol,
        'price_data': price_data,
        'indicators': indicators,
        'news_sentiment': 'Bullish'  # Mock news sentiment
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nSentiment Level: {result['metadata']['sentiment_level']}")
    print(f"RSI: {result['metadata']['rsi']}")
    print(f"Price vs 50-day MA: {result['metadata']['price_vs_sma50']}")
    print(f"Price vs 200-day MA: {result['metadata']['price_vs_sma200']}")
    print(f"News Sentiment: {result['metadata']['news_sentiment']}")
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
