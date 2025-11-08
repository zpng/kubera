"""
News Analyst Agent
Analyzes news sentiment and market-moving headlines
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, List, Optional

from agents.base_agent import BaseAgent
from utils.prompts import (
    NEWS_ANALYST_SYSTEM,
    NEWS_ANALYST_USER,
    format_news_articles
)

logger = logging.getLogger(__name__)


class NewsAnalyst(BaseAgent):
    """
    News Analyst agent specializing in news sentiment and market impact.
    Uses GPT-4o-mini for fast news processing.
    """

    def __init__(self):
        super().__init__(
            name="News Analyst",
            role="News sentiment and market impact expert",
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=1500
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze news articles and sentiment.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - news_articles: List of news article dictionaries

        Returns:
            Analysis report with news sentiment and impact assessment
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            news_articles = data.get('news_articles', [])

            if not news_articles:
                logger.warning(f"No news articles provided for {symbol}")
                # Return minimal analysis
                return self._create_report(
                    analysis="No recent news articles available for analysis. Market moving news should be monitored closely.",
                    metadata={
                        'symbol': symbol,
                        'article_count': 0,
                        'sentiment': 'Unknown',
                        'analysis_type': 'news'
                    }
                )

            # Format news articles for prompt
            news_text = format_news_articles(news_articles, limit=15)

            # Calculate sentiment distribution
            sentiment_counts = self._calculate_sentiment_distribution(news_articles)

            # Create user prompt
            user_prompt = NEWS_ANALYST_USER.format(
                symbol=symbol,
                news_articles=news_text
            )

            # Add sentiment distribution to prompt
            user_prompt += f"\n\nSentiment Distribution:\n{self._format_sentiment_dist(sentiment_counts)}"

            # Get analysis from LLM
            logger.info(f"Analyzing news sentiment for {symbol} ({len(news_articles)} articles)")
            analysis = self._call_llm(
                system_prompt=NEWS_ANALYST_SYSTEM,
                user_prompt=user_prompt
            )

            # Determine overall sentiment
            overall_sentiment = self._determine_overall_sentiment(sentiment_counts)

            # Create report
            report = self._create_report(
                analysis=analysis,
                metadata={
                    'symbol': symbol,
                    'article_count': len(news_articles),
                    'sentiment_distribution': sentiment_counts,
                    'overall_sentiment': overall_sentiment,
                    'analysis_type': 'news'
                }
            )

            logger.info(f"✅ News analysis complete for {symbol} ({overall_sentiment} sentiment)")
            return report

        except Exception as e:
            logger.error(f"News analysis failed: {e}")
            raise

    def _calculate_sentiment_distribution(self, articles: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of sentiment labels."""
        counts = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0, 'Unknown': 0}

        for article in articles:
            sentiment = article.get('overall_sentiment_label', 'Unknown')
            if sentiment in counts:
                counts[sentiment] += 1
            else:
                counts['Unknown'] += 1

        return counts

    def _format_sentiment_dist(self, counts: Dict[str, int]) -> str:
        """Format sentiment distribution for prompt."""
        total = sum(counts.values())
        if total == 0:
            return "No sentiment data available"

        lines = []
        for sentiment, count in counts.items():
            if count > 0:
                pct = (count / total) * 100
                lines.append(f"- {sentiment}: {count} ({pct:.1f}%)")

        return "\n".join(lines)

    def _determine_overall_sentiment(self, counts: Dict[str, int]) -> str:
        """Determine overall sentiment from distribution."""
        total = sum(counts.values())
        if total == 0:
            return 'Unknown'

        bullish_pct = (counts.get('Bullish', 0) / total) * 100
        bearish_pct = (counts.get('Bearish', 0) / total) * 100

        if bullish_pct > 60:
            return 'Very Bullish'
        elif bullish_pct > 40:
            return 'Bullish'
        elif bearish_pct > 60:
            return 'Very Bearish'
        elif bearish_pct > 40:
            return 'Bearish'
        else:
            return 'Neutral'


if __name__ == "__main__":
    # Test the News Analyst
    logging.basicConfig(level=logging.INFO)

    from data import get_news

    print("\n=== Testing News Analyst ===")

    # Get news data
    symbol = "AAPL"
    print(f"\nFetching news for {symbol}...")
    news_articles = get_news([symbol], limit=20)

    print(f"Found {len(news_articles)} news articles")

    # Create analyst
    analyst = NewsAnalyst()

    # Run analysis
    print(f"\nRunning news sentiment analysis...")
    result = analyst.analyze({
        'symbol': symbol,
        'news_articles': news_articles
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nArticles Analyzed: {result['metadata']['article_count']}")
    print(f"Overall Sentiment: {result['metadata']['overall_sentiment']}")
    print(f"\nSentiment Distribution:")
    for sentiment, count in result['metadata']['sentiment_distribution'].items():
        if count > 0:
            print(f"  - {sentiment}: {count}")
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
