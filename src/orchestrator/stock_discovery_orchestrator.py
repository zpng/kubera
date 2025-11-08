"""
Stock Discovery Orchestrator
Combines stock discoveries from YouTube, X/Twitter, and News
Ranks stocks and runs full analysis pipeline on top candidates
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

from agents.discovery.youtube_discovery import YouTubeStockDiscovery
from agents.discovery.x_discovery import XStockDiscovery

logger = logging.getLogger(__name__)


class StockDiscoveryOrchestrator:
    """
    Orchestrates stock discovery from multiple sources and ranks candidates.
    """

    def __init__(self):
        self.youtube_discovery = YouTubeStockDiscovery()
        self.x_discovery = XStockDiscovery()
        logger.info("Stock Discovery Orchestrator initialized")

    def discover_trending_stocks(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Discover trending stocks from all sources and rank them.

        Args:
            top_n: Number of top stocks to return

        Returns:
            List of top trending stocks with combined scores
        """
        try:
            logger.info("="*60)
            logger.info("AUTONOMOUS STOCK DISCOVERY STARTED")
            logger.info("="*60)

            # Discover from all sources
            discoveries = {}

            # YouTube discovery
            logger.info("\n🎥 Discovering from YouTube...")
            youtube_result = self.youtube_discovery.discover_stocks()
            discoveries['youtube'] = youtube_result
            logger.info(f"   Found {len(youtube_result.get('stocks', []))} stocks from YouTube")

            # X/Twitter discovery
            logger.info("\n🐦 Discovering from X/Twitter...")
            x_result = self.x_discovery.discover_stocks()
            discoveries['x'] = x_result
            logger.info(f"   Found {len(x_result.get('stocks', []))} stocks from X/Twitter")

            # Note: News discovery uses existing Alpha Vantage integration
            logger.info("\n📰 News discovery integrated via Alpha Vantage (existing)")

            # Aggregate and rank stocks
            logger.info("\n📊 Aggregating and ranking discovered stocks...")
            ranked_stocks = self._aggregate_and_rank(discoveries, top_n)

            logger.info(f"\n✅ Discovery complete: {len(ranked_stocks)} top stocks identified")
            logger.info("="*60)

            return ranked_stocks

        except Exception as e:
            logger.error(f"Stock discovery failed: {e}", exc_info=True)
            raise

    def _aggregate_and_rank(
        self,
        discoveries: Dict[str, Dict[str, Any]],
        top_n: int
    ) -> List[Dict[str, Any]]:
        """
        Aggregate stocks from all sources and rank by combined score.
        """
        # Collect all stocks with their scores
        stock_data = defaultdict(lambda: {
            'symbol': '',
            'total_score': 0,
            'sources': [],
            'youtube_buzz': 0,
            'x_viral': 0,
            'sentiment_scores': [],
            'key_themes': set(),
            'key_catalysts': set()
        })

        # Process YouTube discoveries
        if 'youtube' in discoveries:
            for stock in discoveries['youtube'].get('stocks', []):
                symbol = stock['symbol']
                stock_data[symbol]['symbol'] = symbol
                stock_data[symbol]['sources'].append('YouTube')

                # YouTube score: buzz_level * mentions * confidence
                youtube_score = (
                    stock.get('buzz_level', 0) *
                    stock.get('mentions', 0) *
                    stock.get('confidence', 0.5)
                )
                stock_data[symbol]['youtube_buzz'] = stock.get('buzz_level', 0)
                stock_data[symbol]['total_score'] += youtube_score

                # Add themes
                for theme in stock.get('key_themes', []):
                    stock_data[symbol]['key_themes'].add(theme)

                # Sentiment
                sentiment = stock.get('sentiment', 'neutral')
                sentiment_value = {'bullish': 1, 'neutral': 0, 'bearish': -1}.get(sentiment, 0)
                stock_data[symbol]['sentiment_scores'].append(sentiment_value)

        # Process X/Twitter discoveries
        if 'x' in discoveries:
            for stock in discoveries['x'].get('stocks', []):
                symbol = stock['symbol']
                stock_data[symbol]['symbol'] = symbol
                stock_data[symbol]['sources'].append('X/Twitter')

                # X score: viral_score * (post_count/10) * confidence
                x_score = (
                    stock.get('viral_score', 0) *
                    (stock.get('post_count', 0) / 10) *
                    stock.get('confidence', 0.5)
                )
                stock_data[symbol]['x_viral'] = stock.get('viral_score', 0)
                stock_data[symbol]['total_score'] += x_score

                # Add catalysts
                for catalyst in stock.get('key_catalysts', []):
                    stock_data[symbol]['key_catalysts'].add(catalyst)

                # Sentiment
                sentiment = stock.get('sentiment', 'neutral')
                sentiment_value = {'bullish': 1, 'neutral': 0, 'bearish': -1}.get(sentiment, 0)
                stock_data[symbol]['sentiment_scores'].append(sentiment_value)

        # Calculate final scores and sentiment
        ranked_stocks = []
        for symbol, data in stock_data.items():
            # Calculate average sentiment
            if data['sentiment_scores']:
                avg_sentiment = sum(data['sentiment_scores']) / len(data['sentiment_scores'])
                if avg_sentiment > 0.3:
                    overall_sentiment = 'bullish'
                elif avg_sentiment < -0.3:
                    overall_sentiment = 'bearish'
                else:
                    overall_sentiment = 'neutral'
            else:
                overall_sentiment = 'neutral'

            # Bonus for multiple sources
            source_bonus = len(data['sources']) * 5

            ranked_stocks.append({
                'symbol': symbol,
                'score': data['total_score'] + source_bonus,
                'sources': data['sources'],
                'source_count': len(data['sources']),
                'youtube_buzz': data['youtube_buzz'],
                'x_viral': data['x_viral'],
                'overall_sentiment': overall_sentiment,
                'key_themes': list(data['key_themes']),
                'key_catalysts': list(data['key_catalysts'])
            })

        # Sort by score (descending) and return top N
        ranked_stocks.sort(key=lambda x: x['score'], reverse=True)
        return ranked_stocks[:top_n]


if __name__ == "__main__":
    # Test stock discovery orchestrator
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("AUTONOMOUS STOCK DISCOVERY SYSTEM - TEST")
    print("="*80)

    # Create orchestrator
    orchestrator = StockDiscoveryOrchestrator()

    # Discover top stocks
    print("\nRunning autonomous discovery...")
    top_stocks = orchestrator.discover_trending_stocks(top_n=5)

    # Display results
    print("\n" + "="*80)
    print("TOP TRENDING STOCKS")
    print("="*80)

    for i, stock in enumerate(top_stocks, 1):
        print(f"\n#{i}. {stock['symbol']} (Score: {stock['score']:.1f})")
        print(f"   Sources: {', '.join(stock['sources'])} ({stock['source_count']} sources)")
        print(f"   Sentiment: {stock['overall_sentiment']}")
        if stock['youtube_buzz'] > 0:
            print(f"   YouTube Buzz: {stock['youtube_buzz']}/10")
        if stock['x_viral'] > 0:
            print(f"   X Viral Score: {stock['x_viral']}/10")
        if stock['key_themes']:
            print(f"   Themes: {', '.join(stock['key_themes'][:3])}")
        if stock['key_catalysts']:
            print(f"   Catalysts: {', '.join(stock['key_catalysts'][:3])}")

    print("\n" + "="*80)
    print("✅ Stock discovery complete!")
    print("="*80)
