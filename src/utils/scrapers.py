"""
Web Scrapers for Stock Discovery
Uses yt-dlp for YouTube and snscrape for Twitter/X
"""

import logging
from typing import Dict, Any, List, Optional
import subprocess
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class YouTubeScraper:
    """
    Scrapes YouTube for finance-related videos using yt-dlp.
    No API key required, unlimited searches.
    """

    def search_finance_videos(
        self,
        query: str = "stock market today",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search YouTube for finance videos.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of video metadata dictionaries
        """
        try:
            # Use yt-dlp to search YouTube
            search_query = f"ytsearch{max_results}:{query}"

            cmd = [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--quiet",
                search_query
            ]

            logger.info(f"Searching YouTube for: {query}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                logger.error(f"yt-dlp error: {result.stderr}")
                return []

            # Parse JSON lines
            videos = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    video_data = json.loads(line)
                    videos.append({
                        'title': video_data.get('title', ''),
                        'description': video_data.get('description', '')[:500],  # Truncate
                        'channel': video_data.get('channel', ''),
                        'views': video_data.get('view_count', 0),
                        'upload_date': video_data.get('upload_date', ''),
                        'url': video_data.get('webpage_url', ''),
                        'duration': video_data.get('duration', 0)
                    })
                except json.JSONDecodeError:
                    continue

            logger.info(f"✅ Found {len(videos)} videos from YouTube")
            return videos

        except subprocess.TimeoutExpired:
            logger.error("YouTube search timed out")
            return []
        except Exception as e:
            logger.error(f"YouTube scraping failed: {e}")
            return []

    def get_trending_finance_videos(self) -> List[Dict[str, Any]]:
        """Get trending finance videos from multiple searches."""
        all_videos = []

        # Multiple search queries for better coverage
        queries = [
            "stock market analysis today",
            "best stocks to buy now",
            "stock picks today",
            "market news today"
        ]

        for query in queries:
            videos = self.search_finance_videos(query, max_results=5)
            all_videos.extend(videos)

        # Deduplicate by title
        seen_titles = set()
        unique_videos = []
        for video in all_videos:
            if video['title'] not in seen_titles:
                seen_titles.add(video['title'])
                unique_videos.append(video)

        logger.info(f"✅ Total unique videos: {len(unique_videos)}")
        return unique_videos[:20]  # Return top 20


class TwitterScraper:
    """
    Scrapes Twitter/X for stock-related tweets.
    Note: snscrape has compatibility issues with Python 3.13+
    Using mock high-quality data until alternative scraper is implemented.
    """

    def search_stock_tweets(
        self,
        query: str = "$AAPL OR $TSLA OR $NVDA",
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search Twitter/X for stock-related tweets.

        Args:
            query: Search query (use $ for tickers)
            max_results: Maximum number of tweets

        Returns:
            List of tweet dictionaries
        """
        logger.info(f"Searching X/Twitter for: {query}")
        logger.warning("snscrape not compatible with Python 3.13+ - using high-quality mock data")

        # High-quality mock data representing real trending stock discussions
        # In production, use official X API v2 or alternative scraper
        mock_tweets = [
            {
                "text": "$NVDA breaking out! New AI chip orders from Microsoft. This is going to $500+ 🚀 #AI #Stocks",
                "author": "TechTrader_",
                "likes": 15000,
                "retweets": 3500,
                "replies": 800,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/1"
            },
            {
                "text": "$TSLA delivery numbers disappointing. Competition from BYD heating up. Bearish short-term ⚠️",
                "author": "MarketAnalyst",
                "likes": 8500,
                "retweets": 2100,
                "replies": 650,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/2"
            },
            {
                "text": "$MSFT, $GOOGL, $META - the AI trinity. All breaking out together. Cloud + AI = unstoppable 💪",
                "author": "AI_Investor",
                "likes": 12000,
                "retweets": 2800,
                "replies": 720,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/3"
            },
            {
                "text": "$AAPL iPhone 16 pre-orders crushing expectations. Services revenue accelerating. Bullish! 📈",
                "author": "AppleInsider",
                "likes": 9500,
                "retweets": 1900,
                "replies": 580,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/4"
            },
            {
                "text": "$AMD stealing market share from Intel. New MI300 chips competitive with $NVDA. Watch this one 👀",
                "author": "ChipAnalyst",
                "likes": 7200,
                "retweets": 1650,
                "replies": 490,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/5"
            },
            {
                "text": "$PLTR Palantir gov contracts growing. AI platform adoption accelerating. Hidden gem? 💎",
                "author": "DataStockPicks",
                "likes": 6800,
                "retweets": 1450,
                "replies": 420,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/6"
            },
            {
                "text": "Just loaded more $NVDA. Jensen's keynote next week. Expecting new product announcements 🔥",
                "author": "TechBull",
                "likes": 5500,
                "retweets": 1200,
                "replies": 350,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/7"
            },
            {
                "text": "$META undervalued at these levels. Reels monetization improving. Metaverse narrative changing 🎯",
                "author": "SocialMediaInvestor",
                "likes": 4200,
                "retweets": 950,
                "replies": 280,
                "posted": datetime.now().isoformat(),
                "url": "https://x.com/example/status/8"
            }
        ]

        logger.info(f"✅ Retrieved {len(mock_tweets)} stock tweets (high-quality mock data)")
        return mock_tweets[:max_results]

    def get_trending_stock_tweets(self) -> List[Dict[str, Any]]:
        """Get trending stock tweets from the past 24 hours."""
        # Search for popular stock tickers
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

        # Build query for top stocks
        query = f"($AAPL OR $TSLA OR $NVDA OR $MSFT OR $GOOGL OR $META OR $AMD OR $AMZN) since:{yesterday}"

        tweets = self.search_stock_tweets(query, max_results=100)

        # Filter for high engagement
        high_engagement_tweets = [
            tweet for tweet in tweets
            if (tweet['likes'] + tweet['retweets']) > 10  # At least some engagement
        ]

        logger.info(f"✅ Filtered to {len(high_engagement_tweets)} high-engagement tweets")
        return high_engagement_tweets[:50]  # Return top 50


class MarketauxScraper:
    """
    Fetches stock news from Marketaux API.
    Free tier: 100 requests/day.
    """

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.api_key = api_key or os.getenv('MARKETAUX_API_KEY')
        self.base_url = "https://api.marketaux.com/v1/news/all"

    def get_stock_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get stock news from Marketaux API.

        Args:
            symbols: Optional list of stock symbols to filter by
            limit: Maximum number of articles

        Returns:
            List of news article dictionaries
        """
        try:
            import requests

            params = {
                'api_token': self.api_key,
                'limit': limit,
                'language': 'en'
            }

            if symbols:
                params['symbols'] = ','.join(symbols)

            logger.info(f"Fetching news from Marketaux API...")

            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            articles = data.get('data', [])

            # Format articles
            formatted_articles = []
            for article in articles:
                formatted_articles.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', ''),
                    'published': article.get('published_at', ''),
                    'url': article.get('url', ''),
                    'symbols': article.get('entities', []),
                    'sentiment': article.get('sentiment', 0.0)
                })

            logger.info(f"✅ Retrieved {len(formatted_articles)} articles from Marketaux")
            return formatted_articles

        except Exception as e:
            logger.error(f"Marketaux API failed: {e}")
            return []


if __name__ == "__main__":
    # Test scrapers
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing YouTube Scraper ===")
    yt = YouTubeScraper()
    videos = yt.get_trending_finance_videos()
    print(f"Found {len(videos)} videos")
    for video in videos[:3]:
        print(f"  - {video['title']} ({video['views']:,} views)")

    print("\n=== Testing Twitter Scraper ===")
    twitter = TwitterScraper()
    tweets = twitter.get_trending_stock_tweets()
    print(f"Found {len(tweets)} tweets")
    for tweet in tweets[:3]:
        engagement = tweet['likes'] + tweet['retweets']
        print(f"  - @{tweet['author']}: {tweet['text'][:80]}... ({engagement} engagement)")

    print("\n✅ Scrapers test complete")
