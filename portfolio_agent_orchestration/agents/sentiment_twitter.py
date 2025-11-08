"""
Twitter Sentiment Agent
Collects and analyzes sentiment from Twitter/X
Model: nousresearch/hermes-3-llama-3.1-405b (strong at multi-turn conversation and sentiment analysis)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base_agent import BaseAgent
from ..config import AGENT_MODELS, DATA_SOURCES

logger = logging.getLogger(__name__)


class TwitterSentimentAgent(BaseAgent):
    """
    Agent responsible for analyzing Twitter/X sentiment for stocks
    """
    
    def __init__(self):
        super().__init__(
            name="TwitterSentiment",
            model=AGENT_MODELS["sentiment_twitter"],
            role="Twitter/X social sentiment analyzer",
            temperature=0.6
        )
        self.enabled = DATA_SOURCES["twitter"]["enabled"]
        self.api_key = DATA_SOURCES["twitter"]["api_key"]
    
    def fetch_twitter_sentiment(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Fetch tweets about a stock symbol
        
        Note: This is a placeholder. Real implementation requires Twitter API v2
        or alternative scraping methods.
        """
        if not self.enabled:
            logger.info(f"Twitter integration disabled for {symbol}")
            return []
        
        # TODO: Implement actual Twitter API integration
        # For now, return mock data structure
        logger.warning(f"Twitter API not implemented for {symbol}. Using placeholder.")
        
        return []
    
    def analyze_tweet_sentiment(self, tweets: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment from tweet data
        
        This uses simple keyword-based sentiment for now.
        Can be enhanced with the LLM for deeper analysis.
        """
        if not tweets:
            return {
                "sentiment_score": 0,
                "sentiment_label": "Neutral",
                "tweet_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0
            }
        
        bullish_keywords = [
            "bullish", "buy", "moon", "rocket", "gains", "breakout",
            "strong", "growth", "bull", "long", "hold"
        ]
        
        bearish_keywords = [
            "bearish", "sell", "crash", "dump", "losses", "breakdown",
            "weak", "decline", "bear", "short", "puts"
        ]
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tweet in tweets:
            text = tweet.get("text", "").lower()
            
            bullish_score = sum(1 for keyword in bullish_keywords if keyword in text)
            bearish_score = sum(1 for keyword in bearish_keywords if keyword in text)
            
            if bullish_score > bearish_score:
                bullish_count += 1
            elif bearish_score > bullish_score:
                bearish_count += 1
            else:
                neutral_count += 1
        
        total = len(tweets)
        
        # Calculate sentiment score (-1 to +1)
        sentiment_score = (bullish_count - bearish_count) / total if total > 0 else 0
        
        if sentiment_score > 0.2:
            sentiment_label = "Bullish"
        elif sentiment_score < -0.2:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"
        
        return {
            "sentiment_score": round(sentiment_score, 3),
            "sentiment_label": sentiment_label,
            "tweet_count": total,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bullish_pct": round((bullish_count / total) * 100, 1) if total > 0 else 0,
            "bearish_pct": round((bearish_count / total) * 100, 1) if total > 0 else 0,
        }
    
    def extract_trending_topics(self, tweets: List[Dict]) -> List[str]:
        """Extract trending topics from tweets"""
        if not tweets:
            return []
        
        # Extract hashtags and common terms
        hashtags = []
        for tweet in tweets:
            text = tweet.get("text", "")
            # Simple hashtag extraction
            words = text.split()
            for word in words:
                if word.startswith("#") and len(word) > 2:
                    hashtags.append(word)
        
        # Count frequency
        from collections import Counter
        if hashtags:
            common_tags = Counter(hashtags).most_common(5)
            return [tag for tag, count in common_tags]
        
        return []
    
    def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process Twitter sentiment for a single symbol"""
        logger.info(f"Analyzing Twitter sentiment for {symbol}")
        
        if not self.enabled:
            return {
                "symbol": symbol,
                "status": "disabled",
                "message": "Twitter integration not enabled. Set TWITTER_API_KEY and enable in config.",
                "sentiment_analysis": {
                    "sentiment_score": 0,
                    "sentiment_label": "Unknown",
                    "tweet_count": 0
                }
            }
        
        tweets = self.fetch_twitter_sentiment(symbol)
        sentiment_analysis = self.analyze_tweet_sentiment(tweets)
        trending_topics = self.extract_trending_topics(tweets)
        
        return {
            "symbol": symbol,
            "status": "success" if tweets else "no_data",
            "sentiment_analysis": sentiment_analysis,
            "trending_topics": trending_topics,
            "data_source": "Twitter/X",
            "timestamp": datetime.now().isoformat()
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with stock_symbols
            
        Returns:
            Updated state with Twitter sentiment data
        """
        logger.info(f"[{self.name}] Starting Twitter sentiment analysis...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        twitter_sentiment = {}
        
        for symbol in symbols:
            data = self.process_symbol(symbol)
            twitter_sentiment[symbol] = data
        
        # Update state
        state["twitter_sentiment"] = twitter_sentiment
        state["twitter_sentiment_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Analyzed Twitter sentiment for {len(twitter_sentiment)} symbols")
        return state


# Test function
def test_twitter_sentiment_agent():
    """Test the TwitterSentimentAgent"""
    print("Testing Twitter Sentiment Agent...")
    
    agent = TwitterSentimentAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["AAPL", "TSLA"]
    }
    
    result_state = agent.process(state)
    
    print(f"\nTwitter Sentiment Results:")
    for symbol, data in result_state.get("twitter_sentiment", {}).items():
        print(f"\n{symbol}:")
        print(f"  Status: {data['status']}")
        sentiment = data.get('sentiment_analysis', {})
        print(f"  Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
        print(f"  Score: {sentiment.get('sentiment_score', 'N/A')}")
        print(f"  Tweets Analyzed: {sentiment.get('tweet_count', 0)}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_twitter_sentiment_agent()

