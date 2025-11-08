"""
Reddit Sentiment Agent
Collects and analyzes sentiment from Reddit (wallstreetbets, stocks, etc.)
Model: nousresearch/hermes-3-llama-3.1-405b (strong at understanding community discussions)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent
from ..config import AGENT_MODELS, DATA_SOURCES

logger = logging.getLogger(__name__)


class RedditSentimentAgent(BaseAgent):
    """
    Agent responsible for analyzing Reddit sentiment for stocks
    """
    
    def __init__(self):
        super().__init__(
            name="RedditSentiment",
            model=AGENT_MODELS["sentiment_reddit"],
            role="Reddit social sentiment analyzer",
            temperature=0.6
        )
        self.enabled = DATA_SOURCES["reddit"]["enabled"]
        self.client_id = DATA_SOURCES["reddit"]["client_id"]
        self.client_secret = DATA_SOURCES["reddit"]["client_secret"]
        
        # Relevant subreddits for stock discussion
        self.subreddits = [
            "wallstreetbets",
            "stocks",
            "investing",
            "StockMarket",
            "options"
        ]
    
    def fetch_reddit_posts(self, symbol: str, limit: int = 50) -> List[Dict]:
        """
        Fetch Reddit posts mentioning a stock symbol
        
        Note: This is a placeholder. Real implementation requires Reddit API (PRAW)
        """
        if not self.enabled:
            logger.info(f"Reddit integration disabled for {symbol}")
            return []
        
        # TODO: Implement actual Reddit API integration using PRAW
        # Example implementation would look like:
        # import praw
        # reddit = praw.Reddit(client_id=self.client_id, client_secret=self.client_secret, user_agent='Kubera')
        # posts = reddit.subreddit('+'.join(self.subreddits)).search(symbol, limit=limit, time_filter='week')
        
        logger.warning(f"Reddit API not implemented for {symbol}. Using placeholder.")
        return []
    
    def analyze_reddit_sentiment(self, posts: List[Dict]) -> Dict[str, Any]:
        """
        Analyze sentiment from Reddit posts
        
        Uses keyword-based sentiment analysis
        Can be enhanced with LLM for deeper context understanding
        """
        if not posts:
            return {
                "sentiment_score": 0,
                "sentiment_label": "Neutral",
                "post_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "avg_upvote_ratio": 0,
                "total_comments": 0
            }
        
        bullish_keywords = [
            "moon", "rocket", "diamond hands", "hold", "buying",
            "calls", "yolo", "to the moon", "bullish", "dd",
            "long", "tendie", "ape", "strong"
        ]
        
        bearish_keywords = [
            "crash", "sell", "puts", "short", "bearish",
            "dump", "overvalued", "bubble", "weak", "dropping"
        ]
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_upvotes = 0
        total_comments = 0
        
        for post in posts:
            title = post.get("title", "").lower()
            body = post.get("body", "").lower()
            text = f"{title} {body}"
            
            bullish_score = sum(1 for keyword in bullish_keywords if keyword in text)
            bearish_score = sum(1 for keyword in bearish_keywords if keyword in text)
            
            if bullish_score > bearish_score:
                bullish_count += 1
            elif bearish_score > bullish_score:
                bearish_count += 1
            else:
                neutral_count += 1
            
            total_upvotes += post.get("upvote_ratio", 0)
            total_comments += post.get("num_comments", 0)
        
        total = len(posts)
        
        # Calculate sentiment score
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
            "post_count": total,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "bullish_pct": round((bullish_count / total) * 100, 1) if total > 0 else 0,
            "bearish_pct": round((bearish_count / total) * 100, 1) if total > 0 else 0,
            "avg_upvote_ratio": round(total_upvotes / total, 2) if total > 0 else 0,
            "total_comments": total_comments,
            "engagement_score": round((total_upvotes + (total_comments / 10)) / total, 2) if total > 0 else 0
        }
    
    def extract_top_discussions(self, posts: List[Dict], limit: int = 3) -> List[Dict]:
        """Extract top Reddit discussions by engagement"""
        if not posts:
            return []
        
        # Sort by engagement (upvotes + comments)
        sorted_posts = sorted(
            posts,
            key=lambda p: p.get("score", 0) + p.get("num_comments", 0),
            reverse=True
        )
        
        top_discussions = []
        for post in sorted_posts[:limit]:
            discussion = {
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "comments": post.get("num_comments", 0),
                "url": post.get("url", "")
            }
            top_discussions.append(discussion)
        
        return top_discussions
    
    def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process Reddit sentiment for a single symbol"""
        logger.info(f"Analyzing Reddit sentiment for {symbol}")
        
        if not self.enabled:
            return {
                "symbol": symbol,
                "status": "disabled",
                "message": "Reddit integration not enabled. Set REDDIT credentials and enable in config.",
                "sentiment_analysis": {
                    "sentiment_score": 0,
                    "sentiment_label": "Unknown",
                    "post_count": 0
                }
            }
        
        posts = self.fetch_reddit_posts(symbol)
        sentiment_analysis = self.analyze_reddit_sentiment(posts)
        top_discussions = self.extract_top_discussions(posts)
        
        return {
            "symbol": symbol,
            "status": "success" if posts else "no_data",
            "sentiment_analysis": sentiment_analysis,
            "top_discussions": top_discussions,
            "subreddits_monitored": self.subreddits,
            "data_source": "Reddit",
            "timestamp": datetime.now().isoformat()
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with stock_symbols
            
        Returns:
            Updated state with Reddit sentiment data
        """
        logger.info(f"[{self.name}] Starting Reddit sentiment analysis...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        reddit_sentiment = {}
        
        for symbol in symbols:
            data = self.process_symbol(symbol)
            reddit_sentiment[symbol] = data
        
        # Update state
        state["reddit_sentiment"] = reddit_sentiment
        state["reddit_sentiment_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Analyzed Reddit sentiment for {len(reddit_sentiment)} symbols")
        return state


# Test function
def test_reddit_sentiment_agent():
    """Test the RedditSentimentAgent"""
    print("Testing Reddit Sentiment Agent...")
    
    agent = RedditSentimentAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["GME", "TSLA"]
    }
    
    result_state = agent.process(state)
    
    print(f"\nReddit Sentiment Results:")
    for symbol, data in result_state.get("reddit_sentiment", {}).items():
        print(f"\n{symbol}:")
        print(f"  Status: {data['status']}")
        sentiment = data.get('sentiment_analysis', {})
        print(f"  Sentiment: {sentiment.get('sentiment_label', 'N/A')}")
        print(f"  Score: {sentiment.get('sentiment_score', 'N/A')}")
        print(f"  Posts Analyzed: {sentiment.get('post_count', 0)}")
        print(f"  Engagement Score: {sentiment.get('engagement_score', 'N/A')}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_reddit_sentiment_agent()

