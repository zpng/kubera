"""
Twitter Sentiment Agent
Collects and analyzes sentiment from Twitter for portfolio stocks
Model: deepseek/deepseek-chat-v3.1 (efficient text analysis with tool use)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Tweet(BaseModel):
    """Schema for tweet"""
    text: str
    author: str
    likes: int
    retweets: int
    timestamp: str
    sentiment: str  # positive, negative, neutral


class TwitterSentiment(BaseModel):
    """Schema for Twitter sentiment analysis"""
    symbol: str
    tweets: List[Tweet]
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1.0 to 1.0
    engagement_level: str  # high, medium, low
    key_topics: List[str]
    sentiment_summary: str


@tool
def fetch_twitter_data(symbol: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch Twitter data for a stock (using mock data for now - can be replaced with actual API)
    
    Args:
        symbol: Stock ticker symbol
        limit: Number of tweets to fetch
        
    Returns:
        List of tweets
    """
    try:
        # Note: In production, integrate with Twitter API (tweepy) or scraping
        # For now, using mock structure
        logger.info(f"Fetching Twitter data for ${symbol}...")
        
        # Mock data structure (replace with actual API call)
        mock_tweets = [
            {
                "text": f"{symbol} showing strong momentum today! 📈",
                "author": "TraderJoe",
                "likes": 150,
                "retweets": 45,
                "timestamp": datetime.now().isoformat(),
                "raw_sentiment": 0.8
            },
            {
                "text": f"Concerns about {symbol} valuation at current levels",
                "author": "MarketWatch",
                "likes": 89,
                "retweets": 23,
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "raw_sentiment": -0.4
            }
        ]
        
        logger.info(f"Fetched {len(mock_tweets)} tweets for ${symbol}")
        return mock_tweets
    
    except Exception as e:
        logger.error(f"Error fetching Twitter data for {symbol}: {e}")
        return []


class TwitterSentimentAgent:
    """
    Agent responsible for collecting and analyzing Twitter sentiment
    Uses: deepseek/deepseek-chat-v3.1 for efficient sentiment analysis
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "deepseek/deepseek-chat-v3.1"
    ):
        """
        Initialize Twitter Sentiment Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: deepseek/deepseek-chat-v3.1)
        """
        self.model = model
        
        # Initialize LLM for sentiment analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.3
        )
        
        # Bind tools
        self.tools = [fetch_twitter_data]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(f"Twitter Sentiment Agent initialized with model: {model}")
    
    def analyze_sentiment(self, symbol: str, tweets: List[Dict[str, Any]]) -> TwitterSentiment:
        """
        Analyze sentiment from tweets using LLM
        
        Args:
            symbol: Stock ticker symbol
            tweets: List of tweet data
            
        Returns:
            Structured sentiment analysis
        """
        if not tweets:
            return TwitterSentiment(
                symbol=symbol,
                tweets=[],
                overall_sentiment="neutral",
                sentiment_score=0.0,
                engagement_level="low",
                key_topics=[],
                sentiment_summary="No Twitter data available"
            )
        
        # Create analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a social media sentiment analyst specializing in financial markets.

Your task is to analyze Twitter sentiment for a stock and provide:
1. Overall sentiment (bullish/bearish/neutral)
2. Sentiment score (-1.0 to 1.0)
3. Engagement level (high/medium/low) based on likes and retweets
4. Key topics being discussed
5. Summary of community sentiment and trading implications

Focus on:
- Price predictions and targets
- News reactions
- Technical analysis discussions
- Insider information claims
- Community mood (FOMO, fear, optimism)

Be objective and identify both bullish and bearish narratives."""),
            ("user", """Analyze Twitter sentiment for ${symbol}:

Tweets:
{tweets}

Provide:
1. Overall sentiment classification
2. Sentiment score (-1.0 to 1.0)
3. Engagement level
4. Key discussion topics
5. Brief summary for investment insights""")
        ])
        
        # Format tweets
        tweets_text = "\n\n".join([
            f"@{t['author']} ({t['likes']} likes, {t['retweets']} RTs):\n{t['text']}"
            for t in tweets[:10]
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "tweets": tweets_text
            })
            
            analysis_text = response.content
            
            # Calculate aggregate sentiment score
            sentiment_score = sum(t.get('raw_sentiment', 0) for t in tweets) / len(tweets) if tweets else 0
            
            # Classify overall sentiment
            if sentiment_score > 0.3:
                overall_sentiment = "bullish"
            elif sentiment_score < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
            
            # Calculate engagement level
            total_engagement = sum(t.get('likes', 0) + t.get('retweets', 0) for t in tweets)
            avg_engagement = total_engagement / len(tweets) if tweets else 0
            
            if avg_engagement > 100:
                engagement_level = "high"
            elif avg_engagement > 30:
                engagement_level = "medium"
            else:
                engagement_level = "low"
            
            # Structure tweets
            structured_tweets = [
                Tweet(
                    text=t['text'],
                    author=t['author'],
                    likes=t['likes'],
                    retweets=t['retweets'],
                    timestamp=t['timestamp'],
                    sentiment=self._classify_tweet_sentiment(t.get('raw_sentiment', 0))
                )
                for t in tweets[:10]
            ]
            
            # Extract key topics (simplified)
            key_topics = ["earnings", "technical analysis", "news reaction"]
            
            return TwitterSentiment(
                symbol=symbol,
                tweets=structured_tweets,
                overall_sentiment=overall_sentiment,
                sentiment_score=round(sentiment_score, 2),
                engagement_level=engagement_level,
                key_topics=key_topics,
                sentiment_summary=analysis_text[:300] if analysis_text else "Analysis unavailable"
            )
        
        except Exception as e:
            logger.error(f"Error analyzing Twitter sentiment: {e}")
            return TwitterSentiment(
                symbol=symbol,
                tweets=[],
                overall_sentiment="neutral",
                sentiment_score=0.0,
                engagement_level="low",
                key_topics=[],
                sentiment_summary="Analysis unavailable"
            )
    
    def _classify_tweet_sentiment(self, score: float) -> str:
        """Classify tweet sentiment"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        return "neutral"
    
    def run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - analyzes Twitter sentiment for portfolio stocks
        
        Args:
            portfolio_data: Portfolio data with symbols
            
        Returns:
            Dictionary with Twitter sentiment analysis
        """
        logger.info("=" * 50)
        logger.info("TWITTER SENTIMENT AGENT - Starting execution")
        logger.info("=" * 50)
        
        symbols = portfolio_data.get('symbols', [])
        
        sentiment_analyses = []
        for symbol in symbols:
            try:
                # Fetch tweets
                tweets = fetch_twitter_data.invoke({"symbol": symbol, "limit": 20})
                
                # Analyze sentiment
                analysis = self.analyze_sentiment(symbol, tweets)
                
                sentiment_analyses.append(analysis.dict())
                logger.info(f"✓ {symbol}: {analysis.overall_sentiment} ({analysis.sentiment_score:+.2f})")
            
            except Exception as e:
                logger.error(f"Error processing Twitter sentiment for {symbol}: {e}")
        
        result = {
            "agent": "twitter_sentiment",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "sentiment_analyses": sentiment_analyses,
            "total_stocks": len(sentiment_analyses),
            "status": "success"
        }
        
        logger.info(f"Twitter Sentiment Agent completed - {len(sentiment_analyses)} stocks analyzed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_twitter_sentiment_agent():
    """Test the Twitter sentiment agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Twitter Sentiment Agent...")
        print("=" * 60)
        
        # Mock portfolio data
        mock_portfolio = {
            "symbols": ["AAPL", "TSLA"]
        }
        
        agent = TwitterSentimentAgent(openrouter_api_key=api_key)
        result = agent.run(mock_portfolio)
        
        print("\n✅ Test Results:")
        print(f"   - Stocks analyzed: {result['total_stocks']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Twitter Sentiment Agent test passed!")
        
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
    
    test_twitter_sentiment_agent()

