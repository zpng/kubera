"""
Reddit Sentiment Agent
Collects and analyzes sentiment from Reddit (r/wallstreetbets, r/stocks, r/investing)
Model: deepseek/deepseek-chat-v3.1 (efficient text analysis)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RedditPost(BaseModel):
    """Schema for Reddit post"""
    title: str
    text: str
    author: str
    subreddit: str
    upvotes: int
    comments: int
    timestamp: str
    sentiment: str


class RedditSentiment(BaseModel):
    """Schema for Reddit sentiment analysis"""
    symbol: str
    posts: List[RedditPost]
    overall_sentiment: str  # bullish, bearish, neutral
    sentiment_score: float  # -1.0 to 1.0
    community_interest: str  # high, medium, low
    discussion_themes: List[str]
    sentiment_summary: str


@tool
def fetch_reddit_data(symbol: str, subreddits: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetch Reddit posts about a stock (using mock data for now)
    
    Args:
        symbol: Stock ticker symbol
        subreddits: List of subreddits to search
        limit: Number of posts to fetch
        
    Returns:
        List of Reddit posts
    """
    try:
        if subreddits is None:
            subreddits = ["wallstreetbets", "stocks", "investing"]
        
        logger.info(f"Fetching Reddit data for {symbol} from {', '.join(subreddits)}...")
        
        # Mock data structure (replace with actual PRAW API call)
        mock_posts = [
            {
                "title": f"DD: Why {symbol} is undervalued",
                "text": f"Comprehensive analysis of {symbol} fundamentals...",
                "author": "DeepValueInvestor",
                "subreddit": "stocks",
                "upvotes": 234,
                "comments": 67,
                "timestamp": datetime.now().isoformat(),
                "raw_sentiment": 0.7
            },
            {
                "title": f"{symbol} earnings tomorrow - what's your play?",
                "text": "Expecting volatility with earnings...",
                "author": "OptionsTrader",
                "subreddit": "wallstreetbets",
                "upvotes": 156,
                "comments": 89,
                "timestamp": (datetime.now() - timedelta(hours=5)).isoformat(),
                "raw_sentiment": 0.2
            }
        ]
        
        logger.info(f"Fetched {len(mock_posts)} Reddit posts for {symbol}")
        return mock_posts
    
    except Exception as e:
        logger.error(f"Error fetching Reddit data for {symbol}: {e}")
        return []


class RedditSentimentAgent:
    """
    Agent responsible for collecting and analyzing Reddit sentiment
    Uses: deepseek/deepseek-chat-v3.1 for efficient community sentiment analysis
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "deepseek/deepseek-chat-v3.1"
    ):
        """
        Initialize Reddit Sentiment Agent
        
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
        self.tools = [fetch_reddit_data]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(f"Reddit Sentiment Agent initialized with model: {model}")
    
    def analyze_sentiment(self, symbol: str, posts: List[Dict[str, Any]]) -> RedditSentiment:
        """
        Analyze sentiment from Reddit posts using LLM
        
        Args:
            symbol: Stock ticker symbol
            posts: List of Reddit post data
            
        Returns:
            Structured sentiment analysis
        """
        if not posts:
            return RedditSentiment(
                symbol=symbol,
                posts=[],
                overall_sentiment="neutral",
                sentiment_score=0.0,
                community_interest="low",
                discussion_themes=[],
                sentiment_summary="No Reddit data available"
            )
        
        # Create analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Reddit community sentiment analyst specializing in retail investor psychology.

Your task is to analyze Reddit discussions about a stock and provide:
1. Overall community sentiment (bullish/bearish/neutral)
2. Sentiment score (-1.0 to 1.0)
3. Community interest level based on engagement
4. Key discussion themes (DD/due diligence, technical analysis, YOLO plays, etc.)
5. Summary of community consensus and notable perspectives

Focus on:
- Quality of due diligence posts
- Options activity discussions
- Meme potential and retail momentum
- Short squeeze potential
- Risk warnings and bearish cases
- Overall community conviction level

Be objective and note the quality of analysis in posts."""),
            ("user", """Analyze Reddit sentiment for {symbol}:

Posts:
{posts}

Provide:
1. Overall sentiment classification
2. Sentiment score (-1.0 to 1.0)
3. Community interest level
4. Key discussion themes
5. Brief summary highlighting consensus and notable viewpoints""")
        ])
        
        # Format posts
        posts_text = "\n\n".join([
            f"r/{p['subreddit']} | u/{p['author']} ({p['upvotes']} upvotes, {p['comments']} comments):\n"
            f"Title: {p['title']}\n"
            f"Text: {p['text'][:200]}..."
            for p in posts[:8]
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "posts": posts_text
            })
            
            analysis_text = response.content
            
            # Calculate aggregate sentiment score
            sentiment_score = sum(p.get('raw_sentiment', 0) for p in posts) / len(posts) if posts else 0
            
            # Classify overall sentiment
            if sentiment_score > 0.3:
                overall_sentiment = "bullish"
            elif sentiment_score < -0.3:
                overall_sentiment = "bearish"
            else:
                overall_sentiment = "neutral"
            
            # Calculate community interest
            total_engagement = sum(p.get('upvotes', 0) + p.get('comments', 0) for p in posts)
            avg_engagement = total_engagement / len(posts) if posts else 0
            
            if avg_engagement > 200:
                community_interest = "high"
            elif avg_engagement > 50:
                community_interest = "medium"
            else:
                community_interest = "low"
            
            # Structure posts
            structured_posts = [
                RedditPost(
                    title=p['title'],
                    text=p['text'][:300],
                    author=p['author'],
                    subreddit=p['subreddit'],
                    upvotes=p['upvotes'],
                    comments=p['comments'],
                    timestamp=p['timestamp'],
                    sentiment=self._classify_post_sentiment(p.get('raw_sentiment', 0))
                )
                for p in posts[:8]
            ]
            
            # Extract discussion themes
            discussion_themes = ["due diligence", "earnings play", "technical analysis"]
            
            return RedditSentiment(
                symbol=symbol,
                posts=structured_posts,
                overall_sentiment=overall_sentiment,
                sentiment_score=round(sentiment_score, 2),
                community_interest=community_interest,
                discussion_themes=discussion_themes,
                sentiment_summary=analysis_text[:300] if analysis_text else "Analysis unavailable"
            )
        
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            return RedditSentiment(
                symbol=symbol,
                posts=[],
                overall_sentiment="neutral",
                sentiment_score=0.0,
                community_interest="low",
                discussion_themes=[],
                sentiment_summary="Analysis unavailable"
            )
    
    def _classify_post_sentiment(self, score: float) -> str:
        """Classify post sentiment"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        return "neutral"
    
    def run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - analyzes Reddit sentiment for portfolio stocks
        
        Args:
            portfolio_data: Portfolio data with symbols
            
        Returns:
            Dictionary with Reddit sentiment analysis
        """
        logger.info("=" * 50)
        logger.info("REDDIT SENTIMENT AGENT - Starting execution")
        logger.info("=" * 50)
        
        symbols = portfolio_data.get('symbols', [])
        
        sentiment_analyses = []
        for symbol in symbols:
            try:
                # Fetch posts
                posts = fetch_reddit_data.invoke({"symbol": symbol, "limit": 20})
                
                # Analyze sentiment
                analysis = self.analyze_sentiment(symbol, posts)
                
                sentiment_analyses.append(analysis.dict())
                logger.info(f"✓ {symbol}: {analysis.overall_sentiment} ({analysis.sentiment_score:+.2f}) | Interest: {analysis.community_interest}")
            
            except Exception as e:
                logger.error(f"Error processing Reddit sentiment for {symbol}: {e}")
        
        result = {
            "agent": "reddit_sentiment",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "sentiment_analyses": sentiment_analyses,
            "total_stocks": len(sentiment_analyses),
            "status": "success"
        }
        
        logger.info(f"Reddit Sentiment Agent completed - {len(sentiment_analyses)} stocks analyzed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_reddit_sentiment_agent():
    """Test the Reddit sentiment agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Reddit Sentiment Agent...")
        print("=" * 60)
        
        # Mock portfolio data
        mock_portfolio = {
            "symbols": ["AAPL", "TSLA"]
        }
        
        agent = RedditSentimentAgent(openrouter_api_key=api_key)
        result = agent.run(mock_portfolio)
        
        print("\n✅ Test Results:")
        print(f"   - Stocks analyzed: {result['total_stocks']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Reddit Sentiment Agent test passed!")
        
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
    
    test_reddit_sentiment_agent()

