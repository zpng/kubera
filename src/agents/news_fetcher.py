"""
News Fetcher Agent
Gathers latest news, earnings, events, and predictions for portfolio stocks
Model: google/gemini-2.0-flash-exp (multimodal, great for diverse data sources)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Import data providers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.alpha_vantage import AlphaVantageProvider

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    """Schema for news article"""
    title: str
    summary: str
    source: str
    published_at: str
    sentiment: str  # positive, negative, neutral
    relevance_score: float


class CompanyEvent(BaseModel):
    """Schema for company event"""
    event_type: str  # earnings, contract, regulatory, product
    description: str
    date: str
    impact: str  # high, medium, low


class StockNews(BaseModel):
    """Schema for stock news analysis"""
    symbol: str
    news_articles: List[NewsArticle]
    key_events: List[CompanyEvent]
    overall_sentiment: str
    news_summary: str


@tool
def fetch_news_articles(symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch latest news articles for a stock
    
    Args:
        symbol: Stock ticker symbol
        limit: Number of articles to fetch
        
    Returns:
        List of news articles
    """
    try:
        alpha_vantage = AlphaVantageProvider()
        news_data = alpha_vantage.get_news(symbol, limit=limit)
        
        articles = []
        for article in news_data.get('feed', [])[:limit]:
            articles.append({
                "title": article.get('title', ''),
                "summary": article.get('summary', ''),
                "source": article.get('source', 'Unknown'),
                "published_at": article.get('time_published', ''),
                "url": article.get('url', ''),
                "sentiment_score": article.get('overall_sentiment_score', 0)
            })
        
        logger.info(f"Fetched {len(articles)} news articles for {symbol}")
        return articles
    
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []


@tool
def analyze_earnings_calendar(symbol: str) -> Dict[str, Any]:
    """
    Get earnings calendar and financial events
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with earnings information
    """
    try:
        import yfinance as yf
        
        stock = yf.Ticker(symbol)
        calendar = stock.calendar
        
        if calendar is None or calendar.empty:
            return {"symbol": symbol, "earnings_date": None}
        
        # Extract earnings date
        earnings_date = None
        if 'Earnings Date' in calendar.index:
            earnings_date = str(calendar.loc['Earnings Date'].iloc[0]) if hasattr(calendar.loc['Earnings Date'], 'iloc') else str(calendar.loc['Earnings Date'])
        
        return {
            "symbol": symbol,
            "earnings_date": earnings_date,
            "has_upcoming_earnings": earnings_date is not None
        }
    
    except Exception as e:
        logger.error(f"Error fetching earnings for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


class NewsFetcherAgent:
    """
    Agent responsible for fetching and analyzing news and events
    Uses: google/gemini-2.0-flash-exp for multimodal data integration
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "google/gemini-2.0-flash-exp"
    ):
        """
        Initialize News Fetcher Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: google/gemini-2.0-flash-exp)
        """
        self.model = model
        
        # Initialize LLM for news analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.3  # Moderate temperature for creative analysis
        )
        
        # Bind tools
        self.tools = [fetch_news_articles, analyze_earnings_calendar]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(f"News Fetcher Agent initialized with model: {model}")
    
    def fetch_stock_news(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive news for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with news and events
        """
        logger.info(f"Fetching news and events for {symbol}...")
        
        # Fetch news articles
        articles = fetch_news_articles.invoke({"symbol": symbol, "limit": 10})
        
        # Fetch earnings calendar
        earnings_info = analyze_earnings_calendar.invoke({"symbol": symbol})
        
        return {
            "symbol": symbol,
            "articles": articles,
            "earnings_info": earnings_info,
            "fetched_at": datetime.now().isoformat()
        }
    
    def analyze_news_with_llm(self, symbol: str, news_data: Dict[str, Any]) -> StockNews:
        """
        Analyze news data using LLM to extract insights
        
        Args:
            symbol: Stock ticker symbol
            news_data: Raw news data
            
        Returns:
            Structured news analysis
        """
        articles = news_data.get('articles', [])
        earnings_info = news_data.get('earnings_info', {})
        
        # Create comprehensive prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial news analyst specializing in extracting actionable insights from news and events.

Your task is to analyze news articles and company events for a stock and provide:
1. Overall sentiment (bullish/bearish/neutral) based on news
2. Key events (earnings, contracts, regulatory changes, product launches)
3. A concise summary highlighting the most important information for investors
4. Impact assessment (high/medium/low) for each event

Focus on:
- Quarterly earnings and financial results
- Government contracts and major deals
- Product launches and innovations
- Regulatory changes
- Management changes
- Market trends affecting the company

Return your analysis in a clear, structured format."""),
            ("user", """Analyze the following news data for {symbol}:

News Articles:
{articles}

Earnings Information:
{earnings_info}

Provide a comprehensive analysis with:
1. Overall sentiment
2. Key events with impact assessment
3. A concise summary for investment decision-making""")
        ])
        
        # Format articles for analysis
        articles_text = "\n\n".join([
            f"- {art['title']}\n  Source: {art['source']}\n  Published: {art['published_at']}\n  Summary: {art['summary'][:200]}..."
            for art in articles[:5]
        ]) if articles else "No recent news articles available"
        
        earnings_text = f"Earnings Date: {earnings_info.get('earnings_date', 'Not scheduled')}" if earnings_info else "No earnings info"
        
        # Invoke LLM
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "articles": articles_text,
                "earnings_info": earnings_text
            })
            
            # Parse response
            analysis_text = response.content
            
            # Extract sentiment
            sentiment = "neutral"
            if "bullish" in analysis_text.lower() or "positive" in analysis_text.lower():
                sentiment = "bullish"
            elif "bearish" in analysis_text.lower() or "negative" in analysis_text.lower():
                sentiment = "bearish"
            
            # Structure news articles
            news_articles = [
                NewsArticle(
                    title=art['title'],
                    summary=art['summary'][:300],
                    source=art['source'],
                    published_at=art['published_at'],
                    sentiment=self._classify_sentiment(art.get('sentiment_score', 0)),
                    relevance_score=abs(art.get('sentiment_score', 0))
                )
                for art in articles[:5]
            ]
            
            # Extract key events
            key_events = []
            if earnings_info.get('has_upcoming_earnings'):
                key_events.append(CompanyEvent(
                    event_type="earnings",
                    description=f"Upcoming earnings report on {earnings_info.get('earnings_date', 'TBD')}",
                    date=earnings_info.get('earnings_date', ''),
                    impact="high"
                ))
            
            return StockNews(
                symbol=symbol,
                news_articles=news_articles,
                key_events=key_events,
                overall_sentiment=sentiment,
                news_summary=analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text
            )
        
        except Exception as e:
            logger.error(f"Error analyzing news with LLM: {e}")
            # Return basic analysis
            return StockNews(
                symbol=symbol,
                news_articles=[],
                key_events=[],
                overall_sentiment="neutral",
                news_summary="Analysis unavailable"
            )
    
    def _classify_sentiment(self, score: float) -> str:
        """Classify sentiment based on score"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        return "neutral"
    
    def run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - fetches and analyzes news for all portfolio stocks
        
        Args:
            portfolio_data: Portfolio data with symbols
            
        Returns:
            Dictionary with news analysis for each stock
        """
        logger.info("=" * 50)
        logger.info("NEWS FETCHER AGENT - Starting execution")
        logger.info("=" * 50)
        
        symbols = portfolio_data.get('symbols', [])
        
        news_analyses = []
        for symbol in symbols:
            try:
                # Fetch news data
                news_data = self.fetch_stock_news(symbol)
                
                # Analyze with LLM
                analysis = self.analyze_news_with_llm(symbol, news_data)
                
                news_analyses.append(analysis.dict())
                logger.info(f"✓ {symbol}: {analysis.overall_sentiment} sentiment, {len(analysis.news_articles)} articles")
            
            except Exception as e:
                logger.error(f"Error processing news for {symbol}: {e}")
        
        result = {
            "agent": "news_fetcher",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "news_analyses": news_analyses,
            "total_stocks": len(news_analyses),
            "status": "success"
        }
        
        logger.info(f"News Fetcher Agent completed - {len(news_analyses)} stocks analyzed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_news_fetcher_agent():
    """Test the news fetcher agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing News Fetcher Agent...")
        print("=" * 60)
        
        # Mock portfolio data
        mock_portfolio = {
            "symbols": ["AAPL", "TSLA"]
        }
        
        agent = NewsFetcherAgent(openrouter_api_key=api_key)
        result = agent.run(mock_portfolio)
        
        print("\n✅ Test Results:")
        print(f"   - Stocks analyzed: {result['total_stocks']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ News Fetcher Agent test passed!")
        
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
    
    test_news_fetcher_agent()

