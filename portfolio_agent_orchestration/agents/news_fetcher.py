"""
News Fetcher Agent
Gathers latest news for portfolio stocks using Alpha Vantage and other sources
Model: google/gemini-2.0-flash-exp (multimodal, good at integrating diverse data)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import requests
from langchain_core.tools import tool

from .base_agent import BaseAgent
from ..config import AGENT_MODELS, DATA_SOURCES

logger = logging.getLogger(__name__)


class NewsFetcherAgent(BaseAgent):
    """
    Agent responsible for fetching and analyzing news for stocks
    """
    
    def __init__(self):
        # Define tools for news fetching
        tools = [self._create_news_tool()]
        
        super().__init__(
            name="NewsFetcher",
            model=AGENT_MODELS["news_fetcher"],
            role="News aggregator and analyzer for stock-related information",
            temperature=0.5,
            tools=tools
        )
        self.alpha_vantage_key = DATA_SOURCES["alpha_vantage"]["api_key"]
    
    @staticmethod
    @tool
    def fetch_news_alpha_vantage(symbol: str, limit: int = 10) -> List[Dict]:
        """Fetch news articles for a stock symbol from Alpha Vantage"""
        pass  # Placeholder for tool registration
    
    def _create_news_tool(self):
        """Create news fetching tool"""
        @tool
        def get_stock_news(symbol: str) -> str:
            """Get latest news for a stock symbol"""
            return str(self._fetch_news_internal(symbol))
        
        return get_stock_news
    
    def _fetch_news_internal(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Internal method to fetch news from Alpha Vantage"""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not configured")
            return []
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": symbol,
                "apikey": self.alpha_vantage_key,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "feed" not in data:
                logger.warning(f"No news found for {symbol}")
                return []
            
            articles = []
            for item in data["feed"][:limit]:
                article = {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                    "published": item.get("time_published", ""),
                    "sentiment": self._extract_sentiment(item, symbol),
                }
                articles.append(article)
            
            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching news for {symbol}: {e}")
            return []
    
    def _extract_sentiment(self, article: Dict, symbol: str) -> Dict[str, Any]:
        """Extract sentiment scores from article"""
        try:
            ticker_sentiments = article.get("ticker_sentiment", [])
            
            for ticker_data in ticker_sentiments:
                if ticker_data.get("ticker") == symbol:
                    return {
                        "score": float(ticker_data.get("ticker_sentiment_score", 0)),
                        "label": ticker_data.get("ticker_sentiment_label", "Neutral")
                    }
            
            # Overall sentiment if ticker-specific not found
            return {
                "score": float(article.get("overall_sentiment_score", 0)),
                "label": article.get("overall_sentiment_label", "Neutral")
            }
            
        except Exception as e:
            logger.error(f"Error extracting sentiment: {e}")
            return {"score": 0, "label": "Neutral"}
    
    def analyze_news_sentiment(self, articles: List[Dict]) -> Dict[str, Any]:
        """Analyze overall sentiment from news articles"""
        if not articles:
            return {
                "average_score": 0,
                "dominant_sentiment": "Neutral",
                "article_count": 0
            }
        
        scores = [art["sentiment"]["score"] for art in articles if "sentiment" in art]
        
        if not scores:
            return {
                "average_score": 0,
                "dominant_sentiment": "Neutral",
                "article_count": len(articles)
            }
        
        avg_score = sum(scores) / len(scores)
        
        # Determine dominant sentiment
        if avg_score > 0.15:
            sentiment = "Bullish"
        elif avg_score < -0.15:
            sentiment = "Bearish"
        else:
            sentiment = "Neutral"
        
        return {
            "average_score": round(avg_score, 3),
            "dominant_sentiment": sentiment,
            "article_count": len(articles),
            "sentiment_range": {
                "min": round(min(scores), 3),
                "max": round(max(scores), 3)
            }
        }
    
    def extract_key_events(self, articles: List[Dict]) -> List[str]:
        """Extract key events from news headlines and summaries"""
        key_events = []
        
        keywords = [
            "earnings", "revenue", "profit", "loss",
            "acquisition", "merger", "partnership",
            "product launch", "contract", "deal",
            "lawsuit", "investigation", "regulatory",
            "dividend", "stock split", "buyback",
            "guidance", "forecast", "outlook"
        ]
        
        for article in articles[:5]:  # Top 5 articles
            title = article.get("title", "").lower()
            summary = article.get("summary", "").lower()
            
            for keyword in keywords:
                if keyword in title or keyword in summary:
                    event = f"{article.get('published', 'Recent')}: {article.get('title', '')[:100]}"
                    if event not in key_events:
                        key_events.append(event)
                    break
        
        return key_events[:5]  # Top 5 key events
    
    def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process news for a single symbol"""
        logger.info(f"Fetching news for {symbol}")
        
        articles = self._fetch_news_internal(symbol)
        
        if not articles:
            return {
                "symbol": symbol,
                "status": "no_news",
                "articles": [],
                "sentiment_analysis": self.analyze_news_sentiment([]),
                "key_events": []
            }
        
        sentiment_analysis = self.analyze_news_sentiment(articles)
        key_events = self.extract_key_events(articles)
        
        return {
            "symbol": symbol,
            "status": "success",
            "articles": articles,
            "sentiment_analysis": sentiment_analysis,
            "key_events": key_events,
            "timestamp": datetime.now().isoformat()
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with stock_symbols
            
        Returns:
            Updated state with news data
        """
        logger.info(f"[{self.name}] Starting news collection...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        news_data = {}
        
        for symbol in symbols:
            data = self.process_symbol(symbol)
            news_data[symbol] = data
        
        # Update state
        state["news_data"] = news_data
        state["news_data_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Collected news for {len(news_data)} symbols")
        return state


# Test function
def test_news_fetcher_agent():
    """Test the NewsFetcherAgent"""
    print("Testing News Fetcher Agent...")
    
    agent = NewsFetcherAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["AAPL", "TSLA"]
    }
    
    result_state = agent.process(state)
    
    print(f"\nNews Results:")
    for symbol, data in result_state.get("news_data", {}).items():
        print(f"\n{symbol}:")
        print(f"  Status: {data['status']}")
        if data['status'] == 'success':
            sentiment = data['sentiment_analysis']
            print(f"  Articles: {sentiment['article_count']}")
            print(f"  Sentiment: {sentiment['dominant_sentiment']} (score: {sentiment['average_score']})")
            print(f"  Key Events: {len(data['key_events'])}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_news_fetcher_agent()

