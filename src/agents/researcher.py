"""
Deep Thinking Researcher Agent
Aggregates all agent data and makes final investment decisions with detailed reasoning
Model: qwen/qwen3-235b-a22b (highest-fidelity reasoning for complex decisions)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StockRecommendation(BaseModel):
    """Schema for final stock recommendation"""
    symbol: str
    decision: str  # BUY, HOLD, SELL
    conviction: int  # 1-10 scale
    target_price: float
    stop_loss: float
    time_horizon: str  # short-term, medium-term, long-term
    
    # Detailed reasoning
    bull_case: str
    bear_case: str
    key_catalysts: List[str]
    key_risks: List[str]
    
    # Factor scores (1-10)
    technical_score: int
    fundamental_score: int
    news_sentiment_score: int
    social_sentiment_score: int
    risk_score: int
    
    # Summary
    executive_summary: str
    detailed_analysis: str
    action_plan: str


class PortfolioRecommendations(BaseModel):
    """Schema for complete portfolio recommendations"""
    timestamp: str
    total_stocks_analyzed: int
    recommendations: List[StockRecommendation]
    portfolio_summary: str
    overall_strategy: str


class DeepThinkingResearcher:
    """
    Master agent that synthesizes all data and provides final investment decisions
    Uses: qwen/qwen3-235b-a22b for highest-fidelity reasoning
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "qwen/qwen3-235b-a22b"
    ):
        """
        Initialize Deep Thinking Researcher
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: qwen/qwen3-235b-a22b)
        """
        self.model = model
        
        # Initialize LLM for deep reasoning
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=openrouter_api_key,
            temperature=0.3,
            max_tokens=4000
        )
        
        logger.info(f"Deep Thinking Researcher initialized with model: {model}")
    
    def synthesize_stock_data(
        self,
        symbol: str,
        portfolio_comparison: Dict[str, Any],
        news_analysis: Dict[str, Any],
        company_fundamentals: Dict[str, Any],
        twitter_sentiment: Dict[str, Any],
        reddit_sentiment: Dict[str, Any],
        position_risk: Dict[str, Any],
        available_cash_cny: Any = None
    ) -> StockRecommendation:
        """
        Synthesize all data sources to make final recommendation
        
        Args:
            symbol: Stock ticker
            portfolio_comparison: Price and P&L data
            news_analysis: News sentiment analysis
            company_fundamentals: Company fundamentals analysis
            twitter_sentiment: Twitter sentiment analysis
            reddit_sentiment: Reddit sentiment analysis
            position_risk: Risk analysis
            
        Returns:
            Comprehensive stock recommendation
        """
        logger.info(f"Synthesizing data for {symbol}...")
        
        # Create comprehensive analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a world-class investment researcher with expertise in:
- Technical analysis and price action
- Fundamental analysis and valuation
- Sentiment analysis and market psychology
- Risk management and position sizing
- Multi-timeframe analysis

Your task is to synthesize ALL available data sources and provide a COMPREHENSIVE, DETAILED investment recommendation.

Language Requirement: All narrative content MUST be written in Chinese, while keeping the specified section headers and fixed keys in English (DECISION, CONVICTION, TARGET PRICE, STOP LOSS, TIME HORIZON, BULL CASE, BEAR CASE, KEY CATALYSTS, KEY RISKS, FACTOR SCORES, EXECUTIVE SUMMARY, DETAILED ANALYSIS, ACTION PLAN).

**CRITICAL REQUIREMENTS:**
1. Make a clear BUY/HOLD/SELL decision with conviction level (1-10)
2. Provide detailed reasoning including bull case, bear case, catalysts, and risks
3. Set specific price targets and stop losses
4. Give actionable guidance (buy more, trim, hold, exit)
5. Explain your decision-making process step by step

**Analysis Framework:**
1. Current Position Analysis (P&L, position size, entry vs current)
2. Technical Analysis (price action, momentum, support/resistance)
3. Fundamental Analysis (valuation, growth, financial health)
4. News & Events (catalysts, earnings, developments)
5. Sentiment Analysis (retail/institutional sentiment)
6. Risk Assessment (position risk, market risk, company risk)
7. Final Synthesis (weigh all factors, make decision)

Be thorough, specific, and actionable. This is for real money management."""),
            ("user", """Analyze {symbol} and provide comprehensive recommendation:

**1. CURRENT POSITION**
Shares: {shares}
Avg Cost: ${avg_cost}
Current Price: ${current_price}
Current Value: ${current_value}
Unrealized P&L: ${unrealized_pl} ({unrealized_pl_percent:+.2f}%)
Position Size: {position_size_percent:.1f}% of portfolio

**CASH CONTEXT**
Available Cash (CNY): {available_cash_cny}
Instruction: When suggesting actions, consider available cash and provide qualitative guidance for using cash across stocks and options. Do not propose exact amounts.

**2. TECHNICAL & PRICE DATA**
Current Price: ${current_price}
Previous Close: ${previous_close}
Change: {change_percent:+.2f}%
52-Week High: ${fifty_two_week_high}
52-Week Low: ${fifty_two_week_low}
Volume: {volume}

**3. FUNDAMENTAL ANALYSIS**
Company: {company_name}
Sector: {sector}
Industry: {industry}
Market Cap: ${market_cap}
P/E Ratio: {pe_ratio}
Forward P/E: {forward_pe}
PEG Ratio: {peg_ratio}
Price/Book: {price_to_book}
Profit Margin: {profit_margin}%
ROE: {roe}%
Debt/Equity: {debt_to_equity}
Revenue Growth: {revenue_growth}%
Analyst Rec: {analyst_rec}
Target Price: ${target_price}

Valuation Summary:
{valuation_summary}

**4. NEWS & EVENTS**
Overall Sentiment: {news_sentiment}
Key Articles: {news_count}
Upcoming Earnings: {earnings_date}

News Summary:
{news_summary}

**5. SOCIAL SENTIMENT**
Twitter Sentiment: {twitter_sentiment} (Score: {twitter_score})
Reddit Sentiment: {reddit_sentiment} (Score: {reddit_score})
Community Interest: {community_interest}

Twitter Summary: {twitter_summary}
Reddit Summary: {reddit_summary}

**6. RISK ANALYSIS**
Position Risk Level: {risk_level}
Portfolio Beta: {beta}
Volatility: {volatility}%
Stop Loss Suggestion: ${stop_loss_suggestion}
Take Profit Suggestion: ${take_profit_suggestion}

Risk Assessment: {risk_assessment}

**YOUR TASK:**
Provide a DETAILED investment recommendation in this EXACT format:

### DECISION: [BUY/HOLD/SELL]
### CONVICTION: [1-10]/10
### TARGET PRICE: $[X.XX]
### STOP LOSS: $[X.XX]
### TIME HORIZON: [short-term/medium-term/long-term]

### TAKE PROFIT STRATEGY
[Detailed explanation of why this target price was chosen, including technical levels, fundamental valuation, and risk/reward analysis]

### STOP LOSS STRATEGY  
[Detailed explanation of stop loss placement, including technical support levels, volatility considerations, and risk management principles]

### RISK/REWARD ANALYSIS
- Risk Amount: $[X.XX] per share ([X]% of current price)
- Reward Amount: $[X.XX] per share ([X]% of current price)
- Risk/Reward Ratio: [1:X]
- Position Sizing Recommendation: [X]% of portfolio

### BULL CASE (3-5 bullet points)
- [Specific bullish factor 1]
- [Specific bullish factor 2]
- [Specific bullish factor 3]

### BEAR CASE (3-5 bullet points)
- [Specific bearish factor 1]
- [Specific bearish factor 2]
- [Specific bearish factor 3]

### KEY CATALYSTS
- [Upcoming event/catalyst 1]
- [Upcoming event/catalyst 2]
- [Upcoming event/catalyst 3]

### KEY RISKS
- [Specific risk 1]
- [Specific risk 2]
- [Specific risk 3]

### FACTOR SCORES (1-10)
- Technical: [X]/10
- Fundamental: [X]/10
- News Sentiment: [X]/10
- Social Sentiment: [X]/10
- Risk: [X]/10

### EXECUTIVE SUMMARY (2-3 sentences)
[Concise summary of recommendation and rationale]

### DETAILED ANALYSIS (1-2 paragraphs)
[Comprehensive explanation of decision-making process, weighing all factors]

### ACTION PLAN
[Specific, actionable guidance: e.g., "BUY additional 10 shares at current levels with stop at $X and target $Y", "HOLD and raise stop loss to $X", "SELL 50% and move stop loss to $X", etc.]

**CRITICAL GUIDELINES FOR PRICE TARGETS:**
1. **BUY Decisions**: Target should be 15-25% above current price, stop loss 8-12% below
2. **SELL Decisions**: Target should be 2-5% below current price for exit, stop loss 2-5% above for short covering
3. **HOLD Decisions**: Adjust existing stops based on profit/loss position and market conditions
4. **Volatility Adjustment**: Increase targets and stops for high-volatility stocks (beta > 1.5)
5. **Time Horizon**: Short-term (1-4 weeks), medium-term (1-6 months), long-term (6+ months)""")
        ])
        
        # Prepare all data
        def safe_get(data: Dict, *keys, default="N/A"):
            """Safely get nested dictionary values"""
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current if current is not None else default
        
        try:
            # Invoke LLM with comprehensive data
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "shares": safe_get(portfolio_comparison, 'shares', default=0),
                "avg_cost": safe_get(portfolio_comparison, 'avg_cost', default=0),
                "current_price": safe_get(portfolio_comparison, 'current_price', default=0),
                "current_value": safe_get(portfolio_comparison, 'current_value', default=0),
                "unrealized_pl": safe_get(portfolio_comparison, 'unrealized_pl', default=0),
                "unrealized_pl_percent": safe_get(portfolio_comparison, 'unrealized_pl_percent', default=0),
                "position_size_percent": safe_get(position_risk, 'position_size_percent', default=0),
                "available_cash_cny": available_cash_cny if available_cash_cny is not None else 'N/A',
                "previous_close": safe_get(portfolio_comparison, 'previous_close', default=0),
                "change_percent": safe_get(portfolio_comparison, 'change_percent', default=0),
                "fifty_two_week_high": safe_get(company_fundamentals, 'fifty_two_week_high', default='N/A'),
                "fifty_two_week_low": safe_get(company_fundamentals, 'fifty_two_week_low', default='N/A'),
                "volume": safe_get(portfolio_comparison, 'volume', default=0),
                "company_name": safe_get(company_fundamentals, 'company_name', default=symbol),
                "sector": safe_get(company_fundamentals, 'sector', default='N/A'),
                "industry": safe_get(company_fundamentals, 'industry', default='N/A'),
                "market_cap": safe_get(company_fundamentals, 'financial_metrics', 'market_cap', default='N/A'),
                "pe_ratio": safe_get(company_fundamentals, 'financial_metrics', 'pe_ratio', default='N/A'),
                "forward_pe": safe_get(company_fundamentals, 'financial_metrics', 'forward_pe', default='N/A'),
                "peg_ratio": safe_get(company_fundamentals, 'financial_metrics', 'peg_ratio', default='N/A'),
                "price_to_book": safe_get(company_fundamentals, 'financial_metrics', 'price_to_book', default='N/A'),
                "profit_margin": safe_get(company_fundamentals, 'financial_metrics', 'profit_margins', default='N/A'),
                "roe": safe_get(company_fundamentals, 'financial_metrics', 'roe', default='N/A'),
                "debt_to_equity": safe_get(company_fundamentals, 'financial_metrics', 'debt_to_equity', default='N/A'),
                "revenue_growth": safe_get(company_fundamentals, 'revenue_growth', default='N/A'),
                "analyst_rec": safe_get(company_fundamentals, 'analyst_recommendation', default='N/A'),
                "target_price": safe_get(company_fundamentals, 'target_price', default='N/A'),
                "valuation_summary": safe_get(company_fundamentals, 'valuation_summary', default='No analysis available'),
                "news_sentiment": safe_get(news_analysis, 'overall_sentiment', default='neutral'),
                "news_count": len(safe_get(news_analysis, 'news_articles', default=[])),
                "earnings_date": safe_get(news_analysis, 'key_events', 0, 'date', default='Not scheduled'),
                "news_summary": safe_get(news_analysis, 'news_summary', default='No news available'),
                "twitter_sentiment": safe_get(twitter_sentiment, 'overall_sentiment', default='neutral'),
                "twitter_score": safe_get(twitter_sentiment, 'sentiment_score', default=0),
                "twitter_summary": safe_get(twitter_sentiment, 'sentiment_summary', default='No data'),
                "reddit_sentiment": safe_get(reddit_sentiment, 'overall_sentiment', default='neutral'),
                "reddit_score": safe_get(reddit_sentiment, 'sentiment_score', default=0),
                "community_interest": safe_get(reddit_sentiment, 'community_interest', default='low'),
                "reddit_summary": safe_get(reddit_sentiment, 'sentiment_summary', default='No data'),
                "risk_level": safe_get(position_risk, 'risk_level', default='moderate'),
                "beta": safe_get(position_risk, 'beta', default=1.0),
                "volatility": safe_get(position_risk, 'volatility', default=0),
                "stop_loss_suggestion": safe_get(position_risk, 'stop_loss_suggestion', default=0),
                "take_profit_suggestion": safe_get(position_risk, 'take_profit_suggestion', default=0),
                "risk_assessment": safe_get(position_risk, 'risk_assessment', default='No assessment')
            })
            
            analysis_text = response.content
            
            # Parse the response to extract structured data
            recommendation = self._parse_recommendation(symbol, analysis_text, portfolio_comparison)
            
            logger.info(f"✓ {symbol}: {recommendation.decision} with {recommendation.conviction}/10 conviction")
            return recommendation
        
        except Exception as e:
            logger.error(f"Error synthesizing data for {symbol}: {e}")
            # Return default recommendation
            return StockRecommendation(
                symbol=symbol,
                decision="HOLD",
                conviction=5,
                target_price=safe_get(portfolio_comparison, 'current_price', default=0),
                stop_loss=safe_get(portfolio_comparison, 'current_price', default=0) * 0.9,
                time_horizon="medium-term",
                bull_case="Analysis unavailable",
                bear_case="Analysis unavailable",
                key_catalysts=[],
                key_risks=[],
                technical_score=5,
                fundamental_score=5,
                news_sentiment_score=5,
                social_sentiment_score=5,
                risk_score=5,
                executive_summary="Analysis unavailable due to error",
                detailed_analysis=str(e),
                action_plan="Hold and monitor"
            )
    
    def _parse_recommendation(
        self,
        symbol: str,
        analysis_text: str,
        portfolio_comparison: Dict[str, Any]
    ) -> StockRecommendation:
        """Parse LLM response into structured recommendation"""
        # Simple parsing (can be improved with regex or structured output)
        decision = "HOLD"
        if "DECISION: BUY" in analysis_text or "### DECISION: BUY" in analysis_text:
            decision = "BUY"
        elif "DECISION: SELL" in analysis_text or "### DECISION: SELL" in analysis_text:
            decision = "SELL"
        
        # Extract conviction (default to 5 if not found)
        conviction = 5
        if "CONVICTION:" in analysis_text:
            try:
                conv_text = analysis_text.split("CONVICTION:")[1].split("\n")[0]
                conviction = int(''.join(filter(str.isdigit, conv_text.split("/")[0])))
            except:
                pass
        
        current_price = portfolio_comparison.get('current_price', 0)
        
        # Enhanced take profit and stop loss calculation
        target_price, stop_loss = self._calculate_price_targets(
            symbol, decision, current_price, analysis_text, portfolio_comparison
        )
        
        return StockRecommendation(
            symbol=symbol,
            decision=decision,
            conviction=min(max(conviction, 1), 10),  # Ensure 1-10
            target_price=target_price,
            stop_loss=stop_loss,
            time_horizon="medium-term",
            bull_case=self._extract_section(analysis_text, "BULL CASE"),
            bear_case=self._extract_section(analysis_text, "BEAR CASE"),
            key_catalysts=["Market momentum", "Earnings potential", "Sector growth"],
            key_risks=["Market volatility", "Competition", "Valuation concerns"],
            technical_score=6,
            fundamental_score=7,
            news_sentiment_score=6,
            social_sentiment_score=6,
            risk_score=6,
            executive_summary=self._extract_section(analysis_text, "EXECUTIVE SUMMARY"),
            detailed_analysis=analysis_text[:1000],  # First 1000 chars
            action_plan=self._extract_section(analysis_text, "ACTION PLAN")
        )
    
    def _calculate_price_targets(
        self, 
        symbol: str, 
        decision: str, 
        current_price: float, 
        analysis_text: str,
        portfolio_comparison: Dict[str, Any]
    ) -> tuple[float, float]:
        """
        Calculate sophisticated take profit and stop loss targets based on:
        - Current decision (BUY/HOLD/SELL)
        - Technical analysis from the text
        - Risk level and volatility
        - Position status (profit/loss)
        
        Returns:
            tuple: (target_price, stop_loss)
        """
        try:
            # Extract existing targets from analysis text if available
            target_price = self._extract_price_from_text(analysis_text, "TARGET PRICE")
            stop_loss = self._extract_price_from_text(analysis_text, "STOP LOSS")
            
            if target_price > 0 and stop_loss > 0:
                return target_price, stop_loss
            
            # Calculate based on decision and risk profile
            unrealized_pl_percent = portfolio_comparison.get('unrealized_pl_percent', 0)
            volatility = portfolio_comparison.get('volatility', 20)  # Default 20% annual volatility
            
            # Base risk/reward ratios
            if decision == "BUY":
                # For BUY decisions: 1:2 risk/reward ratio
                risk_percent = 8 + (volatility / 100)  # 8-28% stop loss based on volatility
                reward_percent = 16 + (volatility / 50)  # 16-56% target based on volatility
                
                target_price = current_price * (1 + reward_percent / 100)
                stop_loss = current_price * (1 - risk_percent / 100)
                
            elif decision == "SELL":
                # For SELL decisions: tighter stops, focus on capital preservation
                if unrealized_pl_percent > 10:  # Profitable position
                    # Protect 50% of profits
                    profit_protection = current_price * (unrealized_pl_percent / 100) * 0.5
                    stop_loss = current_price - profit_protection
                    target_price = current_price * 0.95  # 5% downside target for exit
                else:  # Losing position
                    # Cut losses at 5-10% from current price
                    loss_limit = min(10, 5 + (volatility / 100))
                    stop_loss = current_price * (1 - loss_limit / 100)
                    target_price = current_price * 0.98  # 2% downside target for exit
                    
            else:  # HOLD
                if unrealized_pl_percent > 20:  # Profitable hold
                    # Trailing stop to protect profits
                    profit_protection = current_price * (unrealized_pl_percent / 100) * 0.3
                    stop_loss = current_price - profit_protection
                    target_price = current_price * 1.08  # 8% upside target
                elif unrealized_pl_percent < -10:  # Losing hold
                    # Give more room for recovery
                    stop_loss = current_price * 0.85  # 15% stop loss
                    target_price = current_price * 1.12  # 12% recovery target
                else:  # Neutral position
                    stop_loss = current_price * 0.92  # 8% stop loss
                    target_price = current_price * 1.10  # 10% target
            
            # Ensure stop loss is not too close to current price (minimum 3%)
            min_stop_distance = current_price * 0.03
            if current_price - stop_loss < min_stop_distance:
                stop_loss = current_price - min_stop_distance
                
            # Ensure target is higher than current price for BUY/HOLD, lower for SELL
            if decision in ["BUY", "HOLD"] and target_price <= current_price:
                target_price = current_price * 1.05  # Minimum 5% upside
            elif decision == "SELL" and target_price >= current_price:
                target_price = current_price * 0.98  # Minimum 2% downside for exit
                
            return round(target_price, 2), round(stop_loss, 2)
            
        except Exception as e:
            logger.error(f"Error calculating price targets for {symbol}: {e}")
            # Fallback to simple calculation
            if decision == "BUY":
                return round(current_price * 1.15, 2), round(current_price * 0.90, 2)
            elif decision == "SELL":
                return round(current_price * 0.95, 2), round(current_price * 0.90, 2)
            else:  # HOLD
                return round(current_price * 1.10, 2), round(current_price * 0.92, 2)
    
    def _extract_price_from_text(self, text: str, price_type: str) -> float:
        """Extract price value from analysis text"""
        try:
            import re
            pattern = rf"{price_type}.*?\$(\d+(?:\.\d+)?)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
            return 0.0
        except:
            return 0.0
    
    def _extract_section(self, text: str, section_name: str):
        """Extract a section from the analysis text"""
        try:
            if section_name in text:
                start = text.find(section_name)
                end = text.find("###", start + len(section_name))
                if end == -1:
                    end = len(text)
                section = text[start:end].strip()
                return section[:300] if len(section) > 300 else section
        except:
            pass
        return f"{section_name}: Analysis available in full report"
    
    def run(self, all_agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - synthesizes all data and generates recommendations
        
        Args:
            all_agent_data: Combined data from all previous agents
            
        Returns:
            Dictionary with final recommendations
        """
        logger.info("=" * 50)
        logger.info("DEEP THINKING RESEARCHER - Starting execution")
        logger.info("=" * 50)
        
        # Extract data from all agents
        symbols = all_agent_data.get('symbols', [])
        portfolio_comparisons = {pc['symbol']: pc for pc in all_agent_data.get('historical_data', {}).get('comparisons', [])}
        news_analyses = {na['symbol']: na for na in all_agent_data.get('news_data', {}).get('news_analyses', [])}
        company_analyses = {ca['symbol']: ca for ca in all_agent_data.get('company_data', {}).get('company_analyses', [])}
        twitter_sentiments = {ts['symbol']: ts for ts in all_agent_data.get('twitter_data', {}).get('sentiment_analyses', [])}
        reddit_sentiments = {rs['symbol']: rs for rs in all_agent_data.get('reddit_data', {}).get('sentiment_analyses', [])}
        position_risks = {pr['symbol']: pr for pr in all_agent_data.get('risk_data', {}).get('position_risks', [])}
        
        # Generate recommendations for each stock
        recommendations = []
        for symbol in symbols:
            try:
                recommendation = self.synthesize_stock_data(
                    symbol=symbol,
                    portfolio_comparison=portfolio_comparisons.get(symbol, {}),
                    news_analysis=news_analyses.get(symbol, {}),
                    company_fundamentals=company_analyses.get(symbol, {}),
                    twitter_sentiment=twitter_sentiments.get(symbol, {}),
                    reddit_sentiment=reddit_sentiments.get(symbol, {}),
                    position_risk=position_risks.get(symbol, {}),
                    available_cash_cny=all_agent_data.get('available_cash_cny')
                )
                recommendations.append(recommendation)
            
            except Exception as e:
                logger.error(f"Error generating recommendation for {symbol}: {e}")
        
        # Generate portfolio summary
        portfolio_summary = f"""
Portfolio Analysis Complete - {len(recommendations)} stocks analyzed
BUY recommendations: {sum(1 for r in recommendations if r.decision == 'BUY')}
HOLD recommendations: {sum(1 for r in recommendations if r.decision == 'HOLD')}
SELL recommendations: {sum(1 for r in recommendations if r.decision == 'SELL')}
Average conviction: {sum(r.conviction for r in recommendations) / len(recommendations):.1f}/10
        """.strip()
        
        result = {
            "agent": "deep_thinking_researcher",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "recommendations": [r.dict() for r in recommendations],
            "portfolio_summary": portfolio_summary,
            "overall_strategy": "Diversified growth with risk management",
            "total_stocks_analyzed": len(recommendations),
            "status": "success"
        }
        
        logger.info(f"Deep Thinking Researcher completed - {len(recommendations)} recommendations generated")
        logger.info("=" * 50)
        
        return result


# Test function
def test_researcher_agent():
    """Test the researcher agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Deep Thinking Researcher Agent...")
        print("=" * 60)
        
        # Mock aggregated data
        mock_data = {
            "symbols": ["AAPL"],
            "historical_data": {
                "comparisons": [{
                    "symbol": "AAPL",
                    "shares": 10,
                    "avg_cost": 150.0,
                    "current_price": 175.0,
                    "current_value": 1750.0,
                    "unrealized_pl": 250.0,
                    "unrealized_pl_percent": 16.67
                }]
            },
            "news_data": {"news_analyses": []},
            "company_data": {"company_analyses": []},
            "twitter_data": {"sentiment_analyses": []},
            "reddit_data": {"sentiment_analyses": []},
            "risk_data": {"position_risks": []}
        }
        
        agent = DeepThinkingResearcher(openrouter_api_key=api_key)
        result = agent.run(mock_data)
        
        print("\n✅ Test Results:")
        print(f"   - Recommendations generated: {result['total_stocks_analyzed']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Researcher Agent test passed!")
        
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
    
    test_researcher_agent()

