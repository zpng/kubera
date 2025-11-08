"""
Company Info Agent
Collects comprehensive company fundamentals, earnings, and financial data
Model: deepseek/deepseek-r1-distill-llama-70b (cost-efficient reasoning for financial analysis)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import yfinance as yf

logger = logging.getLogger(__name__)


class FinancialMetrics(BaseModel):
    """Schema for financial metrics"""
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    profit_margins: Optional[float] = None
    operating_margins: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None


class CompanyFundamentals(BaseModel):
    """Schema for company fundamentals"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    description: str
    financial_metrics: FinancialMetrics
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    dividend_yield: Optional[float] = None
    analyst_recommendation: Optional[str] = None
    target_price: Optional[float] = None
    valuation_summary: str


@tool
def get_company_fundamentals(symbol: str) -> Dict[str, Any]:
    """
    Fetch comprehensive company fundamentals
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with company information
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        return {
            "symbol": symbol,
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "description": info.get('longBusinessSummary', ''),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "forward_pe": info.get('forwardPE'),
            "peg_ratio": info.get('pegRatio'),
            "price_to_book": info.get('priceToBook'),
            "price_to_sales": info.get('priceToSalesTrailing12Months'),
            "profit_margins": info.get('profitMargins'),
            "operating_margins": info.get('operatingMargins'),
            "roe": info.get('returnOnEquity'),
            "roa": info.get('returnOnAssets'),
            "debt_to_equity": info.get('debtToEquity'),
            "current_ratio": info.get('currentRatio'),
            "quick_ratio": info.get('quickRatio'),
            "revenue": info.get('totalRevenue'),
            "revenue_growth": info.get('revenueGrowth'),
            "earnings_growth": info.get('earningsGrowth'),
            "dividend_yield": info.get('dividendYield'),
            "analyst_recommendation": info.get('recommendationKey'),
            "target_price": info.get('targetMeanPrice')
        }
    
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


@tool
def get_quarterly_earnings(symbol: str) -> Dict[str, Any]:
    """
    Fetch quarterly earnings history
    
    Args:
        symbol: Stock ticker symbol
        
    Returns:
        Dictionary with earnings data
    """
    try:
        stock = yf.Ticker(symbol)
        quarterly_earnings = stock.quarterly_earnings
        
        if quarterly_earnings is None or quarterly_earnings.empty:
            return {"symbol": symbol, "earnings": []}
        
        # Get latest 4 quarters
        earnings_data = []
        for idx, row in quarterly_earnings.head(4).iterrows():
            earnings_data.append({
                "quarter": str(idx),
                "revenue": float(row.get('Revenue', 0)),
                "earnings": float(row.get('Earnings', 0))
            })
        
        return {
            "symbol": symbol,
            "quarterly_earnings": earnings_data
        }
    
    except Exception as e:
        logger.error(f"Error fetching earnings for {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


class CompanyInfoAgent:
    """
    Agent responsible for collecting comprehensive company information
    Uses: deepseek/deepseek-r1-distill-llama-70b for efficient financial reasoning
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "deepseek/deepseek-r1-distill-llama-70b"
    ):
        """
        Initialize Company Info Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: deepseek/deepseek-r1-distill-llama-70b)
        """
        self.model = model
        
        # Initialize LLM for fundamental analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.2  # Low temperature for analytical precision
        )
        
        # Bind tools
        self.tools = [get_company_fundamentals, get_quarterly_earnings]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        logger.info(f"Company Info Agent initialized with model: {model}")
    
    def fetch_company_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive company data
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company fundamentals and earnings
        """
        logger.info(f"Fetching company data for {symbol}...")
        
        # Fetch fundamentals
        fundamentals = get_company_fundamentals.invoke({"symbol": symbol})
        
        # Fetch earnings
        earnings = get_quarterly_earnings.invoke({"symbol": symbol})
        
        return {
            "symbol": symbol,
            "fundamentals": fundamentals,
            "earnings": earnings,
            "fetched_at": datetime.now().isoformat()
        }
    
    def analyze_valuation(self, symbol: str, company_data: Dict[str, Any]) -> CompanyFundamentals:
        """
        Analyze company valuation using LLM
        
        Args:
            symbol: Stock ticker symbol
            company_data: Raw company data
            
        Returns:
            Structured company fundamentals analysis
        """
        fundamentals = company_data.get('fundamentals', {})
        earnings = company_data.get('earnings', {})
        
        # Create analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a fundamental analyst specializing in company valuation and financial analysis.

Your task is to analyze a company's fundamentals and provide:
1. Valuation assessment (overvalued/fairly valued/undervalued)
2. Financial health analysis (strong/moderate/weak)
3. Growth prospects evaluation
4. Key strengths and concerns
5. Investment thesis summary

Focus on:
- Price multiples (P/E, P/B, P/S ratios)
- Profitability metrics (margins, ROE, ROA)
- Financial leverage (debt ratios)
- Liquidity (current ratio, quick ratio)
- Growth rates (revenue, earnings)
- Industry comparisons

Provide a concise yet comprehensive analysis suitable for investment decision-making."""),
            ("user", """Analyze the following company data for {symbol}:

Company Information:
- Name: {company_name}
- Sector: {sector}
- Industry: {industry}

Financial Metrics:
- Market Cap: ${market_cap}
- P/E Ratio: {pe_ratio}
- Forward P/E: {forward_pe}
- PEG Ratio: {peg_ratio}
- Price/Book: {price_to_book}
- Price/Sales: {price_to_sales}
- Profit Margin: {profit_margin}%
- Operating Margin: {operating_margin}%
- ROE: {roe}%
- ROA: {roa}%
- Debt/Equity: {debt_to_equity}
- Current Ratio: {current_ratio}

Growth Metrics:
- Revenue Growth: {revenue_growth}%
- Earnings Growth: {earnings_growth}%

Analyst Recommendation: {analyst_rec}
Target Price: ${target_price}

Provide a valuation summary and investment thesis.""")
        ])
        
        # Format data
        def safe_format(value, decimal_places=2, prefix="", suffix=""):
            if value is None:
                return "N/A"
            if isinstance(value, (int, float)):
                return f"{prefix}{value:.{decimal_places}f}{suffix}"
            return str(value)
        
        # Invoke LLM
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "company_name": fundamentals.get('company_name', symbol),
                "sector": fundamentals.get('sector', 'Unknown'),
                "industry": fundamentals.get('industry', 'Unknown'),
                "market_cap": safe_format(fundamentals.get('market_cap'), 0),
                "pe_ratio": safe_format(fundamentals.get('pe_ratio')),
                "forward_pe": safe_format(fundamentals.get('forward_pe')),
                "peg_ratio": safe_format(fundamentals.get('peg_ratio')),
                "price_to_book": safe_format(fundamentals.get('price_to_book')),
                "price_to_sales": safe_format(fundamentals.get('price_to_sales')),
                "profit_margin": safe_format(fundamentals.get('profit_margins', 0) * 100 if fundamentals.get('profit_margins') else None),
                "operating_margin": safe_format(fundamentals.get('operating_margins', 0) * 100 if fundamentals.get('operating_margins') else None),
                "roe": safe_format(fundamentals.get('roe', 0) * 100 if fundamentals.get('roe') else None),
                "roa": safe_format(fundamentals.get('roa', 0) * 100 if fundamentals.get('roa') else None),
                "debt_to_equity": safe_format(fundamentals.get('debt_to_equity')),
                "current_ratio": safe_format(fundamentals.get('current_ratio')),
                "revenue_growth": safe_format(fundamentals.get('revenue_growth', 0) * 100 if fundamentals.get('revenue_growth') else None),
                "earnings_growth": safe_format(fundamentals.get('earnings_growth', 0) * 100 if fundamentals.get('earnings_growth') else None),
                "analyst_rec": fundamentals.get('analyst_recommendation', 'N/A'),
                "target_price": safe_format(fundamentals.get('target_price'))
            })
            
            valuation_summary = response.content
            
            # Create structured output
            financial_metrics = FinancialMetrics(
                market_cap=fundamentals.get('market_cap'),
                pe_ratio=fundamentals.get('pe_ratio'),
                forward_pe=fundamentals.get('forward_pe'),
                peg_ratio=fundamentals.get('peg_ratio'),
                price_to_book=fundamentals.get('price_to_book'),
                price_to_sales=fundamentals.get('price_to_sales'),
                profit_margins=fundamentals.get('profit_margins'),
                operating_margins=fundamentals.get('operating_margins'),
                roe=fundamentals.get('roe'),
                roa=fundamentals.get('roa'),
                debt_to_equity=fundamentals.get('debt_to_equity'),
                current_ratio=fundamentals.get('current_ratio'),
                quick_ratio=fundamentals.get('quick_ratio')
            )
            
            return CompanyFundamentals(
                symbol=symbol,
                company_name=fundamentals.get('company_name', symbol),
                sector=fundamentals.get('sector', 'Unknown'),
                industry=fundamentals.get('industry', 'Unknown'),
                description=fundamentals.get('description', '')[:300],
                financial_metrics=financial_metrics,
                revenue=fundamentals.get('revenue'),
                revenue_growth=fundamentals.get('revenue_growth'),
                earnings_growth=fundamentals.get('earnings_growth'),
                dividend_yield=fundamentals.get('dividend_yield'),
                analyst_recommendation=fundamentals.get('analyst_recommendation'),
                target_price=fundamentals.get('target_price'),
                valuation_summary=valuation_summary[:500] if valuation_summary else "Analysis unavailable"
            )
        
        except Exception as e:
            logger.error(f"Error analyzing valuation with LLM: {e}")
            # Return basic data without LLM analysis
            return CompanyFundamentals(
                symbol=symbol,
                company_name=fundamentals.get('company_name', symbol),
                sector=fundamentals.get('sector', 'Unknown'),
                industry=fundamentals.get('industry', 'Unknown'),
                description=fundamentals.get('description', '')[:300],
                financial_metrics=FinancialMetrics(),
                valuation_summary="Analysis unavailable"
            )
    
    def run(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - fetches and analyzes company data
        
        Args:
            portfolio_data: Portfolio data with symbols
            
        Returns:
            Dictionary with company analysis for each stock
        """
        logger.info("=" * 50)
        logger.info("COMPANY INFO AGENT - Starting execution")
        logger.info("=" * 50)
        
        symbols = portfolio_data.get('symbols', [])
        
        company_analyses = []
        for symbol in symbols:
            try:
                # Fetch company data
                company_data = self.fetch_company_data(symbol)
                
                # Analyze valuation
                analysis = self.analyze_valuation(symbol, company_data)
                
                company_analyses.append(analysis.dict())
                logger.info(f"✓ {symbol}: {analysis.sector} | P/E: {analysis.financial_metrics.pe_ratio}")
            
            except Exception as e:
                logger.error(f"Error processing company data for {symbol}: {e}")
        
        result = {
            "agent": "company_info",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "company_analyses": company_analyses,
            "total_stocks": len(company_analyses),
            "status": "success"
        }
        
        logger.info(f"Company Info Agent completed - {len(company_analyses)} stocks analyzed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_company_info_agent():
    """Test the company info agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Company Info Agent...")
        print("=" * 60)
        
        # Mock portfolio data
        mock_portfolio = {
            "symbols": ["AAPL"]
        }
        
        agent = CompanyInfoAgent(openrouter_api_key=api_key)
        result = agent.run(mock_portfolio)
        
        print("\n✅ Test Results:")
        print(f"   - Companies analyzed: {result['total_stocks']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Company Info Agent test passed!")
        
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
    
    test_company_info_agent()

