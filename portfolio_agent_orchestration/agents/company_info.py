"""
Company Info Agent
Collects company fundamentals, earnings, and detailed company information
Model: deepseek/deepseek-r1-distill-llama-70b (cost-efficient reasoning for fundamentals)
"""
import logging
from typing import Dict, Any
from datetime import datetime
import yfinance as yf

from .base_agent import BaseAgent
from ..config import AGENT_MODELS

logger = logging.getLogger(__name__)


class CompanyInfoAgent(BaseAgent):
    """
    Agent responsible for fetching company fundamental information
    """
    
    def __init__(self):
        super().__init__(
            name="CompanyInfo",
            model=AGENT_MODELS["company_info"],
            role="Company fundamentals and information analyst",
            temperature=0.4
        )

    def _normalize_debt_to_equity(self, debt_to_equity_value):
        """
        Normalize debt-to-equity ratio to handle different yfinance formats

        yfinance returns debtToEquity as a PERCENTAGE value:
        - 17.08 means 17.08% (NOT 17.08:1 ratio)
        - This equals 0.1708 as a decimal ratio
        
        We'll return it as a decimal ratio for consistency (e.g., 0.1708 for 17.08%)
        """
        if debt_to_equity_value is None or debt_to_equity_value == "N/A":
            return "N/A"

        try:
            de_float = float(debt_to_equity_value)
            
            # yfinance returns this as a percentage (e.g., 17.08 means 17.08%)
            # Convert to decimal ratio: 17.08% = 0.1708 ratio
            if de_float > 0:
                # Convert percentage to decimal ratio
                decimal_ratio = de_float / 100.0
                return round(decimal_ratio, 4)
            else:
                return 0.0

        except (ValueError, TypeError) as e:
            logger.error(f"Error normalizing debt-to-equity {debt_to_equity_value}: {e}")
            return "N/A"

    def fetch_company_info(self, symbol: str) -> Dict[str, Any]:
        """Fetch basic company information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_info = {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "description": info.get("longBusinessSummary", ""),
                "website": info.get("website", ""),
                "employees": info.get("fullTimeEmployees", 0),
                "market_cap": info.get("marketCap", 0),
                "country": info.get("country", ""),
            }
            
            logger.info(f"Fetched company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            return {"name": symbol, "error": str(e)}
    
    def fetch_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """Fetch key financial metrics"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            metrics = {
                # Valuation
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "price_to_sales": info.get("priceToSalesTrailing12Months"),
                
                # Profitability
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                
                # Growth
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
                
                # Financial Health
                "total_cash": info.get("totalCash"),
                "total_cash_millions": round(info.get("totalCash", 0) / 1_000_000, 1) if info.get("totalCash") else None,
                "total_debt": info.get("totalDebt"),
                "debt_to_equity": self._normalize_debt_to_equity(info.get("debtToEquity")),
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "cash_per_share": info.get("totalCashPerShare"),
                "beta": info.get("beta"),
                
                # Dividends
                "dividend_yield": info.get("dividendYield"),
                "payout_ratio": info.get("payoutRatio"),
                
                # Analyst Ratings
                "target_price": info.get("targetMeanPrice"),
                "recommendation": info.get("recommendationKey", "").upper(),
                "number_of_analysts": info.get("numberOfAnalystOpinions"),
            }
            
            # Clean up None values and format percentages
            for key, value in metrics.items():
                if value is None:
                    metrics[key] = "N/A"
                elif isinstance(value, float) and key in ["profit_margin", "operating_margin", "roe", "roa", "revenue_growth", "earnings_growth", "earnings_quarterly_growth", "dividend_yield", "payout_ratio"]:
                    metrics[key] = round(value * 100, 2)  # Convert to percentage
            
            logger.info(f"Fetched financial metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return {}
    
    def fetch_earnings_dates(self, symbol: str) -> Dict[str, Any]:
        """Fetch upcoming and past earnings dates"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try to get earnings dates
            try:
                earnings_dates = ticker.earnings_dates
                if earnings_dates is not None and not earnings_dates.empty:
                    latest_date = earnings_dates.index[0].strftime("%Y-%m-%d")
                    earnings_info = {
                        "next_earnings_date": latest_date,
                        "has_earnings_data": True
                    }
                else:
                    earnings_info = {
                        "next_earnings_date": "N/A",
                        "has_earnings_data": False
                    }
            except:
                earnings_info = {
                    "next_earnings_date": "N/A",
                    "has_earnings_data": False
                }
            
            # Get earnings history
            try:
                earnings_history = ticker.earnings_history
                if earnings_history is not None and not earnings_history.empty:
                    latest_earnings = earnings_history.iloc[0]
                    earnings_info["last_earnings"] = {
                        "eps_estimate": float(latest_earnings.get("epsEstimate", 0)),
                        "eps_actual": float(latest_earnings.get("epsActual", 0)),
                        "surprise_pct": float(latest_earnings.get("surprisePercent", 0)),
                    }
            except:
                pass
            
            logger.info(f"Fetched earnings info for {symbol}")
            return earnings_info
            
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
            return {}
    
    def analyze_fundamentals(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamental metrics and provide insights"""
        analysis = {
            "valuation": "Fair",
            "profitability": "Neutral",
            "growth": "Neutral",
            "financial_health": "Healthy",
            "overall_score": 5  # Out of 10
        }
        
        score = 5  # Start with neutral
        
        try:
            # Valuation analysis
            pe = metrics.get("pe_ratio")
            if pe != "N/A" and isinstance(pe, (int, float)):
                if pe < 15:
                    analysis["valuation"] = "Undervalued"
                    score += 1
                elif pe > 30:
                    analysis["valuation"] = "Overvalued"
                    score -= 1
            
            # Profitability analysis
            profit_margin = metrics.get("profit_margin")
            if profit_margin != "N/A" and isinstance(profit_margin, (int, float)):
                if profit_margin > 20:
                    analysis["profitability"] = "Strong"
                    score += 1
                elif profit_margin < 5:
                    analysis["profitability"] = "Weak"
                    score -= 1
            
            # Growth analysis
            revenue_growth = metrics.get("revenue_growth")
            if revenue_growth != "N/A" and isinstance(revenue_growth, (int, float)):
                if revenue_growth > 15:
                    analysis["growth"] = "Strong"
                    score += 1
                elif revenue_growth < 0:
                    analysis["growth"] = "Declining"
                    score -= 1
            
            # Financial health
            debt_to_equity = metrics.get("debt_to_equity")
            if debt_to_equity != "N/A" and isinstance(debt_to_equity, (int, float)):
                if debt_to_equity > 200:
                    analysis["financial_health"] = "Risky"
                    score -= 1
                elif debt_to_equity < 50:
                    analysis["financial_health"] = "Strong"
                    score += 1
            
            analysis["overall_score"] = max(0, min(10, score))
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals: {e}")
        
        return analysis
    
    def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """Process company info for a single symbol"""
        logger.info(f"Fetching company info for {symbol}")
        
        company_info = self.fetch_company_info(symbol)
        financial_metrics = self.fetch_financial_metrics(symbol)
        earnings_info = self.fetch_earnings_dates(symbol)
        fundamental_analysis = self.analyze_fundamentals(financial_metrics)
        
        return {
            "symbol": symbol,
            "status": "success" if "error" not in company_info else "error",
            "company_info": company_info,
            "financial_metrics": financial_metrics,
            "earnings_info": earnings_info,
            "fundamental_analysis": fundamental_analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with stock_symbols
            
        Returns:
            Updated state with company information
        """
        logger.info(f"[{self.name}] Starting company info collection...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        company_data = {}
        
        for symbol in symbols:
            data = self.process_symbol(symbol)
            company_data[symbol] = data
        
        # Update state
        state["company_data"] = company_data
        state["company_data_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Collected company info for {len(company_data)} symbols")
        return state


# Test function
def test_company_info_agent():
    """Test the CompanyInfoAgent"""
    print("Testing Company Info Agent...")
    
    agent = CompanyInfoAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["AAPL", "MSFT"]
    }
    
    result_state = agent.process(state)
    
    print(f"\nCompany Info Results:")
    for symbol, data in result_state.get("company_data", {}).items():
        print(f"\n{symbol}:")
        print(f"  Name: {data['company_info'].get('name', 'N/A')}")
        print(f"  Sector: {data['company_info'].get('sector', 'N/A')}")
        metrics = data.get('financial_metrics', {})
        print(f"  P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
        print(f"  Profit Margin: {metrics.get('profit_margin', 'N/A')}%")
        analysis = data.get('fundamental_analysis', {})
        print(f"  Overall Score: {analysis.get('overall_score', 'N/A')}/10")
        print(f"  Valuation: {analysis.get('valuation', 'N/A')}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_company_info_agent()

