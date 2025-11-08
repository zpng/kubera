"""
Fact Checker Agent
Validates that analysis outputs match the actual data and flags hallucinations
"""
import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)


class FactCheckerAgent:
    """
    Validates research outputs against actual data to prevent hallucinations
    """
    
    def __init__(self):
        self.validation_errors = []
        self.hallucination_patterns = [
            # Common hallucination patterns
            (r'\$[\d,]+(?:\.\d{2})?', 'price'),  # Any price mentioned
            (r'-?\d+(?:\.\d+)?%', 'percentage'),  # Any percentage
            (r'\d{4}', 'year'),  # Years (could be wrong)
            (r'52-week (?:low|high)', 'range'),  # 52-week claims
            (r'revenue.*?(-?\d+(?:\.\d+)?%)', 'revenue'),  # Revenue claims
            (r'P/?E.*?(-?\d+(?:\.\d+)?)', 'pe_ratio'),  # P/E claims
        ]
    
    def extract_numbers_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract all numbers and percentages from text"""
        extracted = {}
        for pattern, category in self.hallucination_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[category] = matches
        return extracted
    
    def validate_price_claims(self, rationale: str, symbol: str, actual_data: Dict[str, Any]) -> List[str]:
        """Validate all price claims in the rationale"""
        errors = []
        
        # Get actual prices
        portfolio = actual_data.get("portfolio_position", {})
        historical = actual_data.get("historical_data", {}).get("metrics", {})
        
        actual_current = historical.get("current_price") or portfolio.get("current_price", 0)
        actual_52w_low = historical.get("min_price", None)
        actual_52w_high = historical.get("max_price", None)
        
        # Check for 52-week low/high claims
        if "52-week low" in rationale.lower():
            # Extract the price mentioned near "52-week low"
            low_match = re.search(r'52-week low[^\$]*\$?([\d,]+(?:\.\d{2})?)', rationale, re.IGNORECASE)
            if low_match and actual_52w_low:
                mentioned_low = float(low_match.group(1).replace(',', ''))
                if abs(mentioned_low - actual_52w_low) > actual_52w_low * 0.1:  # More than 10% off
                    errors.append(f"Incorrect 52-week low: mentioned ${mentioned_low:.2f}, actual ${actual_52w_low:.2f}")
        
        if "52-week high" in rationale.lower():
            high_match = re.search(r'52-week high[^\$]*\$?([\d,]+(?:\.\d{2})?)', rationale, re.IGNORECASE)
            if high_match and actual_52w_high:
                mentioned_high = float(high_match.group(1).replace(',', ''))
                if abs(mentioned_high - actual_52w_high) > actual_52w_high * 0.1:
                    errors.append(f"Incorrect 52-week high: mentioned ${mentioned_high:.2f}, actual ${actual_52w_high:.2f}")
        
        # Check current price claims
        current_price_matches = re.findall(r'(?:trading at|current(?:ly)? at|price(?:d)? at)[^\$]*\$?([\d,]+(?:\.\d{2})?)', rationale, re.IGNORECASE)
        for match in current_price_matches:
            mentioned_price = float(match.replace(',', ''))
            if actual_current > 0 and abs(mentioned_price - actual_current) > actual_current * 0.05:
                errors.append(f"Incorrect current price: mentioned ${mentioned_price:.2f}, actual ${actual_current:.2f}")
        
        return errors
    
    def validate_metrics_claims(self, rationale: str, symbol: str, actual_data: Dict[str, Any]) -> List[str]:
        """Validate financial metrics claims"""
        errors = []
        
        company_data = actual_data.get("company_data", {})
        financial_metrics = company_data.get("financial_metrics", {})
        historical = actual_data.get("historical_data", {}).get("metrics", {})
        
        # Check revenue growth claims
        if "revenue growth" in rationale.lower() or "revenue decline" in rationale.lower():
            revenue_match = re.search(r'revenue.*?(-?\d+(?:\.\d+)?%)', rationale, re.IGNORECASE)
            if revenue_match:
                mentioned_revenue = float(revenue_match.group(1).replace('%', ''))
                actual_revenue = financial_metrics.get("revenue_growth")
                if actual_revenue is not None and actual_revenue != "N/A":
                    # yfinance returns revenue_growth as decimal (0.356 = 35.6%)
                    # BUT some stocks may already have it as percentage
                    if abs(actual_revenue) > 10:  # Likely already a percentage
                        actual_revenue_pct = actual_revenue
                    else:  # It's a decimal, convert
                        actual_revenue_pct = actual_revenue * 100
                    
                    if abs(mentioned_revenue - actual_revenue_pct) > 10:  # More than 10 percentage points off
                        errors.append(f"Incorrect revenue growth: mentioned {mentioned_revenue}%, actual {actual_revenue_pct:.1f}%")
        
        # Check P/E ratio claims (distinguish from Forward P/E)
        if "p/e" in rationale.lower() or "pe ratio" in rationale.lower():
            pe_match = re.search(r'P/?E.*?(-?\d+(?:\.\d+)?)', rationale, re.IGNORECASE)
            if pe_match:
                mentioned_pe = float(pe_match.group(1))
                actual_pe = financial_metrics.get("pe_ratio")
                actual_forward_pe = financial_metrics.get("forward_pe")
                
                if actual_pe and actual_pe != "N/A" and isinstance(actual_pe, (int, float)):
                    # Check if they confused P/E with Forward P/E
                    if actual_forward_pe and actual_forward_pe != "N/A" and isinstance(actual_forward_pe, (int, float)):
                        if abs(mentioned_pe - actual_forward_pe) < 2:  # Very close to forward P/E
                            errors.append(f"Used Forward P/E ({actual_forward_pe}) when stating P/E ratio (actual P/E is {actual_pe})")
                    elif abs(mentioned_pe - actual_pe) > max(5, actual_pe * 0.2):  # More than 20% or 5 points off
                        errors.append(f"Incorrect P/E ratio: mentioned {mentioned_pe}, actual {actual_pe}")
        
        # Check debt-to-equity claims
        if "debt" in rationale.lower() and "equity" in rationale.lower():
            de_match = re.search(r'debt[- ]to[- ]equity.*?(\d+(?:\.\d+)?)', rationale, re.IGNORECASE)
            if de_match:
                mentioned_de = float(de_match.group(1))
                actual_de = financial_metrics.get("debt_to_equity")
                
                if actual_de and actual_de != "N/A" and isinstance(actual_de, (int, float)):
                    # Check if value is way off (likely interpreting percentage as ratio)
                    if mentioned_de > 10 and actual_de < 1:
                        errors.append(f"Debt/Equity ratio unit error: mentioned {mentioned_de} (as ratio) but actual is {actual_de:.2f} (should be ~{actual_de:.2%})")
                    elif abs(mentioned_de - actual_de) > max(0.5, actual_de * 0.3):  # More than 30% off
                        errors.append(f"Incorrect debt/equity: mentioned {mentioned_de}, actual {actual_de}")
        
        # Check return/performance claims
        if "return" in rationale.lower() or "gain" in rationale.lower() or "up" in rationale.lower():
            return_match = re.search(r'(?:up|gained?|returned?)[^\d]*?(\d+(?:\.\d+)?%)', rationale, re.IGNORECASE)
            if return_match:
                mentioned_return = float(return_match.group(1).replace('%', ''))
                actual_return = historical.get("period_return_pct")
                if actual_return and abs(mentioned_return - actual_return) > 20:  # More than 20 percentage points off
                    errors.append(f"Incorrect return claim: mentioned {mentioned_return}%, actual {actual_return:.1f}%")
        
        return errors
    
    def check_research_output(self, symbol: str, research_result: Dict[str, Any], all_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a research output against actual data
        
        Returns:
            Dict with validation results and hallucination score
        """
        rationale = research_result.get("rationale", "")
        target_price = research_result.get("target_price", 0)
        decision = research_result.get("decision", "")
        
        validation_errors = []
        
        # 1. Validate price claims
        price_errors = self.validate_price_claims(rationale, symbol, all_data)
        validation_errors.extend(price_errors)
        
        # 2. Validate metrics claims
        metrics_errors = self.validate_metrics_claims(rationale, symbol, all_data)
        validation_errors.extend(metrics_errors)
        
        # 3. Validate target price reasonableness
        portfolio = all_data.get("portfolio_position", {})
        historical = all_data.get("historical_data", {}).get("metrics", {})
        current_price = historical.get("current_price") or portfolio.get("current_price", 0)
        
        if current_price > 0 and target_price > 0:
            price_change_pct = ((target_price - current_price) / current_price) * 100
            
            # Check if target is reasonable given the decision
            if decision == "BUY MORE" and price_change_pct < -5:
                validation_errors.append(f"Illogical: BUY recommendation but target price {price_change_pct:.1f}% below current")
            elif decision == "SELL" and price_change_pct > 10:
                validation_errors.append(f"Illogical: SELL recommendation but target price {price_change_pct:.1f}% above current")
            
            # Check if target is within reasonable bounds
            if abs(price_change_pct) > 50:
                validation_errors.append(f"Unrealistic target: {price_change_pct:.1f}% change from current price")
        
        # 4. Check for data availability claims
        if "no news" in rationale.lower() or "zero news" in rationale.lower():
            news_count = all_data.get("news_data", {}).get("sentiment_analysis", {}).get("article_count", 0)
            if news_count > 5:
                validation_errors.append(f"Incorrect claim: Says 'no news' but {news_count} articles found")
        
        # Calculate hallucination score
        hallucination_score = len(validation_errors)
        is_valid = hallucination_score == 0
        
        # Log errors if found
        if validation_errors:
            logger.warning(f"Fact check failed for {symbol}:")
            for error in validation_errors:
                logger.warning(f"  - {error}")
        
        return {
            "symbol": symbol,
            "is_valid": is_valid,
            "hallucination_score": hallucination_score,
            "validation_errors": validation_errors,
            "checked_fields": ["prices", "metrics", "target_price", "logic"]
        }
    
    def validate_all_research(self, research_results: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all research results
        
        Args:
            research_results: Dict of symbol -> research result
            state: Full workflow state with all data
            
        Returns:
            Validation summary
        """
        logger.info("=" * 60)
        logger.info("FACT CHECKING RESEARCH OUTPUTS")
        logger.info("=" * 60)
        
        validation_results = {}
        total_errors = 0
        
        for symbol, research_result in research_results.items():
            # Aggregate data for this symbol
            all_data = {
                "portfolio_position": next(
                    (s for s in state.get("portfolio_data", {}).get("portfolio", {}).get("stocks", []) 
                     if s["symbol"] == symbol), 
                    {}
                ),
                "historical_data": state.get("historical_data", {}).get(symbol, {}),
                "company_data": state.get("company_data", {}).get(symbol, {}),
                "news_data": state.get("news_data", {}).get(symbol, {}),
            }
            
            # Check this symbol's research
            validation = self.check_research_output(symbol, research_result, all_data)
            validation_results[symbol] = validation
            total_errors += validation["hallucination_score"]
            
            # Log results
            status = "✅ VALID" if validation["is_valid"] else f"⚠️  {len(validation['validation_errors'])} ISSUES"
            logger.info(f"{symbol}: {status}")
            if not validation["is_valid"]:
                for error in validation["validation_errors"]:
                    logger.info(f"   - {error}")
        
        # Overall assessment
        logger.info("-" * 60)
        if total_errors == 0:
            logger.info("✅ ALL RESEARCH OUTPUTS VALIDATED - NO HALLUCINATIONS DETECTED")
        else:
            logger.info(f"⚠️  FOUND {total_errors} POTENTIAL HALLUCINATIONS - REVIEW NEEDED")
        logger.info("=" * 60)
        
        return {
            "validation_results": validation_results,
            "total_errors": total_errors,
            "all_valid": total_errors == 0
        }
