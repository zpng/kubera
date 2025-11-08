"""
Fundamentals Analyst Agent
Analyzes company fundamentals, valuation, and financial health
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompts import (
    FUNDAMENTALS_ANALYST_SYSTEM,
    FUNDAMENTALS_ANALYST_USER,
    format_company_info,
    format_financial_metrics
)

logger = logging.getLogger(__name__)


class FundamentalsAnalyst(BaseAgent):
    """
    Fundamentals Analyst agent specializing in valuation and financial analysis.
    Uses GPT-4o-mini for fundamental analysis.
    """

    def __init__(self):
        super().__init__(
            name="Fundamentals Analyst",
            role="Company valuation and financial analysis expert",
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1500
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze company fundamentals and valuation.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - company_info: Company overview data
                - fundamentals: Financial metrics and ratios
                - earnings: Earnings data (optional)

        Returns:
            Analysis report with fundamental outlook
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            company_info = data.get('company_info', {})
            fundamentals = data.get('fundamentals', {})
            earnings = data.get('earnings', {})

            # Format data for prompt
            company_text = format_company_info(company_info)
            metrics_text = format_financial_metrics(fundamentals)
            earnings_text = self._format_earnings(earnings) if earnings else "Earnings data not available"

            # Create user prompt
            user_prompt = FUNDAMENTALS_ANALYST_USER.format(
                symbol=symbol,
                company_info=company_text,
                financial_metrics=metrics_text,
                earnings_data=earnings_text
            )

            # Get analysis from LLM
            logger.info(f"Analyzing fundamentals for {symbol}")
            analysis = self._call_llm(
                system_prompt=FUNDAMENTALS_ANALYST_SYSTEM,
                user_prompt=user_prompt
            )

            # Extract key metrics for metadata
            key_metrics = self._extract_key_metrics(fundamentals, company_info)

            # Create report
            report = self._create_report(
                analysis=analysis,
                metadata={
                    'symbol': symbol,
                    'key_metrics': key_metrics,
                    'analysis_type': 'fundamentals'
                }
            )

            logger.info(f"✅ Fundamentals analysis complete for {symbol}")
            return report

        except Exception as e:
            logger.error(f"Fundamentals analysis failed: {e}")
            raise

    def _format_earnings(self, earnings: Dict[str, Any]) -> str:
        """Format earnings data for prompt."""
        if not earnings:
            return "No earnings data available"

        lines = []

        # Annual earnings
        if 'annualEarnings' in earnings and earnings['annualEarnings']:
            lines.append("Annual Earnings (Recent):")
            for item in earnings['annualEarnings'][:3]:  # Last 3 years
                fiscal_date = item.get('fiscalDateEnding', 'Unknown')
                eps = item.get('reportedEPS', 'N/A')
                lines.append(f"  - {fiscal_date}: EPS ${eps}")

        # Quarterly earnings
        if 'quarterlyEarnings' in earnings and earnings['quarterlyEarnings']:
            lines.append("\nQuarterly Earnings (Recent):")
            for item in earnings['quarterlyEarnings'][:4]:  # Last 4 quarters
                fiscal_date = item.get('fiscalDateEnding', 'Unknown')
                eps = item.get('reportedEPS', 'N/A')
                surprise = item.get('surprisePercentage', 'N/A')
                lines.append(f"  - {fiscal_date}: EPS ${eps} (Surprise: {surprise}%)")

        return "\n".join(lines) if lines else "Limited earnings data available"

    def _extract_key_metrics(
        self,
        fundamentals: Dict[str, Any],
        company_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract key metrics for report metadata."""
        metrics = {}

        # Valuation metrics
        if 'PERatio' in fundamentals:
            metrics['pe_ratio'] = fundamentals['PERatio']
        if 'ForwardPE' in fundamentals:
            metrics['forward_pe'] = fundamentals['ForwardPE']
        if 'PriceToBook' in fundamentals:
            metrics['price_to_book'] = fundamentals['PriceToBook']
        if 'PEGRatio' in fundamentals:
            metrics['peg_ratio'] = fundamentals['PEGRatio']

        # Growth metrics
        if 'QuarterlyEarningsGrowthYOY' in fundamentals:
            metrics['earnings_growth_yoy'] = fundamentals['QuarterlyEarningsGrowthYOY']
        if 'QuarterlyRevenueGrowthYOY' in fundamentals:
            metrics['revenue_growth_yoy'] = fundamentals['QuarterlyRevenueGrowthYOY']

        # Profitability
        if 'ProfitMargin' in fundamentals:
            metrics['profit_margin'] = fundamentals['ProfitMargin']
        if 'ReturnOnEquityTTM' in fundamentals:
            metrics['roe'] = fundamentals['ReturnOnEquityTTM']

        # Company info
        if 'sector' in company_info:
            metrics['sector'] = company_info['sector']
        if 'industry' in company_info:
            metrics['industry'] = company_info['industry']
        if 'marketCap' in company_info:
            metrics['market_cap'] = company_info['marketCap']

        return metrics


if __name__ == "__main__":
    # Test the Fundamentals Analyst
    logging.basicConfig(level=logging.INFO)

    from data import get_fundamentals
    from data.market_data import MarketDataProvider

    print("\n=== Testing Fundamentals Analyst ===")

    # Get data
    symbol = "AAPL"
    print(f"\nFetching fundamentals for {symbol}...")

    market_data = MarketDataProvider()
    company_info = market_data.get_company_info(symbol)
    fundamentals = get_fundamentals(symbol)

    # Try to get earnings (may fail with rate limits)
    try:
        from data.alpha_vantage import AlphaVantageProvider
        av = AlphaVantageProvider()
        earnings = av.get_earnings(symbol)
    except Exception as e:
        print(f"Could not fetch earnings (likely rate limit): {e}")
        earnings = {}

    # Create analyst
    analyst = FundamentalsAnalyst()

    # Run analysis
    print(f"\nRunning fundamental analysis...")
    result = analyst.analyze({
        'symbol': symbol,
        'company_info': company_info,
        'fundamentals': fundamentals,
        'earnings': earnings
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"\nKey Metrics:")
    for key, value in result['metadata']['key_metrics'].items():
        print(f"  - {key}: {value}")
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
