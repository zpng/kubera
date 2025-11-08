"""
Bear Researcher Agent
Builds the strongest case for caution/selling
Uses GPT-5 for pattern recognition and skeptical analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompts import BEAR_RESEARCHER_SYSTEM

logger = logging.getLogger(__name__)


class BearResearcher(BaseAgent):
    """
    Bear Researcher agent - skeptical analyst identifying risks and overvaluation.
    Uses GPT-5 for pattern recognition and critical analysis.
    """

    def __init__(self):
        super().__init__(
            name="Bear Researcher",
            role="Skeptical analyst identifying risks and overvaluation",
            model="gpt-5",  # GPT-5 for pattern recognition
            temperature=0.6,
            max_tokens=2000
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the bear case for avoiding/selling the stock.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - analyst_reports: Dict with all 4 analyst reports
                - current_price: Current stock price

        Returns:
            Bear thesis report with conviction rating and downside target
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            analyst_reports = data.get('analyst_reports', {})
            current_price = data.get('current_price', 0)

            # Format analyst reports for prompt
            reports_summary = self._format_analyst_reports(analyst_reports)

            # Create user prompt
            user_prompt = f"""Based on the analyst reports below, build the strongest possible BEAR CASE for avoiding or selling {symbol}.

Current Price: ${current_price:.2f}

ANALYST REPORTS:
{reports_summary}

Your task:
1. Identify the 3-5 strongest bearish points from the analyst reports
2. Build compelling arguments for why the stock will go DOWN or stay flat
3. Highlight risks, headwinds, and red flags
4. Address potential bull arguments with skepticism
5. Provide a conviction rating (1-10, where 10 = extremely bearish)
6. Provide a 6-12 month downside target with reasoning

Focus on:
- Valuation concerns and overvaluation risks
- Negative technical indicators and resistance
- Fundamental weaknesses or deterioration
- Bearish news, risks, and headwinds
- Market threats and competitive pressures

Be critical and skeptical, but cite specific evidence from the reports.
"""

            # Get bear thesis from LLM
            logger.info(f"Building bear case for {symbol}")
            analysis = self._call_llm(
                system_prompt=BEAR_RESEARCHER_SYSTEM,
                user_prompt=user_prompt
            )

            # Extract conviction and price target
            conviction = self._extract_conviction(analysis)
            price_target = self._extract_price_target(analysis, current_price)

            # Create report
            report = self._create_report(
                analysis=analysis,
                confidence=conviction / 10.0,  # Convert to 0-1 scale
                metadata={
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'conviction_rating': conviction,
                    'price_target': price_target,
                    'recommendation': 'SELL',
                    'perspective': 'bearish',
                    'analysis_type': 'investment_research'
                }
            )

            logger.info(f"✅ Bear case complete for {symbol} (Conviction: {conviction}/10, Target: ${price_target:.2f})")
            return report

        except Exception as e:
            logger.error(f"Bear research failed: {e}")
            raise

    def _format_analyst_reports(self, reports: Dict[str, Any]) -> str:
        """Format analyst reports for the prompt."""
        lines = []

        for report_type, report in reports.items():
            if not report:
                continue

            lines.append(f"\n{'='*60}")
            lines.append(f"{report['agent'].upper()}")
            lines.append(f"{'='*60}")

            # Add key metadata
            if 'metadata' in report:
                metadata = report['metadata']
                if report_type == 'market':
                    if 'signals' in metadata:
                        lines.append(f"Signals: {', '.join(metadata['signals'])}")
                elif report_type == 'news':
                    lines.append(f"Sentiment: {metadata.get('overall_sentiment', 'Unknown')}")
                    lines.append(f"Articles: {metadata.get('article_count', 0)}")
                elif report_type == 'sentiment':
                    lines.append(f"Level: {metadata.get('sentiment_level', 'Unknown')}")
                    lines.append(f"RSI: {metadata.get('rsi', 'N/A')}")
                elif report_type == 'fundamentals':
                    if 'key_metrics' in metadata:
                        lines.append("Key Metrics:")
                        for key, value in list(metadata['key_metrics'].items())[:5]:
                            lines.append(f"  - {key}: {value}")

            # Add analysis (truncate if too long)
            analysis = report.get('analysis', '')
            if len(analysis) > 1000:
                analysis = analysis[:1000] + "...[truncated]"
            lines.append(f"\n{analysis}")

        return "\n".join(lines)

    def _extract_conviction(self, analysis: str) -> int:
        """Extract conviction rating from analysis (1-10)."""
        import re

        # Look for patterns like "Conviction: 8/10" or "Conviction Rating: 8"
        patterns = [
            r'conviction[:\s]+(\d+)(?:/10)?',
            r'conviction rating[:\s]+(\d+)(?:/10)?',
            r'rating[:\s]+(\d+)/10'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                rating = int(match.group(1))
                return min(max(rating, 1), 10)  # Clamp to 1-10

        # Default to moderate conviction if not found
        return 6

    def _extract_price_target(self, analysis: str, current_price: float) -> float:
        """Extract downside target from analysis."""
        import re

        # Look for patterns like "$180" or "downside target of $180"
        patterns = [
            r'downside target[:\s]+\$?(\d+\.?\d*)',
            r'price target[:\s]+\$?(\d+\.?\d*)',
            r'target[:\s]+\$(\d+\.?\d*)',
            r'\$(\d+\.?\d*)\s+target'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                target = float(match.group(1))
                # Sanity check: target should be within reasonable range
                if 0.3 * current_price <= target <= 1.5 * current_price:
                    return target

        # Default: assume 10% downside if no target found
        return current_price * 0.90


if __name__ == "__main__":
    # Test the Bear Researcher
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Bear Researcher ===")

    # Mock analyst reports (same as bull researcher test)
    mock_reports = {
        'market': {
            'agent': 'Market Analyst',
            'analysis': 'RSI at 69 approaching overbought. Price below 200-day MA indicates long-term weakness.',
            'metadata': {
                'signals': ['RSI Overbought', 'Price below 200-day SMA']
            }
        },
        'news': {
            'agent': 'News Analyst',
            'analysis': 'While earnings beat, some neutral articles warn of competition and supply chain risks.',
            'metadata': {
                'overall_sentiment': 'Bullish',
                'article_count': 25
            }
        },
        'sentiment': {
            'agent': 'Sentiment Analyst',
            'analysis': 'Greed signals potential for pullback. RSI nearing overbought.',
            'metadata': {
                'sentiment_level': 'Greed',
                'rsi': 69.1
            }
        },
        'fundamentals': {
            'agent': 'Fundamentals Analyst',
            'analysis': 'High P/E ratio of 28 suggests premium valuation. Potential for multiple compression.',
            'metadata': {
                'key_metrics': {
                    'pe_ratio': 28,
                    'revenue_growth_yoy': 0.15,
                    'profit_margin': 0.38
                }
            }
        }
    }

    # Create researcher
    researcher = BearResearcher()

    # Run analysis
    print(f"\nBuilding bear case...")
    result = researcher.analyze({
        'symbol': 'AAPL',
        'analyst_reports': mock_reports,
        'current_price': 225.0
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"Conviction: {result['metadata']['conviction_rating']}/10")
    print(f"Downside Target: ${result['metadata']['price_target']:.2f}")
    print(f"Recommendation: {result['metadata']['recommendation']}")
    print(f"\n{'='*60}")
    print("BEAR THESIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
