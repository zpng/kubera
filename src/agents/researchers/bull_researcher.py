"""
Bull Researcher Agent
Builds the strongest possible case for buying the stock
Uses Claude Sonnet 4.5 for deep reasoning and optimistic lens
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompts import BULL_RESEARCHER_SYSTEM

logger = logging.getLogger(__name__)


class BullResearcher(BaseAgent):
    """
    Bull Researcher agent - optimistic analyst seeking growth opportunities.
    Uses Claude Sonnet 4.5 for deep reasoning and constructive analysis.
    """

    def __init__(self):
        super().__init__(
            name="Bull Researcher",
            role="Optimistic investment analyst seeking growth opportunities",
            model="claude-sonnet",  # Claude Sonnet 4.5 for deep reasoning
            temperature=0.6,  # Higher creativity for finding opportunities
            max_tokens=2000
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the bull case for buying the stock.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - analyst_reports: Dict with all 4 analyst reports
                - current_price: Current stock price

        Returns:
            Bull thesis report with conviction rating and price target
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            analyst_reports = data.get('analyst_reports', {})
            current_price = data.get('current_price', 0)

            # Format analyst reports for prompt
            reports_summary = self._format_analyst_reports(analyst_reports)

            # Create user prompt
            user_prompt = f"""请基于下方分析师报告，为 {symbol} 构建最有力的看多论点（BULL CASE）。

当前价格：${current_price:.2f}

分析师报告：
{reports_summary}

任务：
1. 从报告中提炼 3-5 个最强的看多要点
2. 构建有说服力的上行论证
3. 强调增长催化与机会
4. 主动回应潜在看空观点
5. 给出信心评级（1-10）
6. 给出 6-12 个月目标价并说明理由

重点关注：
- 增长潜力与催化剂
- 积极的技术动能
- 扎实的基本面与竞争优势
- 正面的新闻与情绪
- 市场机会与顺风

输出要求：
- 全文使用中文叙述
- 必须在文中单独包含如下两行英文标签，便于系统解析：
  Conviction: [1-10]
  Price Target: $[目标价]
"""

            # Get bull thesis from LLM
            logger.info(f"Building bull case for {symbol}")
            analysis = self._call_llm(
                system_prompt=BULL_RESEARCHER_SYSTEM,
                user_prompt=user_prompt
            )

            # Extract conviction and price target (simple parsing)
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
                    'recommendation': 'BUY',
                    'perspective': 'bullish',
                    'analysis_type': 'investment_research'
                }
            )

            logger.info(f"✅ Bull case complete for {symbol} (Conviction: {conviction}/10, Target: ${price_target:.2f})")
            return report

        except Exception as e:
            logger.error(f"Bull research failed: {e}")
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
        # Simple keyword search for conviction
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

        # Default to moderate-high conviction if not found
        return 7

    def _extract_price_target(self, analysis: str, current_price: float) -> float:
        """Extract price target from analysis."""
        import re

        # Look for patterns like "$150" or "target of $150" or "price target: $150"
        patterns = [
            r'price target[:\s]+\$?(\d+\.?\d*)',
            r'target[:\s]+\$(\d+\.?\d*)',
            r'\$(\d+\.?\d*)\s+target'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                target = float(match.group(1))
                # Sanity check: target should be within reasonable range of current price
                if 0.5 * current_price <= target <= 3 * current_price:
                    return target

        # Default: assume 15% upside if no target found
        return current_price * 1.15


if __name__ == "__main__":
    # Test the Bull Researcher
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Bull Researcher ===")

    # Mock analyst reports
    mock_reports = {
        'market': {
            'agent': 'Market Analyst',
            'analysis': 'Strong uptrend with RSI at 69. Price above 50-day MA. Bullish MACD crossover detected.',
            'metadata': {
                'signals': ['Price above 50-day SMA', 'MACD Bullish Crossover']
            }
        },
        'news': {
            'agent': 'News Analyst',
            'analysis': 'Positive earnings beat. New product launch announced. Strong demand signals.',
            'metadata': {
                'overall_sentiment': 'Bullish',
                'article_count': 25
            }
        },
        'sentiment': {
            'agent': 'Sentiment Analyst',
            'analysis': 'Moderate greed. Investors optimistic about growth prospects.',
            'metadata': {
                'sentiment_level': 'Greed',
                'rsi': 69.1
            }
        },
        'fundamentals': {
            'agent': 'Fundamentals Analyst',
            'analysis': 'Strong revenue growth. Solid profit margins. Market leader in sector.',
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
    researcher = BullResearcher()

    # Run analysis
    print(f"\nBuilding bull case...")
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
    print(f"Price Target: ${result['metadata']['price_target']:.2f}")
    print(f"Recommendation: {result['metadata']['recommendation']}")
    print(f"\n{'='*60}")
    print("BULL THESIS:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
