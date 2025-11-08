"""
Research Judge Agent
Synthesizes bull and bear arguments to make final recommendation
Uses Gemini 2.5 Pro for balanced multi-modal synthesis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
from typing import Dict, Any, Optional

from agents.base_agent import BaseAgent
from utils.prompts import RESEARCH_JUDGE_SYSTEM

logger = logging.getLogger(__name__)


class ResearchJudge(BaseAgent):
    """
    Research Judge agent - impartial analyst synthesizing bull/bear arguments.
    Uses Gemini 2.5 Pro for balanced judgment and multi-modal synthesis.
    """

    def __init__(self):
        super().__init__(
            name="Research Judge",
            role="Impartial analyst synthesizing investment arguments",
            model="gemini",  # Gemini 2.5 Pro for balanced synthesis
            temperature=0.4,  # Lower temp for balanced decision-making
            max_tokens=2500
        )

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize bull and bear cases into final recommendation.

        Args:
            data: Dictionary containing:
                - symbol: Stock ticker
                - current_price: Current stock price
                - bull_case: Bull researcher's report
                - bear_case: Bear researcher's report
                - analyst_reports: Original analyst reports (optional)

        Returns:
            Final investment recommendation (BUY/HOLD/SELL) with reasoning
        """
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            current_price = data.get('current_price', 0)
            bull_case = data.get('bull_case', {})
            bear_case = data.get('bear_case', {})

            # Format the debate for prompt
            debate_summary = self._format_debate(bull_case, bear_case)

            # Create user prompt
            user_prompt = f"""You are evaluating the investment case for {symbol} at ${current_price:.2f}.

Below are the BULL and BEAR research arguments. Your task is to:

1. Summarize the key points from BOTH sides objectively
2. Evaluate the quality and strength of each argument
3. Assess the risk/reward ratio
4. Make a CLEAR final recommendation: BUY, HOLD, or SELL
5. Provide conviction level (1-10)
6. Explain your reasoning thoroughly

{debate_summary}

Consider:
- Which side has stronger evidence?
- What is the risk/reward profile?
- Are there any critical factors being overlooked?
- What is the probability-weighted expected outcome?

Provide a balanced, objective assessment and a CLEAR decision.
"""

            # Get judgment from LLM
            logger.info(f"Synthesizing investment debate for {symbol}")
            analysis = self._call_llm(
                system_prompt=RESEARCH_JUDGE_SYSTEM,
                user_prompt=user_prompt
            )

            # Extract recommendation and conviction
            recommendation = self._extract_recommendation(analysis)
            conviction = self._extract_conviction(analysis)
            price_target = self._extract_price_target(analysis, current_price, recommendation)

            # Create report
            report = self._create_report(
                analysis=analysis,
                confidence=conviction / 10.0,
                metadata={
                    'symbol': symbol,
                    'current_price': float(current_price),
                    'recommendation': recommendation,
                    'conviction_rating': conviction,
                    'price_target': price_target,
                    'bull_conviction': bull_case.get('metadata', {}).get('conviction_rating', 0),
                    'bear_conviction': bear_case.get('metadata', {}).get('conviction_rating', 0),
                    'analysis_type': 'investment_decision'
                }
            )

            logger.info(f"✅ Research judgment complete for {symbol}: {recommendation} (Conviction: {conviction}/10)")
            return report

        except Exception as e:
            logger.error(f"Research judgment failed: {e}")
            raise

    def _format_debate(self, bull_case: Dict[str, Any], bear_case: Dict[str, Any]) -> str:
        """Format bull and bear arguments for the prompt."""
        lines = []

        # Bull Case
        lines.append("="*60)
        lines.append("BULL CASE ARGUMENT")
        lines.append("="*60)
        if bull_case:
            metadata = bull_case.get('metadata', {})
            lines.append(f"Conviction: {metadata.get('conviction_rating', 'Unknown')}/10")
            lines.append(f"Price Target: ${metadata.get('price_target', 'Unknown')}")
            lines.append(f"\n{bull_case.get('analysis', 'No bull case provided')}")
        else:
            lines.append("No bull case provided")

        lines.append("\n")

        # Bear Case
        lines.append("="*60)
        lines.append("BEAR CASE ARGUMENT")
        lines.append("="*60)
        if bear_case:
            metadata = bear_case.get('metadata', {})
            lines.append(f"Conviction: {metadata.get('conviction_rating', 'Unknown')}/10")
            lines.append(f"Downside Target: ${metadata.get('price_target', 'Unknown')}")
            lines.append(f"\n{bear_case.get('analysis', 'No bear case provided')}")
        else:
            lines.append("No bear case provided")

        return "\n".join(lines)

    def _extract_recommendation(self, analysis: str) -> str:
        """Extract BUY/HOLD/SELL recommendation from analysis."""
        analysis_lower = analysis.lower()

        # Look for clear recommendation statements
        if 'recommendation: buy' in analysis_lower or 'final recommendation: buy' in analysis_lower:
            return 'BUY'
        elif 'recommendation: sell' in analysis_lower or 'final recommendation: sell' in analysis_lower:
            return 'SELL'
        elif 'recommendation: hold' in analysis_lower or 'final recommendation: hold' in analysis_lower:
            return 'HOLD'

        # Check for keywords in context
        buy_keywords = ['recommend buying', 'should buy', 'strong buy', 'accumulate']
        sell_keywords = ['recommend selling', 'should sell', 'avoid', 'reduce position']
        hold_keywords = ['recommend holding', 'should hold', 'wait', 'neutral']

        # Count keyword occurrences
        buy_count = sum(1 for kw in buy_keywords if kw in analysis_lower)
        sell_count = sum(1 for kw in sell_keywords if kw in analysis_lower)
        hold_count = sum(1 for kw in hold_keywords if kw in analysis_lower)

        if buy_count > sell_count and buy_count > hold_count:
            return 'BUY'
        elif sell_count > buy_count and sell_count > hold_count:
            return 'SELL'
        elif hold_count > 0:
            return 'HOLD'

        # Default to HOLD if unclear
        logger.warning("Could not clearly extract recommendation, defaulting to HOLD")
        return 'HOLD'

    def _extract_conviction(self, analysis: str) -> int:
        """Extract conviction rating from analysis (1-10)."""
        import re

        patterns = [
            r'conviction[:\s]+(\d+)(?:/10)?',
            r'conviction level[:\s]+(\d+)(?:/10)?',
            r'conviction rating[:\s]+(\d+)(?:/10)?'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                rating = int(match.group(1))
                return min(max(rating, 1), 10)

        # Default to moderate conviction
        return 6

    def _extract_price_target(self, analysis: str, current_price: float, recommendation: str) -> float:
        """Extract price target from analysis."""
        import re

        patterns = [
            r'price target[:\s]+\$?(\d+\.?\d*)',
            r'target[:\s]+\$(\d+\.?\d*)',
            r'\$(\d+\.?\d*)\s+(?:price\s+)?target'
        ]

        for pattern in patterns:
            match = re.search(pattern, analysis.lower())
            if match:
                target = float(match.group(1))
                if 0.5 * current_price <= target <= 2 * current_price:
                    return target

        # Default targets based on recommendation
        if recommendation == 'BUY':
            return current_price * 1.10  # 10% upside
        elif recommendation == 'SELL':
            return current_price * 0.90  # 10% downside
        else:  # HOLD
            return current_price


if __name__ == "__main__":
    # Test the Research Judge
    logging.basicConfig(level=logging.INFO)

    print("\n=== Testing Research Judge ===")

    # Mock bull and bear cases
    mock_bull = {
        'agent': 'Bull Researcher',
        'analysis': """Bull Thesis:
1. Strong technical momentum with price above 50-day MA
2. Positive earnings surprise and new product launches
3. Market leader with strong competitive moats
4. Growing ecosystem driving recurring revenue

Conviction: 8/10
Price Target: $250 (11% upside)""",
        'metadata': {
            'conviction_rating': 8,
            'price_target': 250.0
        }
    }

    mock_bear = {
        'agent': 'Bear Researcher',
        'analysis': """Bear Thesis:
1. RSI at 69 approaching overbought territory
2. High P/E ratio of 28 suggests overvaluation
3. Price below 200-day MA shows long-term weakness
4. Competition intensifying in key markets

Conviction: 6/10
Downside Target: $210 (7% downside)""",
        'metadata': {
            'conviction_rating': 6,
            'price_target': 210.0
        }
    }

    # Create judge
    judge = ResearchJudge()

    # Run analysis
    print(f"\nSynthesizing debate...")
    result = judge.analyze({
        'symbol': 'AAPL',
        'current_price': 225.0,
        'bull_case': mock_bull,
        'bear_case': mock_bear
    })

    # Display results
    print(f"\n{'='*60}")
    print(f"Agent: {result['agent']}")
    print(f"Model: {result['model']}")
    print(f"\nFINAL DECISION:")
    print(f"Recommendation: {result['metadata']['recommendation']}")
    print(f"Conviction: {result['metadata']['conviction_rating']}/10")
    print(f"Price Target: ${result['metadata']['price_target']:.2f}")
    print(f"\nBull Conviction: {result['metadata']['bull_conviction']}/10")
    print(f"Bear Conviction: {result['metadata']['bear_conviction']}/10")
    print(f"\n{'='*60}")
    print("REASONING:")
    print(f"{'='*60}")
    print(result['analysis'])
    print(f"{'='*60}")
