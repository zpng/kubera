"""
Deep Researcher Agent
Synthesizes all collected data and makes detailed BUY/HOLD/SELL decisions
Model: deepseek/deepseek-r1-distill-llama-70b (reliable reasoning without hallucinations)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent
from ..config import AGENT_MODELS

logger = logging.getLogger(__name__)


class DeepResearcherAgent(BaseAgent):
    """
    Agent responsible for deep research and final investment decisions
    Uses all data from previous agents to make comprehensive BUY/HOLD/SELL recommendations

    Uses openrouter/polaris-alpha for advanced reasoning and complex analysis
    """

    def __init__(self):
        super().__init__(
            name="DeepResearcher",
            model=AGENT_MODELS["deep_researcher"],  # openrouter/polaris-alpha for advanced reasoning
            role="Senior investment researcher and decision maker",
            temperature=0.4,  # Balanced for detailed output
            max_tokens=8000  # More tokens for comprehensive analysis
        )
        # Fallback model if primary model fails
        self.fallback_model = "deepseek/deepseek-chat-v3.1"
        self.using_fallback = False

    def reset_fallback_state(self):
        """Reset fallback state to allow retrying primary model for next symbol"""
        self.using_fallback = False
        logger.debug("Reset fallback state - will try primary model first")

    def create_research_prompt(self, symbol: str, all_data: Dict[str, Any]) -> str:
        """Create comprehensive research prompt with all collected data"""
        
        # Extract data from all sources
        portfolio_pos = all_data.get("portfolio_position", {})
        historical = all_data.get("historical_data", {})
        news = all_data.get("news_data", {})
        company = all_data.get("company_data", {})
        twitter_sent = all_data.get("twitter_sentiment", {})
        reddit_sent = all_data.get("reddit_sentiment", {})
        risk = all_data.get("risk_metrics", {})
        
        # Get actual values with fallbacks
        current_price = historical.get('metrics', {}).get('current_price') or portfolio_pos.get('current_price', 0)
        avg_cost = portfolio_pos.get('avg_cost', 0)
        shares = portfolio_pos.get('shares', 0)
        
        # Format revenue growth properly (yfinance returns as decimal)
        revenue_growth_raw = company.get('financial_metrics', {}).get('revenue_growth')
        if revenue_growth_raw is not None and revenue_growth_raw != 'N/A':
            if abs(revenue_growth_raw) > 10:  # Already a percentage
                revenue_growth_str = f"{revenue_growth_raw:.1f}%"
            else:  # It's a decimal, convert to percentage
                revenue_growth_str = f"{revenue_growth_raw * 100:.1f}%"
        else:
            revenue_growth_str = "N/A"
        
        prompt = f"""# Investment Analysis Task for {symbol}

## CRITICAL INSTRUCTIONS - READ CAREFULLY

**ABSOLUTE RULES - BREAKING THESE WILL CAUSE ANALYSIS REJECTION:**

1. **USE ONLY PROVIDED DATA**: Every single number in your analysis MUST come from the data below. DO NOT calculate, estimate, or invent ANY numbers.

2. **EXACT NUMBER MATCHING**:
   - Current Price: ${current_price} ← USE THIS EXACT NUMBER
   - YTD Return: {historical.get('metrics', {}).get('ytd_return_pct', 'N/A')}% ← COPY THIS EXACTLY
   - P/E Ratio: {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')} ← COPY THIS EXACTLY (NOT Forward P/E!)
   - Forward P/E: {company.get('financial_metrics', {}).get('forward_pe', 'N/A')} ← THIS IS DIFFERENT FROM P/E!
   - Revenue Growth: {revenue_growth_str} ← USE THIS EXACT STRING

3. **POSITION CALCULATIONS**:
   - DO NOT recalculate position values
   - Shares: {shares} × Current Price ${current_price} = ${shares * current_price:.2f}
   - Unrealized P/L: ${(current_price - avg_cost) * shares:.2f} ({((current_price/avg_cost - 1) * 100):.1f}%)
   - USE THESE EXACT CALCULATED VALUES

4. **IF DATA IS "N/A"**: Write "N/A" or "Data unavailable" - DO NOT make up a number

5. **TARGET PRICE RULE**: Must be within 50% of current price ${current_price} (range: ${current_price * 0.5:.2f} to ${current_price * 1.5:.2f})

## PROVIDED DATA FOR {symbol}

### 1. PORTFOLIO POSITION (ACTUAL HOLDINGS)
- Shares Owned: {shares}
- Average Cost Basis: ${avg_cost}
- Current Price: ${current_price}
- Position Value: ${shares * current_price:.2f}
- Unrealized P/L: ${(current_price - avg_cost) * shares:.2f} ({((current_price/avg_cost - 1) * 100):.1f}%)

### 2. HISTORICAL PRICE DATA (FROM YFINANCE) - COPY EXACTLY AS SHOWN

**CRITICAL: Each line below has a LABEL (left side) and a VALUE (right side). You MUST use the EXACT VALUE shown after the colon, paired with its correct LABEL.**

| METRIC LABEL | EXACT VALUE TO USE |
|--------------|-------------------|
| Year-to-Date Return | {historical.get('metrics', {}).get('ytd_return_pct', 'N/A')}% |
| 6-Month Return | {historical.get('metrics', {}).get('period_return_pct', 'N/A')}% |
| Volatility | {historical.get('metrics', {}).get('volatility', 'N/A')}% |
| Price Trend | {historical.get('trends', {}).get('trend', 'N/A')} |
| 20-Day SMA | ${historical.get('trends', {}).get('sma_20', 'N/A')} |
| 30-Day Momentum | {historical.get('trends', {}).get('momentum_30d_pct', 'N/A')}% |
| 52-Week Low | ${historical.get('metrics', {}).get('fifty_two_week_low', 'N/A')} |
| 52-Week High | ${historical.get('metrics', {}).get('fifty_two_week_high', 'N/A')} |

**EXAMPLE OF WRONG (DO NOT DO THIS):**
"The 52-Week Low is $41,647M" ← WRONG! That's the cash position, not 52-week low!
"The P/E Ratio is 1.1%" ← WRONG! That's the P/L percentage, not P/E ratio!

**EXAMPLE OF CORRECT:**
"Year-to-Date Return: {historical.get('metrics', {}).get('ytd_return_pct', 'N/A')}%" ← RIGHT! Exact copy from table above.

### 3. COMPANY FUNDAMENTALS (FROM YFINANCE)

**CRITICAL: The table below shows EXACT values for each metric. DO NOT confuse P/E with Forward P/E, or any other metrics. Each row has a LABEL and a VALUE - use them together.**

#### Basic Company Info:
- Company Name: {company.get('company_info', {}).get('name', symbol)}
- Sector: {company.get('company_info', {}).get('sector', 'N/A')}
- Market Cap: ${company.get('company_info', {}).get('market_cap', 'N/A')}

#### Financial Metrics - COPY EXACT VALUES FROM TABLE:

| METRIC LABEL | EXACT VALUE TO USE | CRITICAL NOTES |
|--------------|-------------------|----------------|
| P/E Ratio (Trailing) | {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')} | This is DIFFERENT from Forward P/E! |
| Forward P/E | {company.get('financial_metrics', {}).get('forward_pe', 'N/A')} | This is DIFFERENT from P/E Ratio! |
| Price to Book | {company.get('financial_metrics', {}).get('price_to_book', 'N/A')} | |
| Profit Margin | {company.get('financial_metrics', {}).get('profit_margin', 'N/A')}% | This is a percentage |
| Revenue Growth | {revenue_growth_str} | Use EXACT string shown |
| ROE (Return on Equity) | {company.get('financial_metrics', {}).get('roe', 'N/A')}% | This is a percentage |
| Debt to Equity | {company.get('financial_metrics', {}).get('debt_to_equity', 'N/A')} | Use THIS exact number |
| Beta | {company.get('financial_metrics', {}).get('beta', 'N/A')} | Volatility measure |

**EXAMPLES OF WRONG (ACTUAL ERRORS FROM PREVIOUS RUNS - DO NOT REPEAT):**
- "P/E Ratio: 132.57" ← WRONG! 132.57 is the Forward P/E, not P/E Ratio!
- "P/E Ratio: 1.1" ← WRONG! 1.1% is the P/L percentage, not P/E Ratio!
- "Debt to Equity: 17.08" when actual is 0.17 ← WRONG! Off by 100x!
- "52-Week Low: $41,647M" ← WRONG! That's Total Cash, not 52-week low!

**EXAMPLES OF CORRECT:**
- "P/E Ratio: {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')}" ← RIGHT! From table above
- "Forward P/E: {company.get('financial_metrics', {}).get('forward_pe', 'N/A')}" ← RIGHT! Different metric!
- "Debt to Equity: {company.get('financial_metrics', {}).get('debt_to_equity', 'N/A')}" ← RIGHT! From table above

#### Cash Position - COPY EXACT VALUES FROM TABLE:

| METRIC LABEL | EXACT VALUE TO USE | CRITICAL NOTES |
|--------------|-------------------|----------------|
| Total Cash (USD) | ${company.get('financial_metrics', {}).get('total_cash', 'N/A')} | Full amount in dollars |
| Total Cash (Millions) | ${company.get('financial_metrics', {}).get('total_cash_millions', 'N/A')}M | Abbreviated form |
| Cash per Share | ${company.get('financial_metrics', {}).get('cash_per_share', 'N/A')} | Per share basis |

**WARNING: Do NOT confuse Total Cash with 52-Week Low or any price metric!**

#### Analyst Consensus Data - COPY EXACT VALUES FROM TABLE:

| METRIC LABEL | EXACT VALUE TO USE |
|--------------|-------------------|
| Analyst Recommendation | {company.get('financial_metrics', {}).get('recommendation', 'N/A')} |
| Analyst Target Price | ${company.get('financial_metrics', {}).get('target_price', 'N/A')} |
| Number of Analysts Covering | {company.get('financial_metrics', {}).get('number_of_analysts', 'N/A')} |

### 4. NEWS SENTIMENT
- Articles Found: {news.get('sentiment_analysis', {}).get('article_count', 0)}
- Overall Sentiment: {news.get('sentiment_analysis', {}).get('dominant_sentiment', 'No news data available')}
- Key Events: {', '.join(news.get('key_events', [])) or 'None reported'}

### 5. SOCIAL SENTIMENT
- Twitter Mentions: {twitter_sent.get('sentiment_analysis', {}).get('tweet_count', 0)}
- Reddit Posts: {reddit_sent.get('sentiment_analysis', {}).get('post_count', 0)}
- Social Sentiment: Simulated as neutral (actual API not connected)

### 6. RISK ASSESSMENT
- Position Risk Score: {risk.get('risk_score', 'N/A')}/10
- Risk Level: {risk.get('risk_level', 'N/A')}
- Value at Risk (1-day): ${risk.get('value_at_risk_1day', 'N/A')}

## YOUR ANALYSIS REQUIREMENTS

You MUST provide a comprehensive investment analysis with the following structure:

### FORMAT YOUR RESPONSE EXACTLY AS:

**DECISION**: [BUY MORE / HOLD / SELL / TRIM POSITION]

**CONVICTION**: [1-10] (where 1 = very uncertain, 10 = very confident)

**TARGET PRICE**: $[price] (MUST be based on current price ${current_price}, reasonable range is ${current_price * 0.7:.2f} to ${current_price * 1.5:.2f})

**DETAILED ANALYSIS**:

**1. CURRENT POSITION REVIEW**

ALL DATA IS PRE-FILLED BELOW - DO NOT REPEAT ANY NUMBERS IN YOUR ASSESSMENT:

| Position Metric | Exact Value |
|----------------|-------------|
| Shares Owned | {shares} |
| Average Cost per Share | ${avg_cost} |
| Current Price per Share | ${current_price} |
| Total Position Value | ${shares * current_price:.2f} |
| Total Cost Basis | ${shares * avg_cost:.2f} |
| Unrealized Gain/Loss (USD) | ${(current_price - avg_cost) * shares:.2f} |
| Unrealized Return (%) | {((current_price/avg_cost - 1) * 100):.1f}% |
| Position Status | {"PROFITABLE" if current_price > avg_cost else "LOSS"} |

**Your Position Assessment** (Write interpretation ONLY - do NOT write any numbers):
[Assess whether this is a winning or losing trade, and discuss position size relative to risk. Use phrases like "strong gain", "modest profit", "significant loss" instead of repeating the percentages above.]

**2. TECHNICAL ANALYSIS**

The following data is PRE-FILLED - do NOT modify these numbers, ONLY write your analysis below:

- Current Price: ${current_price}
- Year-to-Date Return: {historical.get('metrics', {}).get('ytd_return_pct', 'N/A')}%
- 6-Month Return: {historical.get('metrics', {}).get('period_return_pct', 'N/A')}%
- Price Trend: {historical.get('trends', {}).get('trend', 'Unknown')}
- 52-Week Range: ${historical.get('metrics', {}).get('fifty_two_week_low', 'N/A')} to ${historical.get('metrics', {}).get('fifty_two_week_high', 'N/A')}
- Position in Range: {((current_price - historical.get('metrics', {}).get('fifty_two_week_low', 0)) / (historical.get('metrics', {}).get('fifty_two_week_high', 1) - historical.get('metrics', {}).get('fifty_two_week_low', 0)) * 100):.0f}% of 52-week range

**Your Technical Analysis** (Write your interpretation WITHOUT repeating the numbers above):
[Analyze the technical position based on the data shown above]

**3. FUNDAMENTAL STRENGTH** (MANDATORY: COPY THESE EXACT NUMBERS FROM SECTION 3 TABLE)

**YOU MUST COPY-PASTE FROM THE TABLE IN SECTION 3 ABOVE. DO NOT TYPE NUMBERS FROM MEMORY.**

| Metric | Exact Value from Section 3 Table | Your Assessment |
|--------|----------------------------------|-----------------|
| P/E Ratio (Trailing) | {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')} | [Is this high/low vs industry?] |
| Forward P/E | {company.get('financial_metrics', {}).get('forward_pe', 'N/A')} | [Expected valuation change] |
| Revenue Growth | {revenue_growth_str} | [Strong/weak growth?] |
| Profit Margin | {company.get('financial_metrics', {}).get('profit_margin', 'N/A')}% | [Profitable or not?] |
| Debt/Equity Ratio | {company.get('financial_metrics', {}).get('debt_to_equity', 'N/A')} | [Financial health?] |
| Total Cash | ${company.get('financial_metrics', {}).get('total_cash_millions', 'N/A')}M | [Runway if unprofitable?] |

**CRITICAL VALIDATION:**
- Is your P/E Ratio value the SAME as in Section 3 Table? YES/NO: ____
- Is your Forward P/E value DIFFERENT from P/E Ratio? YES/NO: ____
- Did you confuse Total Cash with any price metric? YES/NO: ____

**Overall Fundamental Assessment**: [Your interpretation - reference ONLY exact numbers shown in table above]

**4. ANALYST CONSENSUS** (Use provided data)
- Recommendation: {company.get('financial_metrics', {}).get('recommendation', 'N/A')}
- Target Price: ${company.get('financial_metrics', {}).get('target_price', 'N/A')}
- Current vs Target: [Calculate % difference]

**5. NEWS & SENTIMENT**
- News Articles: {news.get('sentiment_analysis', {}).get('article_count', 0)}
- Sentiment: {news.get('sentiment_analysis', {}).get('dominant_sentiment', 'None')}
- Impact: [Positive, negative, or neutral catalysts?]

**6. RISK ASSESSMENT**
- Position Risk: {risk.get('risk_score', 'N/A')}/10
- Risk Level: {risk.get('risk_level', 'N/A')}
- Concern: [What are the main risks?]

**7. INVESTMENT DECISION RATIONALE**

CRITICAL REMINDER - ALL DATA IS ALREADY SHOWN IN SECTIONS 1-6 ABOVE:
- YTD Return: {historical.get('metrics', {}).get('ytd_return_pct', 'N/A')}%
- Current P/L: {((current_price/avg_cost - 1) * 100):.1f}%
- Revenue Growth: {revenue_growth_str}
- P/E Ratio: {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')}
- Forward P/E: {company.get('financial_metrics', {}).get('forward_pe', 'N/A')}
- Debt/Equity: {company.get('financial_metrics', {}).get('debt_to_equity', 'N/A')}
- Total Cash: ${company.get('financial_metrics', {}).get('total_cash_millions', 'N/A')}M
- Analyst Target: ${company.get('financial_metrics', {}).get('target_price', 'N/A')}

Write your rationale below. DO NOT repeat or substitute these numbers - they are already shown above.

**WHY [DECISION]**:
[Explain your decision in 4-5 sentences. Reference the data shown above by name only (e.g., "given the YTD return" not "given the 13% YTD return")]

**KEY FACTORS**:
[List the 3 most important factors from the data above]

**POSITION SIZING**:
[Recommendation on position size given the current P/L shown above]

**TIMEFRAME**:
[Short-term (0-3m) vs Long-term (6-12m) outlook]

**8. ACTION ITEMS**

Current Price: ${current_price}
Your Target Price: $[specify your target]

- **Immediate Action**: [What to do now with this position]
- **Watch For**: [What metrics/events to monitor - reference metrics by name only, not values]
- **Price Target Rationale**: [Why you chose your target price]

CRITICAL: Write AT LEAST 500 words total across all sections. DO NOT write any numbers - all data is already provided in sections 1-7 above!
"""
        
        return prompt
    
    def _format_list(self, items: List[str], limit: int = 5) -> str:
        """Format list items for prompt"""
        if not items:
            return "- None"
        return "\n".join(f"- {item}" for item in items[:limit])

    def _check_for_hallucinations(self, output_text: str, all_data: Dict[str, Any]) -> List[str]:
        """
        Check if LLM output contains hallucinated data by comparing mentioned values
        against actual provided data

        Returns:
            List of warning messages for potential hallucinations
        """
        warnings = []
        historical = all_data.get("historical_data", {})
        company = all_data.get("company_data", {})

        try:
            import re

            # Check YTD return mentions
            ytd_mentions = re.findall(r'YTD.*?(\d+\.?\d*)%', output_text, re.IGNORECASE)
            actual_ytd = historical.get('metrics', {}).get('ytd_return_pct')
            if ytd_mentions and actual_ytd is not None and actual_ytd != 'N/A':
                for mentioned_ytd in ytd_mentions:
                    mentioned_value = float(mentioned_ytd)
                    # Allow 5% tolerance for rounding
                    if abs(mentioned_value - actual_ytd) > 5:
                        warnings.append(f"YTD return mismatch: mentioned {mentioned_value}% vs actual {actual_ytd}%")

            # Check debt-to-equity mentions
            de_mentions = re.findall(r'debt[- ]to[- ]equity.*?(\d+\.?\d*)', output_text, re.IGNORECASE)
            actual_de = company.get('financial_metrics', {}).get('debt_to_equity')
            if de_mentions and actual_de is not None and actual_de != 'N/A':
                for mentioned_de in de_mentions:
                    mentioned_value = float(mentioned_de)
                    # Allow 20% tolerance
                    if abs(mentioned_value - actual_de) / max(actual_de, 0.01) > 0.2:
                        warnings.append(f"D/E ratio mismatch: mentioned {mentioned_value} vs actual {actual_de}")

            # Check P/E ratio mentions (distinguish from Forward P/E)
            pe_mentions = re.findall(r'P/E.*?(\d+\.?\d*)', output_text, re.IGNORECASE)
            actual_pe = company.get('financial_metrics', {}).get('pe_ratio')
            actual_forward_pe = company.get('financial_metrics', {}).get('forward_pe')
            
            if pe_mentions and actual_pe is not None and actual_pe != 'N/A':
                for mentioned_pe in pe_mentions:
                    mentioned_value = float(mentioned_pe)
                    
                    # Check if they confused P/E with Forward P/E
                    if actual_forward_pe and actual_forward_pe != 'N/A':
                        forward_val = float(actual_forward_pe) if isinstance(actual_forward_pe, (int, float, str)) else 0
                        if forward_val > 0 and abs(mentioned_value - forward_val) < 2:
                            warnings.append(f"CRITICAL: Confused P/E with Forward P/E - mentioned {mentioned_value} which is Forward P/E, not P/E ({actual_pe})")
                            continue
                    
                    # Regular P/E check with 20% tolerance
                    if abs(mentioned_value - actual_pe) / max(actual_pe, 0.01) > 0.2:
                        warnings.append(f"P/E ratio mismatch: mentioned {mentioned_value} vs actual {actual_pe}")

        except Exception as e:
            logger.error(f"Error checking for hallucinations: {e}")

        return warnings

    def _detect_numbers_in_prose(self, output_text: str) -> List[str]:
        """
        Detect if LLM wrote numbers in analysis prose (which violates the no-numbers rule)

        Returns:
            List of warnings for numbers found in prose sections
        """
        warnings = []

        try:
            import re

            # Extract analysis sections (prose after "Your ... Assessment" or "Your ... Analysis")
            analysis_sections = re.findall(
                r'\*\*Your (?:Position Assessment|Technical Analysis|Overall Fundamental Assessment)\*\*.*?(?=\n\*\*|\Z)',
                output_text,
                re.DOTALL | re.IGNORECASE
            )

            for section in analysis_sections:
                # Look for monetary amounts: $123.45, $1,234, $41,647M
                money_matches = re.findall(r'\$[\d,]+(?:\.\d+)?[MBK]?', section)
                if money_matches:
                    warnings.append(f"CRITICAL: Found dollar amounts in prose: {', '.join(money_matches[:3])}")

                # Look for percentages: 13.25%, 71.8%
                percent_matches = re.findall(r'\d+\.?\d*%', section)
                if percent_matches:
                    warnings.append(f"CRITICAL: Found percentages in prose: {', '.join(percent_matches[:3])}")

                # Look for numeric values that might be metrics: P/E of 294, ratio of 17.08
                metric_matches = re.findall(r'(?:P/E|ratio|price|return|growth|margin|cash).*?(\d+\.?\d+)', section, re.IGNORECASE)
                if metric_matches:
                    warnings.append(f"CRITICAL: Found metric values in prose: {', '.join(metric_matches[:3])}")

        except Exception as e:
            logger.error(f"Error detecting numbers in prose: {e}")

        return warnings

    def validate_and_parse_output(self, output: str, symbol: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and validate the LLM output"""
        try:
            # Extract actual current price for validation
            portfolio_pos = all_data.get("portfolio_position", {})
            historical = all_data.get("historical_data", {})
            company = all_data.get("company_data", {})
            actual_current_price = historical.get('metrics', {}).get('current_price') or portfolio_pos.get('current_price', 0)

            # Default values
            decision = "HOLD"
            conviction = 5
            target_price = actual_current_price * 1.1  # Default to 10% upside

            # Extract decision
            output_upper = output.upper()
            if "BUY MORE" in output_upper or "STRONG BUY" in output_upper:
                decision = "BUY MORE"
            elif "TRIM" in output_upper:
                decision = "TRIM POSITION"
            elif "SELL" in output_upper and "DON'T SELL" not in output_upper and "NOT SELL" not in output_upper:
                decision = "SELL"
            else:
                decision = "HOLD"

            # Extract conviction (1-10) - try multiple patterns
            import re
            conviction = None

            # Try pattern 1: **CONVICTION**: 7
            conviction_match = re.search(r'\*\*CONVICTION\*\*[:\s]+(\d+)', output, re.IGNORECASE)
            if conviction_match:
                conviction = min(10, max(1, int(conviction_match.group(1))))
            else:
                # Try pattern 2: CONVICTION: 7 (without bold)
                conviction_match = re.search(r'CONVICTION[:\s]+(\d+)', output, re.IGNORECASE)
                if conviction_match:
                    conviction = min(10, max(1, int(conviction_match.group(1))))
                else:
                    # Try pattern 3: look for "conviction" followed by digit anywhere in first 500 chars
                    conviction_match = re.search(r'conviction[:\s]+(\d+)/10', output[:500], re.IGNORECASE)
                    if conviction_match:
                        conviction = min(10, max(1, int(conviction_match.group(1))))

            if conviction is None:
                # If no conviction found, log warning and extract from anywhere
                logger.warning(f"No conviction score found in standard format for {symbol}, searching entire output...")
                # Last resort: find any mention of "conviction" with a number
                conviction_match = re.search(r'conviction[:\s]*[=]?\s*(\d+)', output, re.IGNORECASE)
                if conviction_match:
                    conviction = min(10, max(1, int(conviction_match.group(1))))
                    logger.info(f"Found conviction {conviction} for {symbol} using fallback pattern")
                else:
                    conviction = 5
                    logger.warning(f"No conviction score found anywhere for {symbol}, using default: 5")

            # Extract target price - try multiple patterns
            target_price = actual_current_price * 1.1  # Default fallback

            # Try pattern 1: **TARGET PRICE**: $420.00
            target_match = re.search(r'\*\*TARGET PRICE\*\*[:\s]+\$?([\d,]+(?:\.\d{2})?)', output, re.IGNORECASE)
            if not target_match:
                # Try pattern 2: TARGET PRICE: $420.00 (without bold)
                target_match = re.search(r'TARGET PRICE[:\s]+\$?([\d,]+(?:\.\d{2})?)', output, re.IGNORECASE)
            if not target_match:
                # Try pattern 3: Target: $420.00
                target_match = re.search(r'Target[:\s]+\$?([\d,]+(?:\.\d{2})?)', output[:1000], re.IGNORECASE)

            if target_match:
                extracted_target = float(target_match.group(1).replace(',', ''))
                # Validate target price is reasonable (within 50% up/down from current)
                if actual_current_price > 0:
                    if 0.5 * actual_current_price <= extracted_target <= 1.5 * actual_current_price:
                        target_price = extracted_target
                        logger.info(f"Extracted target price ${target_price} for {symbol}")
                    else:
                        # Target is unreasonable, log and use conservative target
                        logger.warning(f"Unreasonable target price ${extracted_target} for {symbol} (current: ${actual_current_price}), using conservative target")
                        if decision == "BUY MORE":
                            target_price = actual_current_price * 1.15
                        elif decision == "SELL":
                            target_price = actual_current_price * 0.9
                        else:
                            target_price = actual_current_price * 1.05
            else:
                # No target price found, use analyst target or default
                analyst_target = company.get('financial_metrics', {}).get('target_price')
                if analyst_target and analyst_target != "N/A":
                    target_price = float(analyst_target)
                    logger.info(f"No target in output, using analyst target price ${target_price} for {symbol}")
                else:
                    logger.warning(f"No target price found for {symbol}, using default based on decision")
                    if decision == "BUY MORE":
                        target_price = actual_current_price * 1.15
                    elif decision == "SELL":
                        target_price = actual_current_price * 0.9
                    else:
                        target_price = actual_current_price * 1.05
            
            # Clean up and validate the rationale
            rationale = output.strip()

            # Validate for hallucinations - check if LLM used correct data
            hallucination_warnings = self._check_for_hallucinations(rationale, all_data)

            # Check for numbers in prose (violates no-numbers rule)
            prose_number_warnings = self._detect_numbers_in_prose(rationale)

            # Combine all warnings
            all_warnings = hallucination_warnings + prose_number_warnings

            if all_warnings:
                logger.warning(f"Analysis issues detected for {symbol}: {', '.join(all_warnings)}")
                # Add warning to rationale
                if prose_number_warnings:
                    rationale += "\n\n⚠️ WARNING: LLM wrote numbers in prose sections (violates no-numbers rule)."
                if hallucination_warnings:
                    rationale += "\n\n⚠️ Note: Some data discrepancies detected during analysis validation."

            # Check if rationale is too short (likely an error)
            if len(rationale) < 200:
                logger.warning(f"Rationale too short for {symbol} ({len(rationale)} chars), adding context")
                # Add data summary as fallback
                portfolio_pos = all_data.get("portfolio_position", {})
                historical = all_data.get("historical_data", {})
                company = all_data.get("company_data", {})
                
                rationale = f"""Analysis for {symbol}:

**Position**: {portfolio_pos.get('shares', 0)} shares at ${portfolio_pos.get('avg_cost', 0)} (currently ${actual_current_price})
**P/L**: {((actual_current_price/portfolio_pos.get('avg_cost', 1) - 1) * 100):.1f}%

**Technical**: 
- 6M Return: {historical.get('metrics', {}).get('period_return_pct', 'N/A')}%
- Trend: {historical.get('trends', {}).get('trend', 'N/A')}
- 52W Range: ${historical.get('metrics', {}).get('min_price', 'N/A')} - ${historical.get('metrics', {}).get('max_price', 'N/A')}

**Fundamentals**:
- P/E: {company.get('financial_metrics', {}).get('pe_ratio', 'N/A')}
- Revenue Growth: {company.get('financial_metrics', {}).get('revenue_growth', 'N/A')}
- Profit Margin: {company.get('financial_metrics', {}).get('profit_margin', 'N/A')}%

**Analyst View**: {company.get('financial_metrics', {}).get('recommendation', 'N/A')} (Target: ${company.get('financial_metrics', {}).get('target_price', 'N/A')})

**Decision Basis**: {decision} based on current position performance and fundamental metrics. 
Target price of ${target_price} represents {((target_price/actual_current_price - 1) * 100):.1f}% 
{'upside' if target_price > actual_current_price else 'downside'} from current levels.

**Recommendation**: {'Take profits and reduce exposure' if decision in ['SELL', 'TRIM POSITION'] else 'Maintain current position' if decision == 'HOLD' else 'Consider adding to position'} given the current risk/reward profile."""
            
            return {
                "symbol": symbol,
                "decision": decision,
                "conviction": conviction,
                "target_price": round(target_price, 2),
                "rationale": rationale,
                "current_price": actual_current_price,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error parsing research output for {symbol}: {e}")
            # Return safe defaults
            return {
                "symbol": symbol,
                "decision": "HOLD",
                "conviction": 3,
                "target_price": all_data.get("portfolio_position", {}).get("current_price", 0) * 1.05,
                "rationale": f"Analysis error: {str(e)}. Recommend HOLD pending manual review.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def research_symbol(self, symbol: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform deep research on a single symbol"""
        logger.info(f"Conducting deep research for {symbol}...")
        
        # Create research prompt
        prompt = self.create_research_prompt(symbol, all_data)
        
        # Strict system message to prevent hallucinations
        system_msg = SystemMessage(content="""You are a senior investment analyst with expertise in equity research.

╔═══════════════════════════════════════════════════════════════╗
║  ⚠️  ABSOLUTE PROHIBITION - VIOLATING THIS WILL FAIL YOU  ⚠️   ║
║                                                               ║
║  🚫 NEVER WRITE NUMBERS IN YOUR ANALYSIS TEXT 🚫              ║
║                                                               ║
║  ❌ FORBIDDEN: "YTD return of 13.25%"                         ║
║  ❌ FORBIDDEN: "P/E ratio is 294"                             ║
║  ❌ FORBIDDEN: "Current price $429.52"                        ║
║  ❌ FORBIDDEN: "Total cash of $41,647M"                       ║
║  ❌ FORBIDDEN: ANY numeric values in prose                    ║
║                                                               ║
║  ✅ REQUIRED: "YTD return shows strong performance"          ║
║  ✅ REQUIRED: "Valuation multiple appears elevated"          ║
║  ✅ REQUIRED: "Strong balance sheet with ample liquidity"    ║
║  ✅ REQUIRED: Use descriptive words ONLY, NO numbers         ║
║                                                               ║
║  ALL NUMBERS ARE PRE-FILLED IN TABLES                        ║
║  YOUR ONLY JOB: Write qualitative interpretations           ║
╚═══════════════════════════════════════════════════════════════╝

YOUR TASK: Write interpretation and analysis following the template format.

ABSOLUTELY CRITICAL FORMAT REQUIREMENTS:
1. START your response with EXACTLY these three lines (with ** bold markers):
   **DECISION**: [your decision here]
   **CONVICTION**: [number 1-10 only]
   **TARGET PRICE**: $[exact number]

2. Then write "**DETAILED ANALYSIS**:" as a header

3. Then write all 8 sections with bold headers: **1. CURRENT POSITION REVIEW**, **2. TECHNICAL ANALYSIS**, etc.

MANDATORY DATA REQUIREMENTS - CRITICAL TO PREVENT FIELD CONFUSION:
1. **TABLE FORMAT**: The prompt contains DATA TABLES with LABEL | VALUE columns. You MUST:
   - Read BOTH the label AND its corresponding value from the SAME ROW
   - NEVER use a value from one row with a label from another row
   - EXAMPLE: If table shows "| P/E Ratio | 292.19 |", write "P/E Ratio: 292.19" NOT "P/E Ratio: 132.57" (that's Forward P/E!)

2. **COPY EXACTLY**: Use ONLY the specific numbers from the tables - copy them EXACTLY as shown
   - Do NOT calculate, modify, round, or invent any numbers
   - Do NOT use numbers from memory or previous analyses
   - If a value says "N/A", write "N/A" - do NOT make up a number

3. **DISTINGUISH SIMILAR METRICS**:
   - P/E Ratio ≠ Forward P/E (they are DIFFERENT metrics with DIFFERENT values)
   - 52-Week Low ≠ Total Cash (completely unrelated metrics)
   - YTD Return ≠ 6-Month Return (different time periods)
   - Profit Margin % ≠ P/E Ratio (one is %, one is a ratio)

4. **EXACT FORMAT MATCHING**: When referencing data, COPY the exact format from the prompt
   - If it says "13.25%" write "13.25%" not "13%" or "13.3%"
   - If it says "292.19" write "292.19" not "292" or "292.2"

5. **MINIMUM LENGTH**: Write AT LEAST 1000 words total across all sections

EXAMPLE OUTPUT FORMAT:
**DECISION**: HOLD

**CONVICTION**: 7

**TARGET PRICE**: $420.00

**DETAILED ANALYSIS**:

**1. CURRENT POSITION REVIEW**
[Your analysis using exact numbers from the prompt tables...]

**2. TECHNICAL ANALYSIS**
[Your analysis copying exact values from Section 2 table...]

**3. FUNDAMENTAL STRENGTH**
| Metric | Value | Assessment |
|--------|-------|------------|
| P/E Ratio | [EXACT value from Section 3 table] | [Your analysis] |
| Forward P/E | [DIFFERENT value from Section 3 table] | [Your analysis] |
...

... continue through section 8

CRITICAL WARNINGS:
- If you confuse P/E Ratio with Forward P/E, the analysis will be REJECTED
- If you use Total Cash as a price metric (like 52-week low), the analysis will be REJECTED
- If you make up ANY numbers not in the prompt tables, the analysis will be REJECTED
- You MUST complete the validation checkpoints in Section 3 (YES/NO questions)""")
        
        # User message
        user_msg = HumanMessage(content=prompt)
        
        try:
            # Invoke LLM for deep analysis with retry and fallback
            max_retries = 2
            response = None
            models_to_try = [self.model, self.fallback_model] if not self.using_fallback else [self.fallback_model]

            for model_idx, current_model in enumerate(models_to_try):
                if model_idx > 0:
                    logger.warning(f"🔄 Primary model failed, switching to fallback model: {current_model}")
                    self.using_fallback = True
                    # Temporarily switch model
                    original_model = self.model
                    self.model = current_model

                try:
                    for attempt in range(max_retries + 1):
                        model_label = "fallback" if model_idx > 0 else "primary"
                        logger.info(f"Sending {len(prompt)} char prompt to {current_model} ({model_label}) (attempt {attempt + 1}/{max_retries + 1})...")

                        response = self.invoke([system_msg, user_msg])

                        # Log response quality
                        response_length = len(response.content)
                        logger.info(f"Received {response_length} char response from {current_model}")

                        if response_length >= 500:
                            # Good response, break
                            logger.info(f"✅ Successfully received analysis from {current_model}")
                            break
                        elif attempt < max_retries:
                            # Empty response, retry
                            logger.warning(f"⚠️  LLM returned short response ({response_length} chars) for {symbol}, retrying...")
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                        else:
                            # Final attempt failed with this model
                            logger.error(f"❌ {current_model} returned short response after {max_retries + 1} attempts for {symbol}")
                            if model_idx == 0:  # If primary model failed, will try fallback
                                logger.info("Will attempt fallback model next...")
                                raise Exception(f"Primary model {current_model} failed after {max_retries + 1} attempts")

                    # If we got here and have a good response, break out of model loop
                    if response and len(response.content) >= 500:
                        break

                except Exception as e:
                    if model_idx == len(models_to_try) - 1:
                        # This was the last model to try
                        logger.error(f"❌ All models failed for {symbol}: {e}")
                        raise
                    else:
                        # Try next model (fallback)
                        logger.warning(f"⚠️  Model {current_model} failed: {e}")
                        continue
                finally:
                    # Restore original model if we switched
                    if model_idx > 0 and 'original_model' in locals():
                        self.model = original_model
            
            # Validate and parse output
            analysis = self.validate_and_parse_output(response.content, symbol, all_data)

            # Add model info to analysis
            model_used = self.fallback_model if self.using_fallback else self.model
            analysis['model_used'] = model_used
            analysis['used_fallback'] = self.using_fallback

            # Log analysis quality
            rationale_length = len(analysis.get('rationale', ''))
            model_label = f"{model_used} (fallback)" if self.using_fallback else model_used
            logger.info(f"Research complete for {symbol} using {model_label}: {analysis['decision']} (Conviction: {analysis['conviction']}/10, Analysis: {rationale_length} chars)")

            if rationale_length < 300:
                logger.warning(f"⚠️  Short analysis for {symbol} - may lack detail")

            return analysis
            
        except Exception as e:
            logger.error(f"Error during research for {symbol}: {e}")
            current_price = all_data.get("portfolio_position", {}).get("current_price", 0)
            return {
                "symbol": symbol,
                "decision": "HOLD",
                "conviction": 3,
                "target_price": current_price * 1.05 if current_price > 0 else 0,
                "rationale": f"Analysis error: {str(e)}. Unable to complete analysis.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def aggregate_portfolio_data(self, symbol: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate all data for a single symbol from the state"""
        portfolio_stocks = state.get("portfolio_data", {}).get("portfolio", {}).get("stocks", [])
        portfolio_pos = next((s for s in portfolio_stocks if s["symbol"] == symbol), {})
        
        return {
            "symbol": symbol,
            "portfolio_position": portfolio_pos,
            "historical_data": state.get("historical_data", {}).get(symbol, {}),
            "news_data": state.get("news_data", {}).get(symbol, {}),
            "company_data": state.get("company_data", {}).get(symbol, {}),
            "twitter_sentiment": state.get("twitter_sentiment", {}).get(symbol, {}),
            "reddit_sentiment": state.get("reddit_sentiment", {}).get(symbol, {}),
            "risk_metrics": next(
                (r for r in state.get("risk_assessment", {}).get("positions_risk", []) 
                 if r.get("symbol") == symbol),
                {}
            )
        }
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with all collected data
            
        Returns:
            Updated state with research recommendations
        """
        logger.info(f"[{self.name}] Starting deep research analysis...")
        
        symbols = state.get("stock_symbols", [])
        if not symbols:
            logger.error("No stock symbols found in state")
            return state
        
        research_results = {}

        for symbol in symbols:
            # Reset fallback state for each symbol to try primary model first
            self.reset_fallback_state()

            # Aggregate all data for this symbol
            all_data = self.aggregate_portfolio_data(symbol, state)

            # Perform deep research
            analysis = self.research_symbol(symbol, all_data)
            research_results[symbol] = analysis
        
        # Update state
        state["research_results"] = research_results
        state["research_timestamp"] = datetime.now().isoformat()
        
        logger.info(f"[{self.name}] Deep research complete for {len(research_results)} symbols")
        return state


# Test function
def test_deep_researcher_agent():
    """Test the DeepResearcherAgent"""
    print("Testing Deep Researcher Agent...")
    
    agent = DeepResearcherAgent()
    
    # Test with comprehensive sample state
    state = {
        "stock_symbols": ["RGTI"],
        "portfolio_data": {
            "portfolio": {
                "stocks": [
                    {"symbol": "RGTI", "shares": 41.934, "avg_cost": 47.69, "current_price": 34.60}
                ]
            }
        },
        "historical_data": {
            "RGTI": {
                "metrics": {
                    "current_price": 34.60,
                    "period_return_pct": 130.0,  # Up 130% in 2025
                    "volatility": 7.58,
                    "max_price": 58.15,
                    "min_price": 1.13  # Actual 52-week low
                },
                "trends": {
                    "trend": "bullish",
                    "sma_20": 42.02,
                    "momentum_30d_pct": 6.14
                }
            }
        },
        "company_data": {
            "RGTI": {
                "company_info": {"name": "Rigetti Computing", "sector": "Technology"},
                "financial_metrics": {
                    "pe_ratio": "N/A",  # Unprofitable
                    "forward_pe": -95.4,
                    "price_to_book": 18.4,
                    "profit_margin": 0,
                    "revenue_growth": -0.10,  # Actually -10%, not -41.6%
                    "roe": -48.81,
                    "debt_to_equity": 1.45,
                    "target_price": 27.50,  # Actual analyst consensus
                    "recommendation": "Buy",
                    "number_of_analysts": 5
                },
                "fundamental_analysis": {"overall_score": 3, "valuation": "Overvalued"}
            }
        },
        "news_data": {
            "RGTI": {
                "sentiment_analysis": {"dominant_sentiment": "Mixed", "article_count": 15},
                "key_events": ["Quantum computing momentum", "Government funding potential"]
            }
        },
        "twitter_sentiment": {
            "RGTI": {
                "sentiment_analysis": {"sentiment_label": "Bullish", "tweet_count": 250}
            }
        },
        "reddit_sentiment": {
            "RGTI": {
                "sentiment_analysis": {"sentiment_label": "Bullish", "post_count": 85}
            }
        },
        "risk_assessment": {
            "positions_risk": [
                {"symbol": "RGTI", "risk_score": 7.5, "risk_level": "High", "value_at_risk_1day": 150}
            ]
        }
    }
    
    result_state = agent.process(state)

    print("\nResearch Results:")
    for symbol, result in result_state.get("research_results", {}).items():
        print(f"\n{symbol}:")
        print(f"  Decision: {result['decision']}")
        print(f"  Conviction: {result['conviction']}/10")
        print(f"  Target Price: ${result['target_price']}")
        print(f"  Current Price: ${result.get('current_price', 'N/A')}")
        print("\n  Analysis Preview:")
        print(f"  {result['rationale'][:500]}...")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_deep_researcher_agent()