#!/usr/bin/env python3
"""
Test deep_researcher with single stock (TSLA) to verify table format fixes field confusion
"""
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from portfolio_agent_orchestration.agents.historical_data import HistoricalDataAgent
from portfolio_agent_orchestration.agents.company_info import CompanyInfoAgent
from portfolio_agent_orchestration.agents.deep_researcher import DeepResearcherAgent

def test_deep_researcher_tsla():
    """Test deep researcher with real TSLA data"""

    symbol = "TSLA"
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing DeepResearcher with {symbol}")
    logger.info(f"{'='*80}\n")

    # Step 1: Collect historical data
    logger.info("Step 1: Collecting historical data...")
    hist_agent = HistoricalDataAgent()
    hist_data = hist_agent.process_symbol(symbol, period='1y')

    if hist_data['status'] != 'success':
        logger.error(f"Failed to get historical data: {hist_data}")
        return False

    logger.info(f"✓ Historical data collected:")
    logger.info(f"  Current Price: ${hist_data['metrics']['current_price']}")
    logger.info(f"  YTD Return: {hist_data['metrics'].get('ytd_return_pct', 'N/A')}%")
    logger.info(f"  52-Week Range: ${hist_data['metrics'].get('fifty_two_week_low')} - ${hist_data['metrics'].get('fifty_two_week_high')}")

    # Step 2: Collect company info
    logger.info("\nStep 2: Collecting company fundamentals...")
    comp_agent = CompanyInfoAgent()
    comp_data = comp_agent.process_symbol(symbol)

    if comp_data['status'] != 'success':
        logger.error(f"Failed to get company data: {comp_data}")
        return False

    metrics = comp_data['financial_metrics']
    logger.info(f"✓ Company data collected:")
    logger.info(f"  P/E Ratio (Trailing): {metrics['pe_ratio']}")
    logger.info(f"  Forward P/E: {metrics['forward_pe']}")
    logger.info(f"  Debt/Equity: {metrics['debt_to_equity']}")
    logger.info(f"  Total Cash: ${metrics['total_cash_millions']}M")

    # Step 3: Prepare complete data package
    logger.info("\nStep 3: Preparing data package for deep researcher...")
    all_data = {
        'portfolio_position': {
            'shares': 100,
            'avg_cost': 250.00,
            'current_price': hist_data['metrics']['current_price']
        },
        'historical_data': hist_data,
        'company_data': comp_data,
        'news_data': {
            'sentiment_analysis': {
                'article_count': 5,
                'dominant_sentiment': 'NEUTRAL'
            },
            'key_events': []
        },
        'twitter_sentiment': {
            'sentiment_analysis': {'tweet_count': 0}
        },
        'reddit_sentiment': {
            'sentiment_analysis': {'post_count': 0}
        },
        'risk_metrics': {
            'risk_score': 6,
            'risk_level': 'MODERATE',
            'value_at_risk_1day': 500
        }
    }

    # Step 4: Run deep researcher
    logger.info("\nStep 4: Running deep researcher with new table format...")
    logger.info("(This will make an API call to OpenRouter - may take 30-60 seconds)")

    researcher = DeepResearcherAgent()

    try:
        result = researcher.research_symbol(symbol, all_data)
    except Exception as e:
        logger.error(f"❌ Deep researcher failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Validate output
    logger.info("\nStep 5: Validating output for field confusion...")
    output = result['rationale']

    # Save output for inspection
    output_file = Path('/tmp/tsla_analysis_output.txt')
    output_file.write_text(output)
    logger.info(f"✓ Full output saved to: {output_file}")

    # Validation checks
    validation_results = []

    # Check 1: P/E Ratio should be ~294, NOT 132 (Forward P/E)
    pe_mentions = re.findall(r'P/E Ratio[:\s]+(\d+\.?\d*)', output, re.IGNORECASE)
    actual_pe = float(metrics['pe_ratio'])
    forward_pe = float(metrics['forward_pe'])

    logger.info(f"\n--- Validation Check 1: P/E Ratio Accuracy ---")
    logger.info(f"Expected P/E: {actual_pe:.2f}")
    logger.info(f"Forward P/E (should be different): {forward_pe:.2f}")

    pe_correct = True
    for mentioned_pe in pe_mentions:
        pe_val = float(mentioned_pe)
        logger.info(f"Found in output: P/E Ratio: {pe_val}")

        # Check if it's close to actual P/E (within 5%)
        if abs(pe_val - actual_pe) / actual_pe < 0.05:
            logger.info(f"  ✓ CORRECT - matches actual P/E")
        # Check if it's incorrectly using Forward P/E
        elif abs(pe_val - forward_pe) / forward_pe < 0.05:
            logger.error(f"  ❌ WRONG - using Forward P/E ({forward_pe:.2f}) as P/E Ratio!")
            pe_correct = False
        else:
            logger.warning(f"  ⚠️  Neither P/E nor Forward P/E - may be rounded")

    validation_results.append(("P/E Ratio Accuracy", pe_correct))

    # Check 2: 52-Week Low should be ~$214, NOT $41,647M (Total Cash)
    fifty_two_low_mentions = re.findall(r'52-Week Low[:\s]+\$?([\d,]+\.?\d*)', output, re.IGNORECASE)
    actual_52w_low = float(hist_data['metrics'].get('fifty_two_week_low', 0))
    total_cash_millions = float(metrics.get('total_cash_millions', 0))

    logger.info(f"\n--- Validation Check 2: 52-Week Low Accuracy ---")
    logger.info(f"Expected 52-Week Low: ${actual_52w_low:.2f}")
    logger.info(f"Total Cash (should NOT be used): ${total_cash_millions:.1f}M")

    low_correct = True
    for mentioned_low in fifty_two_low_mentions:
        low_val = float(mentioned_low.replace(',', ''))
        logger.info(f"Found in output: 52-Week Low: ${low_val}")

        # Check if it's close to actual 52-week low (within 10%)
        if abs(low_val - actual_52w_low) / actual_52w_low < 0.10:
            logger.info(f"  ✓ CORRECT - matches actual 52-week low")
        # Check if it's incorrectly using Total Cash
        elif abs(low_val - total_cash_millions) / total_cash_millions < 0.01:
            logger.error(f"  ❌ WRONG - using Total Cash (${total_cash_millions:.1f}M) as 52-Week Low!")
            low_correct = False
        else:
            logger.warning(f"  ⚠️  Unexpected value - check manually")

    validation_results.append(("52-Week Low Accuracy", low_correct))

    # Check 3: YTD Return should be ~13.25%, NOT 33.72% (6-month)
    ytd_mentions = re.findall(r'(?:YTD|Year-to-Date).*?(\d+\.?\d*)%', output, re.IGNORECASE)
    actual_ytd = float(hist_data['metrics'].get('ytd_return_pct', 0))
    six_month = float(hist_data['metrics'].get('period_return_pct', 0))

    logger.info(f"\n--- Validation Check 3: YTD Return Accuracy ---")
    logger.info(f"Expected YTD: {actual_ytd:.2f}%")
    logger.info(f"6-Month Return (should be different): {six_month:.2f}%")

    ytd_correct = True
    for mentioned_ytd in ytd_mentions:
        ytd_val = float(mentioned_ytd)
        logger.info(f"Found in output: YTD: {ytd_val}%")

        if abs(ytd_val - actual_ytd) < 2:  # Within 2% tolerance
            logger.info(f"  ✓ CORRECT - matches actual YTD")
        elif abs(ytd_val - six_month) < 2:
            logger.error(f"  ❌ WRONG - using 6-month return ({six_month:.2f}%) as YTD!")
            ytd_correct = False
        else:
            logger.warning(f"  ⚠️  Different value - may be rounded")

    validation_results.append(("YTD Return Accuracy", ytd_correct))

    # Check 4: Conviction and Target Price extracted
    logger.info(f"\n--- Validation Check 4: Conviction & Target Price ---")
    logger.info(f"Conviction: {result.get('conviction', 'NOT FOUND')}/10")
    logger.info(f"Target Price: ${result.get('target_price', 'NOT FOUND')}")
    logger.info(f"Decision: {result.get('decision', 'NOT FOUND')}")

    conviction_ok = result.get('conviction', 5) != 5  # Default is 5
    target_ok = result.get('target_price', 0) > 0

    validation_results.append(("Conviction Extracted", conviction_ok))
    validation_results.append(("Target Price Extracted", target_ok))

    # Check 5: Validation checkpoints completed
    logger.info(f"\n--- Validation Check 5: Validation Checkpoints ---")
    has_validation = "CRITICAL VALIDATION:" in output
    logger.info(f"Has validation checkpoints: {has_validation}")
    if has_validation:
        logger.info("  ✓ Validation checkpoints found in output")
    else:
        logger.warning("  ⚠️  Validation checkpoints not found - LLM may have skipped them")

    validation_results.append(("Validation Checkpoints", has_validation))

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    all_passed = True
    for check_name, passed in validation_results:
        status = "✓ PASS" if passed else "❌ FAIL"
        logger.info(f"{status:10} | {check_name}")
        if not passed:
            all_passed = False

    logger.info(f"{'='*80}")

    if all_passed:
        logger.info("✅ ALL VALIDATIONS PASSED - Table format is working!")
        logger.info(f"\nOutput preview (first 500 chars):")
        logger.info(output[:500] + "...")
        return True
    else:
        logger.error("❌ SOME VALIDATIONS FAILED - Field confusion still present")
        logger.info(f"\nFull output saved to: {output_file}")
        logger.info("Review the output file to identify specific issues")
        return False


if __name__ == "__main__":
    success = test_deep_researcher_tsla()
    sys.exit(0 if success else 1)
