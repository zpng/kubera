---
name: market-analysis
description: Use this skill when analyzing stocks, performing technical analysis, or evaluating market conditions. Provides comprehensive stock analysis using TradingAgents framework including technical indicators, fundamental data, news sentiment, and social media analysis.
---

# Market Analysis Skill

## When to Use
Activate this skill when the user asks to:
- Analyze a specific stock ticker (e.g., "analyze NVDA")
- Perform technical analysis
- Evaluate market conditions
- Get stock recommendations
- Understand price movements
- Compare fundamental metrics

## Available Framework: TradingAgents

Located in `refs/TradingAgents/`, this provides:

### 1. Data Access Tools (refs/TradingAgents/tradingagents/agents/utils/agent_utils.py)
```python
# Import the abstracted data tools
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,      # Price data via yfinance/Alpha Vantage
    get_indicators,      # Technical indicators
    get_fundamentals,    # Company fundamentals
    get_balance_sheet,   # Balance sheet data
    get_cashflow,        # Cash flow statements
    get_income_statement,# Income statement
    get_news,            # Company news
    get_global_news,     # Market-wide news
    get_insider_sentiment,     # Insider trading sentiment
    get_insider_transactions   # Insider transactions
)
```

### 2. Analyst Agents (refs/TradingAgents/tradingagents/agents/analysts/)

#### Market Analyst (market_analyst.py)
**Purpose**: Technical analysis with indicators

**Key indicators to select** (choose 8 complementary ones):
- **Moving Averages**: close_50_sma, close_200_sma, close_10_ema
- **MACD**: macd, macds, macdh
- **Momentum**: rsi
- **Volatility**: boll, boll_ub, boll_lb, atr
- **Volume**: vwma

**Process**:
1. Call `get_stock_data(ticker, start_date, end_date)` first
2. Then call `get_indicators(ticker, indicator_list, start_date, end_date)`
3. Analyze trends, momentum, volatility
4. Provide detailed interpretation (not just "mixed trends")

#### Fundamentals Analyst (fundamentals_analyst.py)
**Purpose**: Analyze company financials and health

**Key metrics**:
- P/E ratio, EPS growth
- Revenue growth, profit margins
- Debt-to-equity ratio
- Cash flow health
- Insider activity patterns

#### News Analyst (news_analyst.py)
**Purpose**: Analyze news impact and sentiment

**Process**:
1. Get recent company news via `get_news(ticker)`
2. Get market-wide news via `get_global_news()`
3. Assess sentiment (bullish/bearish/neutral)
4. Identify catalysts and upcoming events

#### Social Media Analyst (social_media_analyst.py)
**Purpose**: Gauge retail investor sentiment

**Data sources**:
- Reddit sentiment (refs/TradingAgents/tradingagents/dataflows/reddit_utils.py)
- News aggregation for sentiment scoring

## Analysis Workflow

### Step 1: Data Collection
```python
# Get price data (ALWAYS call this first)
stock_data = get_stock_data(ticker, start_date, end_date)

# Calculate technical indicators
indicators = get_indicators(
    ticker,
    ["rsi", "macd", "boll_ub", "boll_lb", "close_50_sma", "close_200_sma", "atr", "vwma"],
    start_date,
    end_date
)

# Get fundamentals
fundamentals = get_fundamentals(ticker)
balance_sheet = get_balance_sheet(ticker)

# Get news
news = get_news(ticker)
global_news = get_global_news()
```

### Step 2: Multi-Dimensional Analysis

Analyze across these dimensions:

**Technical**:
- Trend direction (bullish/bearish/sideways)
- Momentum strength (RSI, MACD)
- Support/resistance levels
- Volatility assessment
- Volume trends

**Fundamental**:
- Valuation (overvalued/fair/undervalued)
- Financial health score
- Growth trajectory
- Red flags or concerns

**Sentiment**:
- News impact (positive/negative/neutral)
- Market mood
- Social sentiment
- Upcoming catalysts

### Step 3: Generate Report

**Required Format**:
```markdown
## Market Analysis Report: {TICKER}
**Date**: {current_date}

### Executive Summary
[One paragraph with key takeaway]

### Technical Analysis
**Trend**: [Bullish/Bearish/Neutral]
**Key Signals**:
- RSI ({value}): {interpretation}
- MACD ({value}): {interpretation}
- Bollinger Bands: {position relative to bands}
- Support: ${level}, Resistance: ${level}

**Volume Analysis**: {increasing/decreasing/stable}

### Fundamental Analysis
**Valuation**: P/E {value} (vs industry avg {value})
**Financial Health**: [Strong/Moderate/Weak]
**Growth Metrics**:
- Revenue: {YoY %}
- EPS: {YoY %}
- Margins: {%}

**Concerns**: {list any red flags}

### News & Sentiment
**Recent Headlines**:
1. {headline 1}
2. {headline 2}
3. {headline 3}

**Overall Sentiment**: [Positive/Neutral/Negative]
**Catalysts**: {upcoming events}

### Key Metrics Table
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Price | ${X} | {vs SMA levels} |
| RSI | {X} | {overbought/neutral/oversold} |
| P/E | {X} | {vs industry} |
| Revenue Growth | {X%} | {strong/weak} |

### Trading Recommendation
[Detailed reasoning combining all analysis]
**Action**: BUY/HOLD/SELL
**Confidence**: High/Medium/Low
**Risk Level**: High/Medium/Low
```

## Important Guidelines

1. **Always call get_stock_data FIRST** before requesting indicators
2. **Select complementary indicators** - avoid redundancy (e.g., don't use both RSI and StochRSI)
3. **Provide detailed, nuanced analysis** - never just say "trends are mixed" without elaboration
4. **Cross-reference signals** - technical should align with fundamental analysis
5. **Include markdown table** at the end for quick reference
6. **Consider multiple timeframes** - short-term vs long-term trends
7. **Document reasoning clearly** for ModelChat logging

## Code References

All code located in `refs/TradingAgents/`:
- Market Analyst: `tradingagents/agents/analysts/market_analyst.py`
- Fundamentals Analyst: `tradingagents/agents/analysts/fundamentals_analyst.py`
- News Analyst: `tradingagents/agents/analysts/news_analyst.py`
- Social Media Analyst: `tradingagents/agents/analysts/social_media_analyst.py`
- Data Tools: `tradingagents/agents/utils/agent_utils.py`
- Data Flows: `tradingagents/dataflows/`

## Example Usage

**User**: "Analyze NVDA stock"

**Response**:
1. Fetch NVDA price data from yfinance
2. Calculate 8 complementary technical indicators
3. Get fundamentals from Alpha Vantage
4. Fetch recent news
5. Perform comprehensive analysis across all dimensions
6. Generate detailed report with recommendation
7. Include metrics table for quick reference

## Integration with Multi-Model System

When multiple AI models use this skill:
- Each model analyzes independently
- Results aggregated by decision_aggregator
- Consensus and disagreements highlighted
- All reasoning logged to ModelChat for transparency
