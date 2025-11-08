# 🤖 Kubera - AI-Powered Portfolio Analysis

Multi-agent AI system for comprehensive stock portfolio analysis using LangChain, OpenRouter, and real-time market data.

## 🎯 What It Does

Analyzes your entire portfolio using **8 specialized AI agents**:
1. **Portfolio Loader** - Loads holdings from `config/portfolio.json`
2. **Historical Data** - Fetches price data, trends, technical indicators
3. **News Fetcher** - Gathers latest news and sentiment
4. **Company Info** - Collects fundamentals, earnings, analyst ratings
5. **Social Sentiment** - Twitter/Reddit sentiment analysis
6. **Risk Manager** - Calculates position and portfolio risk
7. **Deep Researcher** - Makes BUY/HOLD/SELL decisions with detailed rationale
8. **Fact Checker** - Validates all claims against actual data (prevents hallucinations)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create `.env` file:
```bash
# Required
OPENROUTER_API_KEY=your_key_here

# Optional
ALPHA_VANTAGE_API_KEY=your_key_here
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_ADMIN_CHAT_ID=your_chat_id
```

**Get API Keys:**
- OpenRouter: https://openrouter.ai (sign up, create API key)
- Alpha Vantage: https://www.alphavantage.co/support/#api-key (free)
- Telegram: Use @BotFather to create a bot

### 3. Add Your Portfolio
Edit `config/portfolio.json`:
```json
{
  "portfolio": {
    "stocks": [
      {
        "symbol": "AAPL",
        "shares": 10,
        "avg_cost": 150.00,
        "current_price": 175.00
      }
    ]
  }
}
```

### 4. Run Analysis

**Option A: Telegram Bot** (Recommended)
```bash
python -m src.bot.telegram_bot
```
Then in Telegram: `/analyze_portfolio`

**Option B: Test Locally**
```bash
python portfolio_agent_orchestration/test_local.py
```
Results saved to `outputs/portfolio_analysis_*.json`

## 📊 Output

For each stock, you receive:

✅ **Investment Decision**: BUY MORE / HOLD / SELL / TRIM POSITION  
✅ **Conviction Score**: 1-10 rating  
✅ **Target Price**: 3-6 month price target  
✅ **Detailed Analysis** (500+ words):
- Current position P&L breakdown
- Technical analysis (price trends, momentum)
- Fundamental strength (P/E, growth, margins)
- Analyst consensus comparison
- News and sentiment impact
- Risk assessment
- Decision rationale with specific data points
- Action items and watchlist

**Example Output:**
```
🟢 NVDA - BUY MORE
Conviction: 8/10
Target Price: $210.00

Detailed Analysis:
**1. CURRENT POSITION REVIEW**
Holdings: 0.167 shares at $113.48 avg cost
Current Value: $31.44
Unrealized P/L: +$12.41 (+65.5%)

**2. TECHNICAL ANALYSIS**
Current Price: $187.77
6-Month Return: +42.3%
Trend: Strong bullish momentum...

[Continues with 6 more detailed sections]
```

## 🏗️ Architecture

```
kubera/
├── portfolio_agent_orchestration/   # Main multi-agent system
│   ├── agents/                      # 8 specialized AI agents
│   ├── workflows/                   # Orchestration logic
│   └── config.py                    # Configuration
├── src/                             # Legacy components (being phased out)
│   ├── bot/telegram_bot.py         # Telegram integration
│   └── [old agents/data modules]
├── config/portfolio.json            # Your portfolio data
├── terraform/                       # Infrastructure as code
├── Dockerfile                       # Container deployment
└── refs/                           # Reference frameworks (CrewAI, etc.)
```

## 🔧 Key Features

### Fact-Based Analysis (No Hallucinations)
- Strict prompting: "Use ONLY provided data"
- Fact Checker validates all claims
- Automatic fallback if analysis too short
- Logs analysis quality for each stock

### Agent Evaluation
- Scores each agent's performance (/10)
- Identifies missing data and issues
- Overall workflow health assessment
- Penalties for short or missing analysis

### Multi-Stage Workflow
1. Load Portfolio
2. Parallel Data Collection (Historical, News, Company, Sentiment)
3. Risk Assessment
4. Deep Research & Decisions
5. Fact Checking (validates outputs)
6. Agent Performance Evaluation

## 🤖 AI Models Used

Each agent uses the optimal model for its task:

| Agent | Model | Purpose |
|-------|-------|---------|
| Portfolio Loader | deepseek-chat-v3.1 | Fast structured data parsing |
| Historical/News | deepseek-chat-v3.1 | Efficient data retrieval |
| Company Info | deepseek-r1-distill-llama-70b | Cost-efficient analysis |
| Sentiment | hermes-3-llama-3.1-405b | Strong at sentiment/conversation |
| Risk Manager | deepseek-r1-0528 | Complex risk calculations |
| Deep Researcher | deepseek-r1-distill-llama-70b | Reliable fact-based reasoning |

All models accessed via **OpenRouter** for unified API.

## 📡 Deployment

### Local Development
```bash
python -m src.bot.telegram_bot
```

### Railway (Cloud)
```bash
railway up
```
See `DEPLOYMENT.md` for detailed instructions.

### Docker
```bash
docker build -t kubera .
docker run -e OPENROUTER_API_KEY=key kubera
```

### Terraform
```bash
cd terraform
terraform init
terraform apply
```

## 🧪 Testing

**Test Single Stock:**
```bash
python -m portfolio_agent_orchestration.agents.deep_researcher
```

**Test Full Workflow:**
```bash
python portfolio_agent_orchestration/test_local.py
```

## 📝 Configuration

**`config/portfolio.json`** - Your holdings  
**`.env`** - API keys and secrets  
**`portfolio_agent_orchestration/config.py`** - Agent models and settings

## 🎓 Learning Resources

- **`refs/AI-Trader/`** - Reference multi-agent trading system
- **`refs/TradingAgents/`** - CrewAI examples
- **`refs/awesome-quant/`** - Quantitative finance tools

## ⚠️ Disclaimer

This is an AI-powered analysis tool for educational and informational purposes only. 

**NOT financial advice. Always do your own research.**

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using LangChain, OpenRouter, and open-source LLMs**
