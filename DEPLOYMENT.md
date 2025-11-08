# 🚀 Deployment Guide

## Prerequisites

- Python 3.9+
- OpenRouter API key
- (Optional) Alpha Vantage API key
- (Optional) Telegram Bot token

## Local Deployment

### 1. Setup Environment

```bash
# Clone repository
git clone <repo-url>
cd kubera

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

Create `.env` file:
```bash
OPENROUTER_API_KEY=your_openrouter_key
ALPHA_VANTAGE_API_KEY=your_av_key
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ADMIN_CHAT_ID=your_chat_id
```

Edit `config/portfolio.json` with your holdings.

### 3. Run

**Telegram Bot:**
```bash
python -m src.bot.telegram_bot
```

**Test Locally:**
```bash
python portfolio_agent_orchestration/test_local.py
```

---

## Railway Deployment

### Quick Deploy

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### Configuration

Add environment variables in Railway dashboard:
- `OPENROUTER_API_KEY`
- `ALPHA_VANTAGE_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_ADMIN_CHAT_ID`

Railway will automatically detect `requirements.txt` and `Dockerfile`.

---

## Docker Deployment

### Build Image

```bash
docker build -t kubera:latest .
```

### Run Container

```bash
docker run -d \
  -e OPENROUTER_API_KEY=your_key \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -v $(pwd)/config:/app/config \
  --name kubera \
  kubera:latest
```

### Docker Compose

```yaml
version: '3.8'
services:
  kubera:
    build: .
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    volumes:
      - ./config:/app/config
    restart: unless-stopped
```

Run: `docker-compose up -d`

---

## Terraform Deployment (Railway)

### Setup

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values
```

### Deploy

```bash
terraform init
terraform plan
terraform apply
```

### Destroy

```bash
terraform destroy
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | ✅ Yes | OpenRouter API key for LLM access |
| `ALPHA_VANTAGE_API_KEY` | ⚠️ Recommended | For enhanced news/data |
| `TELEGRAM_BOT_TOKEN` | ⚠️ For Telegram | Your Telegram bot token |
| `TELEGRAM_ADMIN_CHAT_ID` | ❌ Optional | Admin chat for notifications |

---

## Monitoring

### Logs

**Local:**
```bash
tail -f portfolio_analysis.log
```

**Docker:**
```bash
docker logs -f kubera
```

**Railway:**
Check logs in Railway dashboard

### Health Check

Test the workflow:
```bash
python portfolio_agent_orchestration/test_local.py
```

Expected output: JSON file in `outputs/` with analysis results.

---

## Troubleshooting

### OpenRouter API Errors

**Error:** `Invalid API key`
- Verify key in `.env`
- Check OpenRouter dashboard for API limits

### Missing Analysis

**Error:** Empty analysis for stocks
- Check console logs for agent warnings
- Verify data sources are accessible
- Review agent evaluation scores

### Telegram Bot Not Responding

**Error:** Bot doesn't respond to commands
- Verify bot token is correct
- Check bot is running: `ps aux | grep telegram_bot`
- Review logs for connection errors

---

## Scaling

### Multiple Instances

For high-volume analysis, deploy multiple instances:

```bash
# Instance 1: Portfolio analysis
python -m src.bot.telegram_bot

# Instance 2: Watchlist monitoring (future)
# python -m src.workflows.watchlist_monitor
```

### Rate Limiting

OpenRouter rate limits:
- Free tier: 20 requests/minute
- Paid tiers: Higher limits

Adjust `temperature` and `max_tokens` in `portfolio_agent_orchestration/config.py` to optimize costs.

---

## Backup & Recovery

### Portfolio Data

Backup `config/portfolio.json` regularly:
```bash
cp config/portfolio.json config/portfolio.backup.json
```

### Analysis History

Archive outputs periodically:
```bash
tar -czf outputs_$(date +%Y%m%d).tar.gz outputs/
```

---

## Updates

### Pull Latest Code

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Database Migrations

(None currently - all data in JSON files)

---

## Security

### API Keys

- Never commit `.env` to git
- Use environment variables in production
- Rotate keys periodically

### Access Control

Configure Telegram user whitelist in bot:
```python
ALLOWED_USERS = [123456789]  # Telegram user IDs
```

---

## Performance

### Optimization Tips

1. **Cache Data:** Historical data cached in `data/cache/`
2. **Parallel Execution:** Enabled by default in workflow
3. **Model Selection:** Use faster models for non-critical agents

### Expected Performance

- Single stock analysis: ~25 seconds
- 9 stock portfolio: ~3-4 minutes
- Includes data fetching, AI processing, fact-checking

---

## Support

For issues or questions:
1. Check logs for error messages
2. Review agent evaluation scores
3. Test individual agents: `python -m portfolio_agent_orchestration.agents.<agent_name>`

---

**Last Updated:** November 2025
