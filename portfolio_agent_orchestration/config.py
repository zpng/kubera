"""
Configuration for Portfolio Agent Orchestration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
PORTFOLIO_PATH = CONFIG_DIR / "portfolio.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model Selection for Each Agent
# Note: deep_researcher has automatic fallback to deepseek/deepseek-chat-v3.1 if primary model fails
AGENT_MODELS = {
    "portfolio_loader": "deepseek/deepseek-chat-v3.1",
    "historical_data": "deepseek/deepseek-chat-v3.1",
    "news_fetcher": "google/gemini-2.0-flash-exp",
    "company_info": "deepseek/deepseek-r1-distill-llama-70b",
    "sentiment_twitter": "nousresearch/hermes-3-llama-3.1-405b",
    "sentiment_reddit": "nousresearch/hermes-3-llama-3.1-405b",
    "risk_manager": "deepseek/deepseek-r1-0528",
    "deep_researcher": "openrouter/polaris-alpha"  # Advanced reasoning model with fallback to deepseek/deepseek-chat-v3.1
}

# Agent Configuration
AGENT_CONFIG = {
    "temperature": 0.7,
    "max_tokens": 4000,
    "timeout": 120,
}

# Data Source Configuration
DATA_SOURCES = {
    "yfinance": {
        "enabled": True,
        "cache_duration": 300,  # 5 minutes
    },
    "alpha_vantage": {
        "enabled": True,
        "api_key": ALPHA_VANTAGE_API_KEY,
        "rate_limit": 5,  # calls per minute
    },
    "twitter": {
        "enabled": False,  # Set to True when API keys are available
        "api_key": os.getenv("TWITTER_API_KEY"),
    },
    "reddit": {
        "enabled": False,  # Set to True when API keys are available
        "client_id": os.getenv("REDDIT_CLIENT_ID"),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET"),
    }
}

# Workflow Configuration
WORKFLOW_CONFIG = {
    "parallel_execution": True,
    "enable_caching": True,
    "retry_failed_agents": True,
    "max_retries": 3,
}

