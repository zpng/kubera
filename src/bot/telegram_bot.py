"""
Kubera Telegram Bot
Main bot interface for user interaction
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

# Setup logging FIRST
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Old imports - made optional since we use new workflow
try:
    from src.orchestrator.autonomous_pipeline import AutonomousTradingPipeline
    from src.data.market_data import MarketDataProvider
    OLD_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Old pipeline not available: {e}")
    OLD_PIPELINE_AVAILABLE = False
    AutonomousTradingPipeline = None
    MarketDataProvider = None


class TelegramBot:
    """Telegram Bot for Kubera Trading System"""
    
    def __init__(self, token: str, allowed_users: Optional[List[int]] = None):
        """Initialize bot with token and optional user whitelist"""
        self.token = token
        self.allowed_users = allowed_users if allowed_users else []
        
        # Initialize pipeline and data provider (optional - only for old commands)
        self.pipeline = None
        self.market_data_provider = None
        
        if OLD_PIPELINE_AVAILABLE:
            try:
                self.pipeline = AutonomousTradingPipeline()
                self.market_data_provider = MarketDataProvider()
                logger.info("✅ Old pipeline initialized (for legacy commands)")
            except Exception as e:
                logger.warning(f"⚠️  Old pipeline failed to initialize: {e}")
                logger.info("✅ Bot will use new workflow for /analyze_portfolio")
        else:
            logger.info("✅ Using new multi-agent workflow only")
        
        # Build application
        self.application = Application.builder().token(token).build()
        
        # Register handlers
        self._register_handlers()
        
        logger.info("✅ Telegram bot initialized")
    
    def _register_handlers(self):
        """Register all command and callback handlers"""
        # Commands
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        self.application.add_handler(CommandHandler("watchlist", self.watchlist_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("analyze_portfolio", self.analyze_portfolio_command))
        self.application.add_handler(CommandHandler("analyze_watchlist", self.analyze_watchlist_command))
        self.application.add_handler(CommandHandler("discover", self.discover_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        
        # Callback query handler for buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
    
    def _check_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed (if whitelist is enabled)"""
        if not self.allowed_users:
            return True  # No whitelist, allow everyone
        return user_id in self.allowed_users
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command"""
        user = update.effective_user
        
        if not self._check_user_allowed(user.id):
            await update.message.reply_text("❌ Unauthorized. Contact admin for access.")
            return
        
        keyboard = [
            [
                InlineKeyboardButton("📊 Portfolio", callback_data="portfolio"),
                InlineKeyboardButton("👀 Watchlist", callback_data="watchlist"),
            ],
            [
                InlineKeyboardButton("🔍 Analyze Portfolio", callback_data="analyze_portfolio"),
                InlineKeyboardButton("🔍 Analyze Watchlist", callback_data="analyze_watchlist"),
            ],
            [
                InlineKeyboardButton("🚀 Discover Stocks", callback_data="discover"),
                InlineKeyboardButton("📈 Full Analysis", callback_data="analyze_all"),
            ],
            [
                InlineKeyboardButton("⚙️ Status", callback_data="status"),
                InlineKeyboardButton("❓ Help", callback_data="help"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = f"""
🤖 **Welcome to Kubera Trading Bot!**

Hi {user.first_name}! I'm your AI-powered trading assistant.

**What I can do:**
• 📊 Track your portfolio & watchlist
• 🔍 Run deep AI analysis on stocks
• 🚀 Discover trending stocks from YouTube/X/News
• 📈 Provide investment recommendations

**Quick Start:**
Use the buttons below or type:
• /portfolio - View your holdings
• /analyze_portfolio - Analyze portfolio stocks
• /discover - Find trending stocks
• /help - Show all commands

Let's get started! 🚀
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = """
📚 **Kubera Bot Commands**

**Portfolio & Watchlist:**
• `/portfolio` - View current holdings with P&L
• `/watchlist` - View stocks on watchlist

**Analysis:**
• `/analyze_portfolio` - Analyze portfolio stocks (Priority 1)
• `/analyze_watchlist` - Analyze watchlist stocks (Priority 2)
• `/discover` - Discover & analyze trending stocks (Priority 3)
• `/analyze` - Full analysis (all priorities)

**System:**
• `/status` - Check system status
• `/help` - Show this help message

**Analysis Priority:**
1️⃣ **Portfolio** - Your current holdings (9 stocks)
2️⃣ **Watchlist** - Future candidates (15 stocks)
3️⃣ **Discovered** - Trending from YouTube/X/News

**How it works:**
The bot runs a 2-stage AI analysis:
- Stage 1: 4 analyst reports (Market, News, Sentiment, Fundamentals)
- Stage 2: Investment debate (Bull vs Bear, Judge decision)

Results include: BUY/HOLD/SELL with conviction score & price targets.

**Pro Tips:**
• Start with `/analyze_portfolio` (analyzes only your holdings)
• Full analysis takes 5-10 minutes per stock
• Use `/discover` to find new opportunities
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /portfolio command"""
        try:
            # Load portfolio from config file directly
            import json
            from pathlib import Path
            
            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            portfolio = portfolio_data.get('portfolio', {}).get('stocks', [])
            
            if not portfolio:
                await update.message.reply_text("📊 Your portfolio is empty.")
                return
            
            message = "📊 **Your Portfolio**\n\n"
            total_equity = 0
            total_pl = 0
            
            for stock in portfolio:
                symbol = stock['symbol']
                shares = stock['shares']
                avg_cost = stock['avg_cost']
                current_price = stock.get('current_price', avg_cost)
                equity = shares * current_price
                pl = equity - (shares * avg_cost)
                pl_pct = (pl / (shares * avg_cost)) * 100 if shares * avg_cost > 0 else 0
                
                total_equity += equity
                total_pl += pl
                
                emoji = "🟢" if pl >= 0 else "🔴"
                message += f"{emoji} **{symbol}**\n"
                message += f"  Shares: {shares:.2f}\n"
                message += f"  Avg Cost: ${avg_cost:.2f}\n"
                message += f"  Current: ${current_price:.2f}\n"
                message += f"  Equity: ${equity:.2f}\n"
                message += f"  P&L: ${pl:.2f} ({pl_pct:+.2f}%)\n\n"
            
            total_pl_pct = (total_pl / (total_equity - total_pl)) * 100 if (total_equity - total_pl) > 0 else 0
            emoji = "🟢" if total_pl >= 0 else "🔴"
            
            message += f"**Total**\n"
            message += f"  Equity: ${total_equity:.2f}\n"
            message += f"  P&L: {emoji} ${total_pl:.2f} ({total_pl_pct:+.2f}%)"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in portfolio_command: {e}")
            await update.message.reply_text(f"❌ Error loading portfolio: {str(e)}")
    
    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /watchlist command"""
        try:
            watchlist = self.pipeline.portfolio_config.get('watchlist', {}).get('stocks', [])
            
            if not watchlist:
                await update.message.reply_text("👀 Your watchlist is empty.")
                return
            
            message = "👀 **Your Watchlist**\n\n"
            
            for i, symbol in enumerate(watchlist, 1):
                # Get current price
                try:
                    price_info = self.market_data_provider.get_stock_price(symbol)
                    current_price = price_info.get('current_price', 'N/A')
                    change_pct = price_info.get('change_percent', 0)
                    emoji = "🟢" if change_pct >= 0 else "🔴"
                    message += f"{i}. **{symbol}** - ${current_price} {emoji} {change_pct:+.2f}%\n"
                except:
                    message += f"{i}. **{symbol}**\n"
            
            message += f"\n**Total:** {len(watchlist)} stocks"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in watchlist_command: {e}")
            await update.message.reply_text(f"❌ Error loading watchlist: {str(e)}")
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command - Full analysis"""
        await update.message.reply_text(
            "🚀 **Starting Full Analysis**\n\n"
            "This will analyze:\n"
            "1️⃣ Portfolio holdings (9 stocks)\n"
            "2️⃣ Watchlist stocks (15 stocks)\n"
            "3️⃣ Discovered trending stocks\n\n"
            "⏱️ This may take 20-30 minutes...\n"
            "I'll send updates as I progress."
        )
        
        try:
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=True,
                analyze_watchlist=True,
                analyze_discovered=True,
                top_n_discovered=5
            )
            
            # Send results
            await self._send_recommendations(update, recommendations, "Full Analysis")
            
        except Exception as e:
            logger.error(f"Error in analyze_command: {e}")
            await update.message.reply_text(f"❌ Analysis failed: {str(e)}")
    
    async def analyze_portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze_portfolio command - NEW MULTI-AGENT WORKFLOW"""
        try:
            # Import the new workflow
            from portfolio_agent_orchestration.workflows.main_workflow import PortfolioAnalysisWorkflow
            import json
            from pathlib import Path
            
            # Load portfolio count
            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            portfolio_count = len(portfolio_data.get('portfolio', {}).get('stocks', []))
            
            await update.message.reply_text(
                f"🤖 **NEW: Multi-Agent Portfolio Analysis**\n\n"
                f"📊 Analyzing {portfolio_count} stocks\n"
                f"⏱️ Estimated time: ~{portfolio_count * 25} seconds\n\n"
                f"🔬 Running 8 specialized AI agents...\n"
                f"• Portfolio Loader\n"
                f"• Historical Data\n"
                f"• News Fetcher\n"
                f"• Company Fundamentals\n"
                f"• Risk Manager\n"
                f"• Deep Researcher\n\n"
                f"Please wait..."
            )
            
            # Run new workflow
            workflow = PortfolioAnalysisWorkflow()
            result_state = await asyncio.to_thread(workflow.run)
            
            if result_state.get("workflow_error"):
                await update.message.reply_text(f"❌ Analysis failed: {result_state['workflow_error']}")
                return
            
            # Format results for Telegram
            await self._send_new_portfolio_results(update, result_state)
            
        except Exception as e:
            logger.error(f"Error in analyze_portfolio_command: {e}")
            await update.message.reply_text(f"❌ Analysis failed: {str(e)}")
    
    async def _send_new_portfolio_results(self, update: Update, state: Dict[str, Any]) -> None:
        """Send formatted results from new workflow"""
        summary = state.get("portfolio_summary", {})
        risk = state.get("risk_assessment", {}).get("portfolio_risk", {})
        research = state.get("research_results", {})
        
        # Portfolio Overview
        message = f"""🤖 **KUBERA PORTFOLIO ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━

📊 **Portfolio Overview**
• Stocks: {summary.get('total_stocks', 0)}
• Total Value: ${summary.get('total_equity', 0):,.2f}
• Total P&L: ${summary.get('unrealized_pl', 0):,.2f}

⚠️ **Risk Assessment**
• Risk Level: {risk.get('portfolio_risk_level', 'Unknown')}
• Risk Score: {risk.get('portfolio_risk_score', 'N/A')}/10
• Diversification: {risk.get('diversification_grade', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━
🔬 **RECOMMENDATIONS**
"""
        
        # Send overview
        await update.message.reply_text(message, parse_mode='Markdown')
        
        # Send each stock recommendation
        emoji_map = {
            "BUY MORE": "🟢",
            "HOLD": "🟡",
            "SELL": "🔴",
            "TRIM POSITION": "🟠"
        }
        
        for symbol, result in research.items():
            decision = result.get("decision", "HOLD")
            conviction = result.get("conviction", 5)
            target_price = result.get("target_price", 0)
            rationale = result.get("rationale", "No analysis available")
            
            emoji = emoji_map.get(decision, "⚪")
            
            stock_msg = f"""
{emoji} **{symbol}** - {decision}

**Conviction:** {conviction}/10
"""
            if target_price > 0:
                stock_msg += f"**Target Price:** ${target_price:.2f}\n"
            
            # Split rationale into chunks if too long (Telegram limit is 4096 chars per message)
            stock_msg += f"\n**Detailed Analysis:**\n{rationale}\n"
            stock_msg += "─" * 30
            
            # Send stock message, splitting if too long
            try:
                if len(stock_msg) > 4000:
                    # Split into multiple messages
                    parts = []
                    current_part = f"\n{emoji} **{symbol}** - {decision}\n\n**Conviction:** {conviction}/10\n"
                    if target_price > 0:
                        current_part += f"**Target Price:** ${target_price:.2f}\n"
                    current_part += f"\n**Detailed Analysis (Part 1):**\n"
                    parts.append(current_part)
                    
                    # Split rationale into chunks
                    remaining = rationale
                    part_num = 1
                    while remaining:
                        chunk_size = 3500  # Leave room for headers
                        chunk = remaining[:chunk_size]
                        remaining = remaining[chunk_size:]
                        
                        if part_num == 1:
                            parts[0] += chunk
                        else:
                            parts.append(f"**{symbol} Analysis (Part {part_num + 1}):**\n{chunk}")
                        part_num += 1
                    
                    # Send each part
                    for part in parts:
                        await update.message.reply_text(part, parse_mode='Markdown')
                        await asyncio.sleep(0.5)  # Small delay between messages
                else:
                    await update.message.reply_text(stock_msg, parse_mode='Markdown')
            except Exception as e:
                # If markdown fails, send as plain text
                logger.warning(f"Markdown failed for {symbol}, sending plain text: {e}")
                plain_msg = stock_msg.replace('*', '').replace('_', '')
                if len(plain_msg) > 4000:
                    # Split plain text too
                    for i in range(0, len(plain_msg), 4000):
                        await update.message.reply_text(plain_msg[i:i+4000])
                        await asyncio.sleep(0.5)
                else:
                    await update.message.reply_text(plain_msg)
        
        # Final summary with agent evaluation
        duration = state.get("workflow_duration", 0)
        evaluation = state.get("agent_evaluation", {})
        
        final_msg = f"""✅ **Analysis Complete**

⏱️ Duration: {duration:.1f}s
💡 This is not financial advice. Always do your own research."""
        
        # Add agent performance summary if available
        if evaluation:
            overall_pct = evaluation.get("percentage", 0)
            status = evaluation.get("status", "UNKNOWN")
            
            if overall_pct >= 80:
                performance_emoji = "🌟"
            elif overall_pct >= 70:
                performance_emoji = "✅"
            else:
                performance_emoji = "⚠️"
            
            final_msg += f"\n\n{performance_emoji} **Agent Performance: {overall_pct:.0f}%** ({status})"
        
        await update.message.reply_text(final_msg)

        logger.info("Portfolio analysis results sent to Telegram.")
    
    async def analyze_watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze_watchlist command"""
        watchlist_count = len(self.pipeline.portfolio_config.get('watchlist', {}).get('stocks', []))
        
        await update.message.reply_text(
            f"🔍 **Analyzing Watchlist**\n\n"
            f"Stocks to analyze: {watchlist_count}\n"
            f"⏱️ Estimated time: {watchlist_count * 2}-{watchlist_count * 3} minutes\n\n"
            f"Running deep AI analysis..."
        )
        
        try:
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=False,
                analyze_watchlist=True,
                analyze_discovered=False
            )
            
            await self._send_recommendations(update, recommendations, "Watchlist Analysis")
            
        except Exception as e:
            logger.error(f"Error in analyze_watchlist_command: {e}")
            await update.message.reply_text(f"❌ Analysis failed: {str(e)}")
    
    async def discover_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /discover command"""
        await update.message.reply_text(
            "🚀 **Discovering Trending Stocks**\n\n"
            "Scanning:\n"
            "• 📺 YouTube finance videos\n"
            "• 🐦 X/Twitter discussions\n"
            "• 📰 Financial news\n\n"
            "⏱️ This may take 10-15 minutes..."
        )
        
        try:
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=False,
                analyze_watchlist=False,
                analyze_discovered=True,
                top_n_discovered=5
            )
            
            await self._send_recommendations(update, recommendations, "Stock Discovery")
            
        except Exception as e:
            logger.error(f"Error in discover_command: {e}")
            await update.message.reply_text(f"❌ Discovery failed: {str(e)}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        try:
            portfolio_count = len(self.pipeline.portfolio_config.get('portfolio', {}).get('stocks', []))
            watchlist_count = len(self.pipeline.portfolio_config.get('watchlist', {}).get('stocks', []))
            
            status_text = f"""
⚙️ **System Status**

✅ **Online**

**Portfolio:**
• Holdings: {portfolio_count} stocks
• Watchlist: {watchlist_count} stocks

**AI Models:**
• Stage 1 Analysts: 4 models
• Stage 2 Debate: 3 models
• Discovery: 2 models

**Data Sources:**
• Market Data: yfinance + Alpha Vantage
• News: Alpha Vantage API
• Discovery: YouTube + X/Twitter

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button clicks"""
        query = update.callback_query
        await query.answer()
        
        # Map callback data to commands
        command_map = {
            "portfolio": self.portfolio_command,
            "watchlist": self.watchlist_command,
            "analyze_portfolio": self.analyze_portfolio_command,
            "analyze_watchlist": self.analyze_watchlist_command,
            "analyze_all": self.analyze_command,
            "discover": self.discover_command,
            "status": self.status_command,
            "help": self.help_command,
        }
        
        callback_data = query.data
        if callback_data in command_map:
            # Create a fake update with message
            fake_update = Update(
                update_id=update.update_id,
                message=query.message,
            )
            await command_map[callback_data](fake_update, context)
    
    async def _send_recommendations(
        self,
        update: Update,
        recommendations: List[Dict[str, Any]],
        title: str
    ) -> None:
        """Send formatted recommendations"""
        if not recommendations:
            await update.message.reply_text(f"✅ **{title} Complete**\n\nNo recommendations to display.")
            return
        
        message = f"✅ **{title} Complete**\n\n"
        message += f"**Top Recommendations:**\n\n"
        
        for i, rec in enumerate(recommendations[:10], 1):  # Top 10
            symbol = rec['symbol']
            decision = rec.get('final_decision', 'N/A')
            conviction = rec.get('conviction', 0)
            target = rec.get('target_price', 0)
            priority_label = rec.get('priority_label', 'Discovered')
            
            emoji_map = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
            emoji = emoji_map.get(decision, "⚪")
            
            message += f"{i}. {emoji} **{symbol}** - {decision}\n"
            message += f"   Priority: {priority_label}\n"
            message += f"   Conviction: {conviction}/10\n"
            message += f"   Target: ${target:.2f}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors"""
        logger.error(f"Exception while handling an update: {context.error}")
        
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "❌ An error occurred. Please try again later."
            )
    
    def run(self) -> None:
        """Start the bot"""
        logger.info("🤖 Starting Kubera Telegram Bot...")
        logger.info("✅ Kubera Bot is running! Press Ctrl+C to stop.")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main entry point"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get bot token
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set in environment variables!")
        logger.error("Add it to your .env file or export it:")
        logger.error("  export TELEGRAM_BOT_TOKEN='your_token_here'")
        sys.exit(1)
    
    # Get allowed users (optional)
    allowed_users_str = os.getenv('TELEGRAM_ALLOWED_USERS', '')
    allowed_users = [int(uid.strip()) for uid in allowed_users_str.split(',') if uid.strip()]
    
    if allowed_users:
        logger.info(f"✅ User whitelist enabled: {len(allowed_users)} users")
    else:
        logger.info("⚠️  No user whitelist - bot is open to everyone")
    
    # Create and run bot
    try:
        bot = TelegramBot(token, allowed_users if allowed_users else None)
        bot.run()
    except KeyboardInterrupt:
        logger.info("\n🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Bot failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
