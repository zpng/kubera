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
            await update.message.reply_text("❌ 未授权。请联系管理员获取访问权限。")
            return
        
        keyboard = [
            [
                InlineKeyboardButton("📊 持仓", callback_data="portfolio"),
                InlineKeyboardButton("👀 自选", callback_data="watchlist"),
            ],
            [
                InlineKeyboardButton("🔍 分析持仓", callback_data="analyze_portfolio"),
                InlineKeyboardButton("🔍 分析自选", callback_data="analyze_watchlist"),
            ],
            [
                InlineKeyboardButton("🚀 发现股票", callback_data="discover"),
                InlineKeyboardButton("📈 全量分析", callback_data="analyze_all"),
            ],
            [
                InlineKeyboardButton("⚙️ 状态", callback_data="status"),
                InlineKeyboardButton("❓ 帮助", callback_data="help"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        welcome_text = f"""
🤖 **欢迎使用 Kubera 股票助手！**

你好 {user.first_name}，我是你的 AI 投研助手。

**我能做什么：**
• 📊 跟踪你的持仓与自选
• 🔍 对股票进行深度 AI 分析
• 🚀 从 YouTube/X/新闻发现热门股票
• 📈 提供投资建议与目标价

**快速开始：**
可点击下方按钮或输入命令：
• /portfolio - 查看当前持仓
• /analyze_portfolio - 分析持仓股票
• /discover - 发现热门股票
• /help - 查看所有命令

一起开始吧！🚀
        """
        
        await update.message.reply_text(
            welcome_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command"""
        help_text = """
📚 **Kubera 机器人命令**

**持仓与自选：**
• `/portfolio` - 查看持仓与盈亏
• `/watchlist` - 查看自选列表

**分析：**
• `/analyze_portfolio` - 分析持仓股票（优先级 1）
• `/analyze_watchlist` - 分析自选股票（优先级 2）
• `/discover` - 发现并分析热门股票（优先级 3）
• `/analyze` - 全量分析（所有优先级）

**系统：**
• `/status` - 查看系统状态
• `/help` - 显示本帮助

**分析流程：**
共两阶段 AI 分析：
- 阶段 1：四位分析师（行情、新闻、情绪、基本面）
- 阶段 2：多轮辩论（多头 vs 空头，裁判定论）

输出包含：买入/持有/卖出建议、信心评分、目标价等。

**建议：**
• 先运行 `/analyze_portfolio`（仅分析持仓，更快）
• 单支股票完整分析约需 5–10 分钟
• 用 `/discover` 发现新的机会
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
                await update.message.reply_text("📊 你的持仓为空。")
                return
            
            message = "📊 **当前持仓**\n\n"
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
                message += f"  持股数: {shares:.2f}\n"
                message += f"  平均成本: ${avg_cost:.2f}\n"
                message += f"  当前价格: ${current_price:.2f}\n"
                message += f"  市值: ${equity:.2f}\n"
                message += f"  未实现盈亏: ${pl:.2f} ({pl_pct:+.2f}%)\n\n"
            
            total_pl_pct = (total_pl / (total_equity - total_pl)) * 100 if (total_equity - total_pl) > 0 else 0
            emoji = "🟢" if total_pl >= 0 else "🔴"
            
            message += f"**合计**\n"
            message += f"  总市值: ${total_equity:.2f}\n"
            message += f"  总盈亏: {emoji} ${total_pl:.2f} ({total_pl_pct:+.2f}%)"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in portfolio_command: {e}")
            await update.message.reply_text(f"❌ 加载持仓失败：{str(e)}")
    
    async def watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /watchlist command"""
        try:
            import json
            from pathlib import Path

            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)

            watchlist = portfolio_data.get('watchlist', {}).get('stocks', [])
            
            if not watchlist:
                await update.message.reply_text("👀 你的自选列表为空。")
                return
            
            message = "👀 **自选列表**\n\n"
            
            for i, symbol in enumerate(watchlist, 1):
                try:
                    if self.market_data_provider:
                        latest_price = self.market_data_provider.get_latest_price(symbol)
                        if latest_price is not None:
                            message += f"{i}. **{symbol}** - ${latest_price}\n"
                        else:
                            message += f"{i}. **{symbol}**\n"
                    else:
                        message += f"{i}. **{symbol}**\n"
                except Exception:
                    message += f"{i}. **{symbol}**\n"
            
            message += f"\n**合计：** {len(watchlist)} 支股票"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in watchlist_command: {e}")
            await update.message.reply_text(f"❌ 加载自选失败：{str(e)}")
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command - Full analysis"""
        await update.message.reply_text(
            "🚀 **开始全量分析**\n\n"
            "将分析：\n"
            "1️⃣ 持仓股票\n"
            "2️⃣ 自选股票\n"
            "3️⃣ 发现的热门股票\n\n"
            "⏱️ 预计耗时 20–30 分钟\n"
            "分析进度将逐步发送。"
        )
        
        try:
            if not self.pipeline:
                await update.message.reply_text(
                    "⚠️ 旧版分析管线不可用。请使用 `/analyze_portfolio` 运行全新多智能体工作流。",
                    parse_mode='Markdown'
                )
                return
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=True,
                analyze_watchlist=True,
                analyze_discovered=True,
                top_n_discovered=5
            )
            
            # Send results
            await self._send_recommendations(update, recommendations, "全量分析")
            
        except Exception as e:
            logger.error(f"Error in analyze_command: {e}")
            await update.message.reply_text(f"❌ 分析失败：{str(e)}")
    
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
                f"🤖 **全新：多智能体持仓分析**\n\n"
                f"📊 分析股票数：{portfolio_count}\n"
                f"⏱️ 预计耗时：~{portfolio_count * 25} 秒\n\n"
                f"🔬 运行 8 个专用 AI 智能体：\n"
                f"• 组合加载\n"
                f"• 历史与实时行情\n"
                f"• 新闻与事件\n"
                f"• 公司基本面\n"
                f"• 风险管理\n"
                f"• 深度研究\n\n"
                f"请稍候..."
            )
            
            # Run new workflow
            workflow = PortfolioAnalysisWorkflow()
            result_state = await asyncio.to_thread(workflow.run)
            
            if result_state.get("workflow_error"):
                await update.message.reply_text(f"❌ 分析失败：{result_state['workflow_error']}")
                return
            
            # Format results for Telegram
            await self._send_new_portfolio_results(update, result_state)
            
        except Exception as e:
            logger.error(f"Error in analyze_portfolio_command: {e}")
            await update.message.reply_text(f"❌ 分析失败：{str(e)}")
    
    async def _send_new_portfolio_results(self, update: Update, state: Dict[str, Any]) -> None:
        """Send formatted results from new workflow"""
        summary = state.get("portfolio_summary", {})
        risk = state.get("risk_assessment", {}).get("portfolio_risk", {})
        research = state.get("research_results", {})
        
        # Portfolio Overview
        message = f"""🤖 **KUBERA 组合分析**
━━━━━━━━━━━━━━━━━━━━━━

📊 **组合概览**
• 股票数量：{summary.get('total_stocks', 0)}
• 总市值：${summary.get('total_equity', 0):,.2f}
• 未实现盈亏：${summary.get('unrealized_pl', 0):,.2f}

⚠️ **风险评估**
• 风险等级：{risk.get('portfolio_risk_level', '未知')}
• 风险评分：{risk.get('portfolio_risk_score', 'N/A')}/10
• 分散度：{risk.get('diversification_grade', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━
🔬 **投资建议**
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
            options = result.get("options_strategy", {})
            
            emoji = emoji_map.get(decision, "⚪")
            
            stock_msg = f"""
{emoji} **{symbol}** - {decision}

**信心：** {conviction}/10
"""
            
            # Enhanced price targets display
            if target_price > 0:
                stock_msg += f"🎯 **目标价：** ${target_price:.2f}\n"
            
            # Get stop loss from the research results
            stop_loss = result.get("stop_loss", 0)
            if stop_loss > 0:
                stock_msg += f"🛑 **止损价：** ${stop_loss:.2f}\n"
                
            # Calculate risk/reward
            current_price = result.get("current_price", target_price)  # Fallback to target if not available
            if target_price > 0 and stop_loss > 0 and current_price > 0:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
                stock_msg += f"📊 **风险/回报：** 1:{risk_reward_ratio:.1f} (风险 ${risk:.2f} / 回报 ${reward:.2f})\n"

            if options:
                strat_name = options.get("strategy") or "N/A"
                summary = options.get("summary") or ""
                params = options.get("parameters") or ""
                take_profit_strategy = options.get("take_profit_strategy", "")
                stop_loss_strategy = options.get("stop_loss_strategy", "")
                profit_target_percent = options.get("profit_target_percent", 0)
                loss_limit_percent = options.get("loss_limit_percent", 0)
                
                stock_msg += "\n📈 **期权策略：** " + str(strat_name) + "\n"
                
                # Options take profit and stop loss
                if take_profit_strategy:
                    stock_msg += f"💰 **期权止盈：** {take_profit_strategy}\n"
                if stop_loss_strategy:
                    stock_msg += f"🛡️ **期权止损：** {stop_loss_strategy}\n"
                if profit_target_percent > 0 and loss_limit_percent > 0:
                    stock_msg += f"🎯 **期权目标：** 盈利 {profit_target_percent}% / 亏损限制 {loss_limit_percent}%\n"
                    
                if summary:
                    stock_msg += "**策略理由：** " + str(summary) + "\n"
                if params:
                    stock_msg += "**参数建议：** " + str(params) + "\n"
            
            # Split rationale into chunks if too long (Telegram limit is 4096 chars per message)
            stock_msg += f"\n**详细分析：**\n{rationale}\n"
            stock_msg += "─" * 30
            
            # Send stock message, splitting if too long
            try:
                if len(stock_msg) > 4000:
                    # Split into multiple messages
                    parts = []
                    current_part = f"\n{emoji} **{symbol}** - {decision}\n\n**信心：** {conviction}/10\n"
                    if target_price > 0:
                        current_part += f"🎯 **目标价：** ${target_price:.2f}\n"
                    if stop_loss > 0:
                        current_part += f"🛑 **止损价：** ${stop_loss:.2f}\n"
                    
                    # Add options strategy summary to first part
                    if options:
                        strat_name = options.get("strategy") or "N/A"
                        current_part += f"\n📈 **期权策略：** {str(strat_name)}\n"
                        take_profit_strategy = options.get("take_profit_strategy", "")
                        stop_loss_strategy = options.get("stop_loss_strategy", "")
                        if take_profit_strategy:
                            current_part += f"💰 **期权止盈：** {take_profit_strategy[:100]}...\n"
                        if stop_loss_strategy:
                            current_part += f"🛡️ **期权止损：** {stop_loss_strategy[:100]}...\n"
                    
                    current_part += f"\n**详细分析（第 1 部分）：**\n"
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
                            parts.append(f"**{symbol} 详细分析（第 {part_num + 1} 部分）：**\n{chunk}")
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
        
        final_msg = f"""✅ **分析完成**

⏱️ 耗时：{duration:.1f}s
💡 免责声明：非投资建议，请自行研究。"""
        
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
            
            final_msg += f"\n\n{performance_emoji} **智能体表现：{overall_pct:.0f}%**（{status}）"
        
        await update.message.reply_text(final_msg)

        logger.info("Portfolio analysis results sent to Telegram.")
    
    async def analyze_watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze_watchlist command"""
        try:
            import json
            from pathlib import Path
            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            watchlist_count = len(portfolio_data.get('watchlist', {}).get('stocks', []))
        except Exception:
            watchlist_count = 0
        
        await update.message.reply_text(
            f"🔍 **分析自选列表**\n\n"
            f"待分析股票数：{watchlist_count}\n"
            f"⏱️ 预计耗时：{watchlist_count * 2}-{watchlist_count * 3} 分钟\n\n"
            f"正在进行深度 AI 分析..."
        )
        
        try:
            if not self.pipeline:
                await update.message.reply_text(
                    "⚠️ 旧版分析管线不可用。该命令暂不可用。请使用 `/analyze_portfolio`。",
                    parse_mode='Markdown'
                )
                return
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=False,
                analyze_watchlist=True,
                analyze_discovered=False
            )
            
            await self._send_recommendations(update, recommendations, "自选分析")
            
        except Exception as e:
            logger.error(f"Error in analyze_watchlist_command: {e}")
            await update.message.reply_text(f"❌ 分析失败：{str(e)}")
    
    async def discover_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /discover command"""
        await update.message.reply_text(
            "🚀 **发现热门股票**\n\n"
            "扫描来源：\n"
            "• 📺 YouTube 财经视频\n"
            "• 🐦 X/Twitter 讨论\n"
            "• 📰 财经新闻\n\n"
            "⏱️ 预计耗时 10–15 分钟"
        )
        
        try:
            if not self.pipeline:
                await update.message.reply_text(
                    "⚠️ 旧版分析管线不可用。该命令暂不可用。请使用 `/analyze_portfolio`。",
                    parse_mode='Markdown'
                )
                return
            recommendations = await asyncio.to_thread(
                self.pipeline.run_daily_analysis,
                analyze_portfolio=False,
                analyze_watchlist=False,
                analyze_discovered=True,
                top_n_discovered=5
            )
            
            await self._send_recommendations(update, recommendations, "热门股票发现")
            
        except Exception as e:
            logger.error(f"Error in discover_command: {e}")
            await update.message.reply_text(f"❌ 发现失败：{str(e)}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /status command"""
        try:
            import json
            from pathlib import Path
            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
            with open(portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            portfolio_count = len(portfolio_data.get('portfolio', {}).get('stocks', []))
            watchlist_count = len(portfolio_data.get('watchlist', {}).get('stocks', []))
            
            status_text = f"""
⚙️ **系统状态**

✅ **在线**

**组合：**
• 持仓：{portfolio_count} 支股票
• 自选：{watchlist_count} 支股票

**AI 模型：**
• 阶段 1 分析师：4 个模型
• 阶段 2 辩论：3 个模型
• 发现：2 个模型

**数据来源：**
• 行情：yfinance + Alpha Vantage
• 新闻：Alpha Vantage API
• 发现：YouTube + X/Twitter

**最近更新：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            await update.message.reply_text(status_text, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in status_command: {e}")
            await update.message.reply_text(f"❌ 错误：{str(e)}")
    
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
            await update.message.reply_text(f"✅ **{title} 完成**\n\n暂无推荐结果。")
            return
        
        message = f"✅ **{title} 完成**\n\n"
        message += f"**优选推荐：**\n\n"
        
        for i, rec in enumerate(recommendations[:10], 1):  # Top 10
            symbol = rec['symbol']
            decision = rec.get('final_decision', 'N/A')
            conviction = rec.get('conviction', 0)
            target = rec.get('target_price', 0)
            priority_label = rec.get('priority_label', 'Discovered')
            
            emoji_map = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
            emoji = emoji_map.get(decision, "⚪")
            
            message += f"{i}. {emoji} **{symbol}** - {decision}\n"
            message += f"   优先级：{priority_label}\n"
            message += f"   信心：{conviction}/10\n"
            message += f"   目标价：${target:.2f}\n\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle errors"""
        logger.error(f"Exception while handling an update: {context.error}")
        
        if isinstance(update, Update) and update.effective_message:
            await update.effective_message.reply_text(
                "❌ 发生错误，请稍后重试。"
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
