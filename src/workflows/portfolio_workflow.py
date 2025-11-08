"""
Portfolio Analysis Workflow
Orchestrates all agents using sequential execution pattern
Inspired by CrewAI workflow management
"""

import logging
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

# Import all agents
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.portfolio_loader import PortfolioLoaderAgent
from agents.historical_data import HistoricalDataAgent
from agents.news_fetcher import NewsFetcherAgent
from agents.company_info import CompanyInfoAgent
from agents.sentiment_twitter import TwitterSentimentAgent
from agents.sentiment_reddit import RedditSentimentAgent
from agents.risk_manager import RiskManagerAgent
from agents.researcher import DeepThinkingResearcher

logger = logging.getLogger(__name__)


class PortfolioAnalysisWorkflow:
    """
    Main workflow orchestrator for portfolio analysis
    Coordinates all agents in a sequential, deterministic manner
    """
    
    def __init__(
        self,
        portfolio_path: str = None,
        openrouter_api_key: str = None,
        output_dir: str = None
    ):
        """
        Initialize Portfolio Analysis Workflow
        
        Args:
            portfolio_path: Path to portfolio.json
            openrouter_api_key: OpenRouter API key
            output_dir: Directory for output files
        """
        self.portfolio_path = portfolio_path
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "outputs"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize all agents
        self._initialize_agents()
        
        logger.info("Portfolio Analysis Workflow initialized")
    
    def _initialize_agents(self):
        """Initialize all agents with their specific models"""
        logger.info("Initializing agents...")
        
        try:
            # Agent 1: Portfolio Loader (deepseek/deepseek-chat-v3.1)
            self.portfolio_loader = PortfolioLoaderAgent(
                portfolio_path=self.portfolio_path,
                openrouter_api_key=self.openrouter_api_key,
                model="deepseek/deepseek-chat-v3.1"
            )
            logger.info("✓ Portfolio Loader Agent initialized")
            
            # Agent 2: Historical Data (openai/gpt-oss-20b)
            self.historical_data_agent = HistoricalDataAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="openai/gpt-oss-20b"
            )
            logger.info("✓ Historical Data Agent initialized")
            
            # Agent 3: News Fetcher (google/gemini-2.0-flash-exp)
            self.news_fetcher = NewsFetcherAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="google/gemini-2.0-flash-exp"
            )
            logger.info("✓ News Fetcher Agent initialized")
            
            # Agent 4: Company Info (deepseek/deepseek-r1-distill-llama-70b)
            self.company_info_agent = CompanyInfoAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="deepseek/deepseek-r1-distill-llama-70b"
            )
            logger.info("✓ Company Info Agent initialized")
            
            # Agent 5: Twitter Sentiment (deepseek/deepseek-chat-v3.1)
            self.twitter_agent = TwitterSentimentAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="deepseek/deepseek-chat-v3.1"
            )
            logger.info("✓ Twitter Sentiment Agent initialized")
            
            # Agent 6: Reddit Sentiment (deepseek/deepseek-chat-v3.1)
            self.reddit_agent = RedditSentimentAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="deepseek/deepseek-chat-v3.1"
            )
            logger.info("✓ Reddit Sentiment Agent initialized")
            
            # Agent 7: Risk Manager (deepseek/deepseek-r1-0528)
            self.risk_manager = RiskManagerAgent(
                openrouter_api_key=self.openrouter_api_key,
                model="deepseek/deepseek-r1-0528"
            )
            logger.info("✓ Risk Manager Agent initialized")
            
            # Agent 8: Deep Thinking Researcher (qwen/qwen3-235b-a22b)
            self.researcher = DeepThinkingResearcher(
                openrouter_api_key=self.openrouter_api_key,
                model="qwen/qwen3-235b-a22b"
            )
            logger.info("✓ Deep Thinking Researcher initialized")
            
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete workflow
        
        Returns:
            Dictionary with all results and final recommendations
        """
        logger.info("\n" + "=" * 70)
        logger.info("🚀 PORTFOLIO ANALYSIS WORKFLOW - STARTING")
        logger.info("=" * 70 + "\n")
        
        start_time = datetime.now()
        workflow_results = {}
        
        try:
            # STAGE 1: Load Portfolio
            logger.info("📂 STAGE 1: Loading Portfolio...")
            portfolio_data = self.portfolio_loader.run()
            workflow_results['portfolio_data'] = portfolio_data
            logger.info(f"✅ Portfolio loaded: {portfolio_data['total_holdings']} stocks\n")
            
            # STAGE 2: Fetch Real-Time Data & Compare
            logger.info("📊 STAGE 2: Fetching Real-Time Market Data...")
            historical_data = self.historical_data_agent.run(portfolio_data)
            workflow_results['historical_data'] = historical_data
            logger.info(f"✅ Market data fetched for {historical_data['total_holdings']} stocks\n")
            
            # STAGE 3: Fetch News & Events
            logger.info("📰 STAGE 3: Fetching News & Events...")
            news_data = self.news_fetcher.run(portfolio_data)
            workflow_results['news_data'] = news_data
            logger.info(f"✅ News fetched for {news_data['total_stocks']} stocks\n")
            
            # STAGE 4: Fetch Company Fundamentals
            logger.info("💼 STAGE 4: Analyzing Company Fundamentals...")
            company_data = self.company_info_agent.run(portfolio_data)
            workflow_results['company_data'] = company_data
            logger.info(f"✅ Fundamentals analyzed for {company_data['total_stocks']} stocks\n")
            
            # STAGE 5: Twitter Sentiment
            logger.info("🐦 STAGE 5: Analyzing Twitter Sentiment...")
            twitter_data = self.twitter_agent.run(portfolio_data)
            workflow_results['twitter_data'] = twitter_data
            logger.info(f"✅ Twitter sentiment analyzed for {twitter_data['total_stocks']} stocks\n")
            
            # STAGE 6: Reddit Sentiment
            logger.info("🤖 STAGE 6: Analyzing Reddit Sentiment...")
            reddit_data = self.reddit_agent.run(portfolio_data)
            workflow_results['reddit_data'] = reddit_data
            logger.info(f"✅ Reddit sentiment analyzed for {reddit_data['total_stocks']} stocks\n")
            
            # STAGE 7: Risk Analysis
            logger.info("⚠️  STAGE 7: Performing Risk Analysis...")
            risk_data = self.risk_manager.run(historical_data)
            workflow_results['risk_data'] = risk_data
            logger.info(f"✅ Risk analysis complete\n")
            
            # STAGE 8: Deep Research & Final Recommendations
            logger.info("🧠 STAGE 8: Generating Final Investment Recommendations...")
            all_agent_data = {
                'symbols': portfolio_data['symbols'],
                'historical_data': historical_data,
                'news_data': news_data,
                'company_data': company_data,
                'twitter_data': twitter_data,
                'reddit_data': reddit_data,
                'risk_data': risk_data
            }
            final_recommendations = self.researcher.run(all_agent_data)
            workflow_results['final_recommendations'] = final_recommendations
            logger.info(f"✅ Final recommendations generated for {final_recommendations['total_stocks_analyzed']} stocks\n")
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Save results
            output_file = self._save_results(workflow_results)
            
            # Summary
            logger.info("\n" + "=" * 70)
            logger.info("✅ WORKFLOW COMPLETE")
            logger.info("=" * 70)
            logger.info(f"Total Execution Time: {execution_time:.2f} seconds")
            logger.info(f"Stocks Analyzed: {final_recommendations['total_stocks_analyzed']}")
            logger.info(f"Output Saved: {output_file}")
            logger.info("=" * 70 + "\n")
            
            return {
                "status": "success",
                "execution_time": execution_time,
                "workflow_results": workflow_results,
                "output_file": str(output_file)
            }
        
        except Exception as e:
            logger.error(f"\n❌ WORKFLOW FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "workflow_results": workflow_results
            }
    
    def _save_results(self, results: Dict[str, Any]) -> Path:
        """Save workflow results to JSON file"""
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"portfolio_analysis_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    def get_telegram_summary(self, workflow_results: Dict[str, Any]) -> str:
        """
        Generate detailed Telegram-formatted summary
        
        Args:
            workflow_results: Complete workflow results
            
        Returns:
            Formatted string for Telegram
        """
        final_recs = workflow_results.get('final_recommendations', {})
        recommendations = final_recs.get('recommendations', [])
        
        summary = "📊 **PORTFOLIO ANALYSIS COMPLETE**\n\n"
        
        # Portfolio summary
        risk_data = workflow_results.get('risk_data', {})
        portfolio_risk = risk_data.get('portfolio_risk', {})
        
        summary += f"**Portfolio Overview:**\n"
        summary += f"• Total Value: ${portfolio_risk.get('total_portfolio_value', 0):.2f}\n"
        summary += f"• Risk Level: {portfolio_risk.get('overall_risk_level', 'N/A').upper()}\n"
        summary += f"• Diversification Score: {portfolio_risk.get('diversification_score', 0):.0f}/100\n\n"
        
        # Recommendations by decision
        buy_recs = [r for r in recommendations if r['decision'] == 'BUY']
        hold_recs = [r for r in recommendations if r['decision'] == 'HOLD']
        sell_recs = [r for r in recommendations if r['decision'] == 'SELL']
        
        summary += f"**Recommendations:**\n"
        summary += f"🟢 BUY: {len(buy_recs)} stocks\n"
        summary += f"🟡 HOLD: {len(hold_recs)} stocks\n"
        summary += f"🔴 SELL: {len(sell_recs)} stocks\n\n"
        
        # Detailed recommendations
        summary += "=" * 40 + "\n\n"
        
        for rec in recommendations:
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(rec['decision'], "⚪")
            
            summary += f"{emoji} **{rec['symbol']} - {rec['decision']}**\n"
            summary += f"Conviction: {rec['conviction']}/10\n"
            summary += f"Target: ${rec['target_price']:.2f} | Stop: ${rec['stop_loss']:.2f}\n\n"
            
            summary += f"**Executive Summary:**\n{rec['executive_summary']}\n\n"
            
            summary += f"**Bull Case:**\n{rec['bull_case'][:200]}...\n\n"
            summary += f"**Bear Case:**\n{rec['bear_case'][:200]}...\n\n"
            
            summary += f"**Action Plan:**\n{rec['action_plan']}\n\n"
            summary += "=" * 40 + "\n\n"
        
        return summary


# Main execution function
def main():
    """Main execution function for testing"""
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("❌ OPENROUTER_API_KEY not found in environment")
        logger.error("Please set it in your .env file:")
        logger.error("  OPENROUTER_API_KEY=your_key_here")
        return
    
    try:
        # Create and run workflow
        workflow = PortfolioAnalysisWorkflow()
        results = workflow.run()
        
        if results['status'] == 'success':
            logger.info("\n✅ Workflow executed successfully!")
            logger.info(f"Results saved to: {results['output_file']}")
            
            # Generate Telegram summary
            telegram_summary = workflow.get_telegram_summary(results['workflow_results'])
            
            # Save Telegram summary
            summary_file = Path(results['output_file']).parent / f"telegram_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_file, 'w') as f:
                f.write(telegram_summary)
            logger.info(f"Telegram summary saved to: {summary_file}")
        else:
            logger.error("\n❌ Workflow failed!")
    
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

