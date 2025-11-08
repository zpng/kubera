"""
Main Portfolio Analysis Workflow
Orchestrates all agents using LangChain and CrewAI patterns
"""
import logging
from typing import Dict, Any
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..agents import (
    PortfolioLoaderAgent,
    HistoricalDataAgent,
    NewsFetcherAgent,
    CompanyInfoAgent,
    TwitterSentimentAgent,
    RedditSentimentAgent,
    RiskManagerAgent,
    DeepResearcherAgent
)
from ..agents.agent_evaluator import AgentEvaluator
from ..agents.fact_checker import FactCheckerAgent
from ..config import WORKFLOW_CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)


class PortfolioAnalysisWorkflow:
    """
    Main workflow orchestrator for portfolio analysis
    
    Workflow Steps:
    1. Load Portfolio (Sequential)
    2. Data Collection (Parallel): Historical, News, Company Info, Social Sentiment
    3. Risk Assessment (Sequential)
    4. Deep Research & Decision (Sequential)
    5. Output Generation (Sequential)
    """
    
    def __init__(self):
        """Initialize workflow with all agents"""
        logger.info("Initializing Portfolio Analysis Workflow...")
        
        # Stage 1: Portfolio Loading
        self.portfolio_loader = PortfolioLoaderAgent()
        
        # Stage 2: Data Collection (can run in parallel)
        self.historical_data = HistoricalDataAgent()
        self.news_fetcher = NewsFetcherAgent()
        self.company_info = CompanyInfoAgent()
        self.twitter_sentiment = TwitterSentimentAgent()
        self.reddit_sentiment = RedditSentimentAgent()
        
        # Stage 3: Risk Assessment
        self.risk_manager = RiskManagerAgent()
        
        # Stage 4: Deep Research & Decision Making
        self.deep_researcher = DeepResearcherAgent()
        
        # Validators
        self.evaluator = AgentEvaluator()
        self.fact_checker = FactCheckerAgent()
        
        self.parallel_execution = WORKFLOW_CONFIG.get("parallel_execution", True)
        
        logger.info("✅ Workflow initialized with 8 agents + evaluator + fact checker")
    
    def run_data_collection_parallel(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run data collection agents in parallel for efficiency
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state with all data collected
        """
        logger.info("Starting parallel data collection...")
        
        agents = [
            ("Historical Data", self.historical_data),
            ("News", self.news_fetcher),
            ("Company Info", self.company_info),
            ("Twitter Sentiment", self.twitter_sentiment),
            ("Reddit Sentiment", self.reddit_sentiment)
        ]
        
        if not self.parallel_execution:
            # Sequential execution
            for name, agent in agents:
                logger.info(f"Running {name} agent...")
                state = agent.process(state)
            return state
        
        # Parallel execution
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all agents
            future_to_agent = {
                executor.submit(agent.process, state.copy()): name
                for name, agent in agents
            }
            
            # Collect results
            results = {}
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    results[agent_name] = result
                    logger.info(f"✅ {agent_name} completed")
                except Exception as e:
                    logger.error(f"❌ {agent_name} failed: {e}")
                    results[agent_name] = state.copy()  # Use original state on failure
            
            # Merge all results into state
            for agent_name, result in results.items():
                state.update(result)
        
        logger.info("✅ Parallel data collection complete")
        return state
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete workflow
        
        Returns:
            Final state with all analysis results
        """
        logger.info("=" * 60)
        logger.info("STARTING PORTFOLIO ANALYSIS WORKFLOW")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        state = {
            "workflow_start_time": start_time.isoformat(),
            "errors": []
        }
        
        try:
            # Stage 1: Load Portfolio
            logger.info("\n📂 STAGE 1: Loading Portfolio...")
            state = self.portfolio_loader.process(state)
            
            if not state.get("portfolio_loaded"):
                logger.error("❌ Portfolio loading failed. Aborting workflow.")
                return state
            
            logger.info(f"✅ Loaded {len(state.get('stock_symbols', []))} stocks from portfolio")
            
            # Stage 2: Data Collection (Parallel)
            logger.info("\n📊 STAGE 2: Collecting Data from Multiple Sources...")
            state = self.run_data_collection_parallel(state)
            
            # Stage 3: Risk Assessment
            logger.info("\n⚠️  STAGE 3: Assessing Portfolio Risk...")
            state = self.risk_manager.process(state)
            
            portfolio_risk = state.get("risk_assessment", {}).get("portfolio_risk", {})
            logger.info(f"✅ Risk Assessment Complete. Portfolio Risk: {portfolio_risk.get('portfolio_risk_level', 'Unknown')}")
            
            # Stage 4: Deep Research & Decision Making
            logger.info("\n🔬 STAGE 4: Conducting Deep Research & Making Decisions...")
            state = self.deep_researcher.process(state)
            
            research_results = state.get("research_results", {})
            logger.info(f"✅ Research Complete for {len(research_results)} stocks")
            
            # Stage 5: Fact Checking
            logger.info("\n🔍 STAGE 5: Fact Checking Research Outputs...")
            fact_check_results = self.fact_checker.validate_all_research(research_results, state)
            state["fact_check_results"] = fact_check_results
            
            if not fact_check_results["all_valid"]:
                logger.warning(f"⚠️  Found {fact_check_results['total_errors']} potential hallucinations")
            else:
                logger.info("✅ All research outputs validated - no hallucinations detected")
            
            # Stage 6: Agent Evaluation
            logger.info("\n📊 STAGE 6: Evaluating Agent Performance...")
            evaluation_results = self.evaluator.evaluate_workflow(state)
            state["agent_evaluation"] = evaluation_results
            
            # Calculate workflow duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            state["workflow_end_time"] = end_time.isoformat()
            state["workflow_duration"] = duration
            
            logger.info("\n" + "=" * 60)
            logger.info(f"✅ WORKFLOW COMPLETED SUCCESSFULLY in {duration:.1f} seconds")
            logger.info("=" * 60)
            
            return state
            
        except Exception as e:
            logger.error(f"❌ Workflow failed with error: {e}", exc_info=True)
            state["workflow_error"] = str(e)
            return state
    
    def save_results(self, state: Dict[str, Any], filename: str = None) -> str:
        """
        Save workflow results to file
        
        Args:
            state: Workflow state with results
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"portfolio_analysis_{timestamp}.json"
        
        filepath = OUTPUT_DIR / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"✅ Results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return ""
    
    def format_results_for_telegram(self, state: Dict[str, Any]) -> str:
        """
        Format results for Telegram output with detailed information
        
        Args:
            state: Workflow state with all results
            
        Returns:
            Formatted message string for Telegram
        """
        message = "📊 **PORTFOLIO ANALYSIS COMPLETE**\n\n"
        
        # Portfolio Summary
        summary = state.get("portfolio_summary", {})
        message += f"💼 **Portfolio Overview**\n"
        message += f"Total Stocks: {summary.get('total_stocks', 0)}\n"
        message += f"Total Value: ${summary.get('total_equity', 0):,.2f}\n"
        message += f"Total P&L: ${summary.get('unrealized_pl', 0):,.2f}\n\n"
        
        # Portfolio Risk
        portfolio_risk = state.get("risk_assessment", {}).get("portfolio_risk", {})
        message += f"⚠️  **Risk Assessment**\n"
        message += f"Risk Level: {portfolio_risk.get('portfolio_risk_level', 'N/A')}\n"
        message += f"Risk Score: {portfolio_risk.get('portfolio_risk_score', 'N/A')}/10\n"
        message += f"Value at Risk (1-day): ${portfolio_risk.get('portfolio_var_1day', 0):,.2f}\n"
        message += f"Diversification Grade: {portfolio_risk.get('diversification_grade', 'N/A')}\n\n"
        
        # Research Results - Detailed for each stock
        research_results = state.get("research_results", {})
        
        if research_results:
            message += "🔬 **INVESTMENT RECOMMENDATIONS**\n\n"
            
            for symbol, result in research_results.items():
                decision = result.get("decision", "HOLD")
                conviction = result.get("conviction", 5)
                target_price = result.get("target_price", 0)
                
                # Emoji based on decision
                emoji_map = {
                    "BUY MORE": "🟢",
                    "HOLD": "🟡",
                    "SELL": "🔴",
                    "TRIM POSITION": "🟠"
                }
                emoji = emoji_map.get(decision, "⚪")
                
                message += f"{emoji} **{symbol}** - {decision}\n"
                message += f"Conviction: {conviction}/10\n"
                if target_price > 0:
                    message += f"Target Price: ${target_price:.2f}\n"
                message += f"\n"
                
                # Add truncated detailed analysis
                analysis = result.get("detailed_analysis", "")
                if analysis:
                    # Take first 200 characters of analysis
                    preview = analysis[:200] + "..." if len(analysis) > 200 else analysis
                    message += f"{preview}\n"
                
                message += "\n" + "-" * 40 + "\n\n"
        
        # Workflow metadata
        duration = state.get("workflow_duration_seconds", 0)
        message += f"\n⏱️ Analysis completed in {duration:.1f} seconds"
        
        return message


# Test function
def test_workflow():
    """Test the complete workflow"""
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO ANALYSIS WORKFLOW")
    print("=" * 60 + "\n")
    
    # Initialize workflow
    workflow = PortfolioAnalysisWorkflow()
    
    # Run workflow
    result_state = workflow.run()
    
    # Print summary
    print("\n" + "=" * 60)
    print("WORKFLOW RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"\nPortfolio: {len(result_state.get('stock_symbols', []))} stocks")
    print(f"Duration: {result_state.get('workflow_duration_seconds', 0):.1f} seconds")
    
    # Print research recommendations
    print("\n📊 Investment Recommendations:")
    research_results = result_state.get("research_results", {})
    for symbol, result in research_results.items():
        print(f"\n{symbol}:")
        print(f"  Decision: {result.get('decision', 'N/A')}")
        print(f"  Conviction: {result.get('conviction', 'N/A')}/10")
        print(f"  Target: ${result.get('target_price', 0):.2f}")
    
    # Save results
    filepath = workflow.save_results(result_state)
    print(f"\n✅ Results saved to: {filepath}")
    
    # Format for Telegram
    telegram_message = workflow.format_results_for_telegram(result_state)
    print("\n" + "=" * 60)
    print("TELEGRAM MESSAGE PREVIEW")
    print("=" * 60)
    print(telegram_message)
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_workflow()

