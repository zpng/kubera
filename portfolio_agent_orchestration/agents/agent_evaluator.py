"""
Agent Evaluator
Monitors and evaluates the performance of all agents in the workflow
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """
    Evaluates agent performance based on output quality and completeness
    """
    
    def __init__(self):
        self.evaluation_results = {}
        self.thresholds = {
            "data_completeness": 0.7,  # 70% completeness required
            "processing_time": 60,  # Max 60 seconds per agent
            "error_rate": 0.1  # Max 10% error rate
        }
    
    def evaluate_portfolio_loader(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Portfolio Loader Agent"""
        score = 10.0
        issues = []
        
        portfolio_data = state.get("portfolio_data", {})
        stock_symbols = state.get("stock_symbols", [])
        
        # Check if portfolio was loaded
        if not portfolio_data:
            score -= 5.0
            issues.append("Portfolio data not loaded")
        
        # Check if symbols extracted
        if not stock_symbols:
            score -= 3.0
            issues.append("No stock symbols extracted")
        elif len(stock_symbols) < 1:
            score -= 2.0
            issues.append("Insufficient stocks in portfolio")
        
        # Check portfolio summary
        summary = state.get("portfolio_data", {}).get("summary", {})
        if not summary:
            score -= 2.0
            issues.append("Portfolio summary missing")
        
        return {
            "agent": "PortfolioLoader",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 7.0 else "FAIL",
            "issues": issues,
            "stocks_loaded": len(stock_symbols)
        }
    
    def evaluate_historical_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Historical Data Agent"""
        score = 10.0
        issues = []
        
        historical_data = state.get("historical_data", {})
        stock_symbols = state.get("stock_symbols", [])
        
        # Check if data exists for all symbols
        missing_symbols = [s for s in stock_symbols if s not in historical_data]
        if missing_symbols:
            score -= 3.0
            issues.append(f"Missing data for: {', '.join(missing_symbols)}")
        
        # Check data completeness for each symbol
        incomplete = []
        for symbol in stock_symbols:
            data = historical_data.get(symbol, {})
            if not data.get("metrics") or not data.get("trends"):
                incomplete.append(symbol)
        
        if incomplete:
            score -= 2.0 * (len(incomplete) / len(stock_symbols))
            issues.append(f"Incomplete data for: {', '.join(incomplete)}")
        
        return {
            "agent": "HistoricalData",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 7.0 else "FAIL",
            "issues": issues,
            "symbols_processed": len(historical_data),
            "expected_symbols": len(stock_symbols)
        }
    
    def evaluate_news_fetcher(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate News Fetcher Agent"""
        score = 10.0
        issues = []
        
        news_data = state.get("news_data", {})
        stock_symbols = state.get("stock_symbols", [])
        
        # Check coverage
        coverage = len(news_data) / len(stock_symbols) if stock_symbols else 0
        
        # Check news quality
        symbols_with_news = 0
        for symbol in stock_symbols:
            news = news_data.get(symbol, {})
            if news.get("articles") or news.get("sentiment_analysis"):
                symbols_with_news += 1
        
        # News is optional but improves analysis
        news_rate = symbols_with_news / len(stock_symbols) if stock_symbols else 0
        if news_rate < 0.3:
            score -= 3.0
            issues.append(f"Low news coverage: {news_rate*100:.0f}%")
        
        return {
            "agent": "NewsFetcher",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 6.0 else "WARNING",  # Lower threshold for news
            "issues": issues,
            "coverage": f"{news_rate*100:.0f}%",
            "symbols_with_news": symbols_with_news
        }
    
    def evaluate_company_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Company Info Agent"""
        score = 10.0
        issues = []
        
        company_data = state.get("company_data", {})
        stock_symbols = state.get("stock_symbols", [])
        
        # Check if data exists for all symbols
        missing_symbols = [s for s in stock_symbols if s not in company_data]
        if missing_symbols:
            score -= 4.0
            issues.append(f"Missing data for: {', '.join(missing_symbols)}")
        
        # Check data quality
        incomplete = []
        for symbol in stock_symbols:
            data = company_data.get(symbol, {})
            if not data.get("company_info") or not data.get("financial_metrics"):
                incomplete.append(symbol)
        
        if incomplete:
            score -= 3.0 * (len(incomplete) / len(stock_symbols))
            issues.append(f"Incomplete data for: {', '.join(incomplete)}")
        
        return {
            "agent": "CompanyInfo",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 7.0 else "FAIL",
            "issues": issues,
            "symbols_processed": len(company_data),
            "expected_symbols": len(stock_symbols)
        }
    
    def evaluate_sentiment_agents(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Social Sentiment Agents (Twitter & Reddit)"""
        score = 10.0
        issues = []
        
        twitter_data = state.get("twitter_sentiment", {})
        reddit_data = state.get("reddit_sentiment", {})
        stock_symbols = state.get("stock_symbols", [])
        
        # Sentiment is nice-to-have but not critical
        twitter_coverage = len(twitter_data) / len(stock_symbols) if stock_symbols else 0
        reddit_coverage = len(reddit_data) / len(stock_symbols) if stock_symbols else 0
        
        avg_coverage = (twitter_coverage + reddit_coverage) / 2
        
        if avg_coverage < 0.5:
            score -= 2.0
            issues.append(f"Low sentiment coverage: {avg_coverage*100:.0f}%")
        
        return {
            "agent": "SocialSentiment",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 6.0 else "WARNING",
            "issues": issues,
            "twitter_coverage": f"{twitter_coverage*100:.0f}%",
            "reddit_coverage": f"{reddit_coverage*100:.0f}%"
        }
    
    def evaluate_risk_manager(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Risk Manager Agent"""
        score = 10.0
        issues = []
        
        risk_data = state.get("risk_assessment", {})
        
        if not risk_data:
            score -= 5.0
            issues.append("Risk assessment missing")
        else:
            # Check for position risks
            positions_risk = risk_data.get("positions_risk", [])
            if not positions_risk:
                score -= 3.0
                issues.append("Position risk data missing")
            
            # Check for portfolio risk
            portfolio_risk = risk_data.get("portfolio_risk", {})
            if not portfolio_risk:
                score -= 2.0
                issues.append("Portfolio risk metrics missing")
        
        return {
            "agent": "RiskManager",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 7.0 else "FAIL",
            "issues": issues
        }
    
    def evaluate_deep_researcher(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Deep Researcher Agent - MOST CRITICAL"""
        score = 10.0
        issues = []
        
        research_results = state.get("research_results", {})
        stock_symbols = state.get("stock_symbols", [])
        
        if not research_results:
            score = 0
            issues.append("CRITICAL: No research results generated")
            return {
                "agent": "DeepResearcher",
                "score": 0,
                "max_score": 10.0,
                "status": "FAIL",
                "issues": issues,
                "critical": True
            }
        
        # Check completeness
        missing_symbols = [s for s in stock_symbols if s not in research_results]
        if missing_symbols:
            score -= 3.0
            issues.append(f"Missing analysis for: {', '.join(missing_symbols)}")
        
        # Check quality of analysis - MORE STRICT
        low_quality = []
        missing_rationale = []
        short_analysis = []
        
        for symbol, result in research_results.items():
            # Check for rationale - NOW ENFORCING MINIMUM LENGTH
            rationale = result.get("rationale", "")
            rationale_length = len(rationale)
            
            if not rationale or rationale == "No analysis available":
                missing_rationale.append(symbol)
                score -= 3.0  # Severe penalty for missing analysis
            elif rationale_length < 300:
                short_analysis.append(f"{symbol} ({rationale_length} chars)")
                score -= 1.5  # Penalty for short analysis
            elif rationale_length < 500:
                short_analysis.append(f"{symbol} ({rationale_length} chars)")
                score -= 0.5  # Minor penalty
            
            # Check conviction
            conviction = result.get("conviction", 0)
            if conviction < 1 or conviction > 10:
                low_quality.append(symbol)
                score -= 1.0
        
        if missing_rationale:
            issues.append(f"CRITICAL: Missing rationale for: {', '.join(missing_rationale)}")
        
        if short_analysis:
            issues.append(f"Short analysis (<500 words): {', '.join(short_analysis)}")
        
        if low_quality:
            issues.append(f"Invalid conviction scores: {', '.join(low_quality)}")
        
        return {
            "agent": "DeepResearcher",
            "score": max(score, 0),
            "max_score": 10.0,
            "status": "PASS" if score >= 8.0 else "FAIL",  # Higher threshold for research
            "issues": issues,
            "symbols_analyzed": len(research_results),
            "expected_symbols": len(stock_symbols),
            "missing_rationale_count": len(missing_rationale),
            "critical": len(missing_rationale) > 0
        }
    
    def evaluate_workflow(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate entire workflow and all agents"""
        logger.info("=" * 60)
        logger.info("AGENT PERFORMANCE EVALUATION")
        logger.info("=" * 60)
        
        evaluations = []
        
        # Evaluate each agent
        evaluations.append(self.evaluate_portfolio_loader(state))
        evaluations.append(self.evaluate_historical_data(state))
        evaluations.append(self.evaluate_news_fetcher(state))
        evaluations.append(self.evaluate_company_info(state))
        evaluations.append(self.evaluate_sentiment_agents(state))
        evaluations.append(self.evaluate_risk_manager(state))
        evaluations.append(self.evaluate_deep_researcher(state))
        
        # Calculate overall score
        total_score = sum(e["score"] for e in evaluations)
        max_total = sum(e["max_score"] for e in evaluations)
        overall_pct = (total_score / max_total * 100) if max_total > 0 else 0
        
        # Print results
        for eval_result in evaluations:
            agent = eval_result["agent"]
            score = eval_result["score"]
            max_score = eval_result["max_score"]
            status = eval_result["status"]
            issues = eval_result["issues"]
            
            status_emoji = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
            
            logger.info(f"\n{status_emoji} {agent}: {score:.1f}/{max_score} ({status})")
            if issues:
                for issue in issues:
                    logger.info(f"   - {issue}")
            
            # Log additional metrics
            for key, value in eval_result.items():
                if key not in ["agent", "score", "max_score", "status", "issues", "critical"]:
                    logger.info(f"   • {key}: {value}")
        
        # Overall assessment
        logger.info("\n" + "=" * 60)
        logger.info(f"OVERALL SCORE: {total_score:.1f}/{max_total} ({overall_pct:.1f}%)")
        
        if overall_pct >= 80:
            logger.info("✅ WORKFLOW STATUS: EXCELLENT")
        elif overall_pct >= 70:
            logger.info("✅ WORKFLOW STATUS: GOOD")
        elif overall_pct >= 60:
            logger.info("⚠️  WORKFLOW STATUS: ACCEPTABLE")
        else:
            logger.info("❌ WORKFLOW STATUS: NEEDS IMPROVEMENT")
        
        logger.info("=" * 60)
        
        # Store results
        self.evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "agent_evaluations": evaluations,
            "overall_score": total_score,
            "max_score": max_total,
            "percentage": overall_pct,
            "status": "PASS" if overall_pct >= 70 else "FAIL"
        }
        
        return self.evaluation_results

