"""
Autonomous Trading Pipeline
Discovers trending stocks and runs complete analysis pipeline autonomously
Prioritizes: 1) Portfolio stocks, 2) Watchlist stocks, 3) Discovered stocks
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import json
from pathlib import Path

from orchestrator.stock_discovery_orchestrator import StockDiscoveryOrchestrator

# Import Stage 1 analysts
from agents.analysts import (
    MarketAnalyst,
    NewsAnalyst,
    SentimentAnalyst,
    FundamentalsAnalyst
)

# Import Stage 2 researchers
from agents.researchers import (
    BullResearcher,
    BearResearcher,
    ResearchJudge
)

# Import data layer
from data import get_stock_data, get_news, get_fundamentals, TechnicalIndicators
from data.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class AutonomousTradingPipeline:
    """
    Autonomous pipeline that:
    1. Analyzes user's portfolio stocks (PRIORITY 1)
    2. Analyzes user's watchlist stocks (PRIORITY 2)
    3. Discovers trending stocks from YouTube/X/News (PRIORITY 3)
    4. Runs full 2-stage analysis on all
    5. Generates investment recommendations
    """

    def __init__(self, portfolio_path: Optional[str] = None):
        # Discovery
        self.discovery = StockDiscoveryOrchestrator()

        # Stage 1: Analysts
        self.market_analyst = MarketAnalyst()
        self.news_analyst = NewsAnalyst()
        self.sentiment_analyst = SentimentAnalyst()
        self.fundamentals_analyst = FundamentalsAnalyst()

        # Stage 2: Researchers
        self.bull_researcher = BullResearcher()
        self.bear_researcher = BearResearcher()
        self.research_judge = ResearchJudge()

        # Data provider
        self.market_data = MarketDataProvider()
        self.indicators_calc = TechnicalIndicators()

        # Load portfolio and watchlist
        self.portfolio_config = self._load_portfolio(portfolio_path)
        
        logger.info("Autonomous Trading Pipeline initialized")
        logger.info(f"Portfolio: {len(self.portfolio_config.get('portfolio', {}).get('stocks', []))} holdings")
        logger.info(f"Watchlist: {len(self.portfolio_config.get('watchlist', {}).get('stocks', []))} stocks")
    
    def _load_portfolio(self, portfolio_path: Optional[str] = None) -> Dict[str, Any]:
        """Load portfolio configuration from JSON file."""
        if portfolio_path is None:
            # Default to config/portfolio.json
            portfolio_path = Path(__file__).parent.parent.parent / "config" / "portfolio.json"
        
        try:
            with open(portfolio_path, 'r') as f:
                portfolio = json.load(f)
            logger.info(f"Loaded portfolio from {portfolio_path}")
            return portfolio
        except FileNotFoundError:
            logger.warning(f"Portfolio file not found: {portfolio_path}")
            return {"portfolio": {"stocks": []}, "watchlist": {"stocks": []}}
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            return {"portfolio": {"stocks": []}, "watchlist": {"stocks": []}}

    def run_daily_analysis(
        self,
        analyze_portfolio: bool = True,
        analyze_watchlist: bool = True,
        analyze_discovered: bool = True,
        top_n_discovered: int = 5,
        min_discovery_score: float = 10.0
    ) -> List[Dict[str, Any]]:
        """
        Run complete autonomous daily analysis with prioritization.

        Priority:
        1. Portfolio stocks (current holdings - HIGHEST PRIORITY)
        2. Watchlist stocks (future candidates - SECOND PRIORITY)
        3. Discovered stocks (YouTube/X/News - THIRD PRIORITY)

        Args:
            analyze_portfolio: Whether to analyze portfolio holdings
            analyze_watchlist: Whether to analyze watchlist stocks
            analyze_discovered: Whether to discover and analyze new stocks
            top_n_discovered: Number of top discovered stocks to analyze
            min_discovery_score: Minimum discovery score for discovered stocks

        Returns:
            List of investment recommendations sorted by priority
        """
        try:
            start_time = datetime.now()

            print("\n" + "="*80)
            print("🤖 AUTONOMOUS TRADING PIPELINE - DAILY RUN")
            print("="*80)
            print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)
            
            # Build prioritized stock list
            analysis_queue = []
            
            # PRIORITY 1: Portfolio Holdings
            portfolio_stocks = self.portfolio_config.get('portfolio', {}).get('stocks', [])
            if analyze_portfolio and portfolio_stocks:
                print("\n💼 PRIORITY 1: PORTFOLIO HOLDINGS")
                print(f"   Analyzing {len(portfolio_stocks)} current holdings")
                for holding in portfolio_stocks:
                    analysis_queue.append({
                        'symbol': holding['symbol'],
                        'priority': 1,
                        'priority_label': 'PORTFOLIO',
                        'score': 100.0,  # Max score for portfolio
                        'sources': ['Portfolio'],
                        'overall_sentiment': 'neutral',
                        'context': f"Current holding: {holding['shares']} shares @ ${holding['avg_cost']:.2f}"
                    })
            
            # PRIORITY 2: Watchlist
            watchlist_stocks = self.portfolio_config.get('watchlist', {}).get('stocks', [])
            if analyze_watchlist and watchlist_stocks:
                print(f"\n⭐ PRIORITY 2: WATCHLIST")
                print(f"   Analyzing {len(watchlist_stocks)} watchlist stocks")
                for symbol in watchlist_stocks:
                    # Skip if already in portfolio
                    if symbol not in [h['symbol'] for h in portfolio_stocks]:
                        analysis_queue.append({
                            'symbol': symbol,
                            'priority': 2,
                            'priority_label': 'WATCHLIST',
                            'score': 50.0,  # High score for watchlist
                            'sources': ['Watchlist'],
                            'overall_sentiment': 'neutral',
                            'context': 'Future investment candidate'
                        })
            
            # PRIORITY 3: Discovered Stocks
            if analyze_discovered:
                print("\n📡 PRIORITY 3: DISCOVERING TRENDING STOCKS...")
                trending_stocks = self.discovery.discover_trending_stocks(top_n=top_n_discovered)
                
                # Filter by minimum score and exclude portfolio/watchlist
                existing_symbols = set(
                    [h['symbol'] for h in portfolio_stocks] +
                    watchlist_stocks
                )
                
                for stock in trending_stocks:
                    if (stock['score'] >= min_discovery_score and 
                        stock['symbol'] not in existing_symbols):
                        analysis_queue.append({
                            'symbol': stock['symbol'],
                            'priority': 3,
                            'priority_label': 'DISCOVERED',
                            'score': stock['score'],
                            'sources': stock['sources'],
                            'overall_sentiment': stock['overall_sentiment'],
                            'context': f"Discovered from {', '.join(stock['sources'])}"
                        })
                
                discovered_count = len([s for s in analysis_queue if s['priority'] == 3])
                print(f"   Found {len(trending_stocks)} stocks, {discovered_count} meet criteria")
            
            # Sort by priority (1 first, then 2, then 3)
            analysis_queue.sort(key=lambda x: x['priority'])
            
            total_stocks = len(analysis_queue)
            print(f"\n📊 TOTAL STOCKS TO ANALYZE: {total_stocks}")
            print(f"   • Portfolio: {len([s for s in analysis_queue if s['priority'] == 1])}")
            print(f"   • Watchlist: {len([s for s in analysis_queue if s['priority'] == 2])}")
            print(f"   • Discovered: {len([s for s in analysis_queue if s['priority'] == 3])}")

            if not analysis_queue:
                logger.warning("No stocks to analyze")
                return []

            # Step 2: Analyze each stock
            print(f"\n📊 STEP 2: ANALYZING STOCKS IN PRIORITY ORDER...")
            recommendations = []

            for i, stock_info in enumerate(analysis_queue, 1):
                symbol = stock_info['symbol']
                priority_label = stock_info['priority_label']

                print(f"\n{'='*80}")
                print(f"[{priority_label}] Stock {i}/{total_stocks}: {symbol}")
                print(f"Priority: {stock_info['priority']} | Score: {stock_info['score']:.1f}")
                print(f"Context: {stock_info['context']}")
                print(f"{'='*80}")

                try:
                    # Run full analysis
                    analysis_result = self._analyze_stock(symbol, stock_info)
                    recommendations.append(analysis_result)

                    # Brief pause between stocks to avoid rate limits
                    if i < total_stocks:
                        time.sleep(2)

                except Exception as e:
                    logger.error(f"Failed to analyze {symbol}: {e}")
                    continue

            # Step 3: Rank recommendations
            print(f"\n📈 STEP 3: RANKING RECOMMENDATIONS...")
            final_recommendations = self._rank_recommendations(recommendations)

            # Display final results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print("\n" + "="*80)
            print("🎯 FINAL INVESTMENT RECOMMENDATIONS")
            print("="*80)
            print(f"Analysis Duration: {duration:.1f} seconds")
            print(f"Stocks Analyzed: {len(recommendations)}")
            print(f"Recommendations: {len(final_recommendations)}")
            print("="*80)

            for i, rec in enumerate(final_recommendations, 1):
                priority_badge = f"[{rec['priority_label']}]"
                print(f"\n#{i}. {priority_badge} {rec['symbol']} - {rec['recommendation']}")
                print(f"   Conviction: {rec['conviction']}/10")
                print(f"   Price: ${rec['current_price']:.2f} → Target: ${rec['price_target']:.2f} ({rec['potential_return']:+.1f}%)")
                print(f"   Source: {rec['discovery_sources']} | Score: {rec['discovery_score']:.1f}")
                if 'holding_info' in rec:
                    print(f"   Current: {rec['holding_info']}")
                print(f"   Reasoning: {rec['reasoning'][:200]}...")

            print("\n" + "="*80)
            print("✅ AUTONOMOUS ANALYSIS COMPLETE")
            print("="*80)

            return final_recommendations

        except Exception as e:
            logger.error(f"Autonomous pipeline failed: {e}", exc_info=True)
            raise

    def _analyze_stock(self, symbol: str, discovery_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run full 2-stage analysis on a single stock."""

        # Fetch data
        logger.info(f"  📥 Fetching data for {symbol}...")
        price_data = get_stock_data(symbol, days_back=100)
        current_price = price_data['Close'].iloc[-1]

        indicators = self.indicators_calc.get_latest_indicators(price_data)
        indicator_summary = self.indicators_calc.get_indicator_summary(price_data)

        news_articles = get_news(symbol, limit=20)
        company_info = self.market_data.get_company_info(symbol)
        fundamentals = get_fundamentals(symbol)

        # Stage 1: Analysts
        logger.info(f"  🔍 Stage 1: Running analyst reports...")
        analyst_reports = {}

        analyst_reports['market'] = self.market_analyst.analyze({
            'symbol': symbol,
            'price_data': price_data,
            'indicators': indicators,
            'indicator_summary': indicator_summary
        })

        analyst_reports['news'] = self.news_analyst.analyze({
            'symbol': symbol,
            'news_articles': news_articles
        })

        analyst_reports['sentiment'] = self.sentiment_analyst.analyze({
            'symbol': symbol,
            'price_data': price_data,
            'indicators': indicators,
            'news_sentiment': analyst_reports['news']['metadata']['overall_sentiment']
        })

        analyst_reports['fundamentals'] = self.fundamentals_analyst.analyze({
            'symbol': symbol,
            'company_info': company_info,
            'fundamentals': fundamentals
        })

        # Stage 2: Investment Debate
        logger.info(f"  💭 Stage 2: Running investment debate...")

        bull_case = self.bull_researcher.analyze({
            'symbol': symbol,
            'analyst_reports': analyst_reports,
            'current_price': current_price
        })

        bear_case = self.bear_researcher.analyze({
            'symbol': symbol,
            'analyst_reports': analyst_reports,
            'current_price': current_price
        })

        final_decision = self.research_judge.analyze({
            'symbol': symbol,
            'current_price': current_price,
            'bull_case': bull_case,
            'bear_case': bear_case
        })

        # Compile recommendation
        recommendation = final_decision['metadata']['recommendation']
        conviction = final_decision['metadata']['conviction_rating']
        price_target = final_decision['metadata']['price_target']
        potential_return = ((price_target / current_price) - 1) * 100

        logger.info(f"  ✅ {symbol}: {recommendation} (Conviction: {conviction}/10, Return: {potential_return:+.1f}%)")

        result = {
            'symbol': symbol,
            'recommendation': recommendation,
            'conviction': conviction,
            'current_price': current_price,
            'price_target': price_target,
            'potential_return': potential_return,
            'reasoning': final_decision['analysis'][:500],
            'discovery_score': discovery_info['score'],
            'discovery_sources': ', '.join(discovery_info['sources']),
            'priority': discovery_info['priority'],
            'priority_label': discovery_info['priority_label'],
            'bull_conviction': bull_case['metadata']['conviction_rating'],
            'bear_conviction': bear_case['metadata']['conviction_rating'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add holding info if it's a portfolio stock
        if discovery_info['priority'] == 1:
            portfolio_stocks = self.portfolio_config.get('portfolio', {}).get('stocks', [])
            holding = next(h for h in portfolio_stocks if h['symbol'] == symbol)
            current_value = holding['shares'] * current_price
            cost_basis = holding['shares'] * holding['avg_cost']
            current_return = ((current_value - cost_basis) / cost_basis) * 100
            
            result['holding_info'] = (
                f"{holding['shares']} shares, "
                f"Cost: ${holding['avg_cost']:.2f}, "
                f"Current: ${current_price:.2f}, "
                f"Return: {current_return:+.1f}%"
            )
        
        return result

    def _rank_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank recommendations by priority and combined score.
        
        Sorting logic:
        1. First by priority (Portfolio > Watchlist > Discovered)
        2. Then by ranking score within each priority
        """

        # Calculate ranking score
        for rec in recommendations:
            # Base score = conviction * |potential_return| * (discovery_score/100)
            base_score = (
                rec['conviction'] *
                abs(rec['potential_return']) *
                (rec['discovery_score'] / 100)
            )

            # Bonus for BUY recommendations
            if rec['recommendation'] == 'BUY':
                base_score *= 1.2
            
            # Penalty for SELL recommendations
            if rec['recommendation'] == 'SELL':
                base_score *= 0.8
            
            # Priority multiplier (portfolio gets highest weight)
            priority_multiplier = {
                1: 1000,  # Portfolio stocks get massive boost
                2: 100,   # Watchlist stocks get medium boost  
                3: 1      # Discovered stocks get no boost
            }
            
            rec['ranking_score'] = base_score * priority_multiplier[rec['priority']]

        # Sort by ranking score (which incorporates priority)
        recommendations.sort(key=lambda x: x['ranking_score'], reverse=True)

        return recommendations


if __name__ == "__main__":
    # Test autonomous pipeline with prioritization
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = AutonomousTradingPipeline()

    # Run daily analysis with all priorities
    results = pipeline.run_daily_analysis(
        analyze_portfolio=True,      # Priority 1: Analyze current holdings
        analyze_watchlist=True,      # Priority 2: Analyze watchlist
        analyze_discovered=True,     # Priority 3: Discover new stocks
        top_n_discovered=3,          # Limit discovered stocks
        min_discovery_score=10.0     # Minimum score for discovered
    )

    print(f"\n✅ Generated {len(results)} investment recommendations")
