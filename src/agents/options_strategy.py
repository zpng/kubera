"""
Options Strategy Agent
Generates options trading strategies with take profit and stop loss recommendations
Model: deepseek-reasoner (for complex options strategy analysis)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OptionsStrategy(BaseModel):
    """Schema for options strategy recommendation"""
    strategy: str  # Strategy name (Covered Call, Protective Put, etc.)
    summary: str  # One-line summary in Chinese
    parameters: str  # Suggested parameters (OTM level, expiration, etc.)
    rationale: str  # Detailed reasoning in Chinese
    risk_notes: str  # Risk and suitability notes
    suitability: str  # Applicable scenarios
    
    # Take profit and stop loss for options
    take_profit_strategy: str  # When to take profits on the options position
    stop_loss_strategy: str  # When to cut losses on the options position
    profit_target_percent: float  # Target profit percentage
    loss_limit_percent: float  # Maximum loss percentage
    
    # Additional risk management
    position_size_recommendation: str  # Recommended position size
    adjustment_strategy: str  # How to adjust if underlying moves
    exit_conditions: List[str]  # Specific conditions for exiting


class OptionsStrategyAgent:
    """
    Agent for generating options trading strategies with comprehensive risk management
    Uses: deepseek-reasoner for complex options analysis
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "deepseek-reasoner"
    ):
        """
        Initialize Options Strategy Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: deepseek-reasoner)
        """
        self.model = model
        
        # Initialize LLM for options strategy analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            openai_api_key=openrouter_api_key,
            temperature=0.3,
            max_tokens=2000
        )
        
        logger.info(f"Options Strategy Agent initialized with model: {model}")
    
    def analyze_stock_for_options(
        self,
        symbol: str,
        stock_decision: str,
        conviction: int,
        current_price: float,
        avg_cost: float,
        shares: int,
        risk_level: str,
        volatility: float,
        technical_score: int,
        fundamental_score: int
    ) -> OptionsStrategy:
        """
        Analyze a stock and recommend appropriate options strategy
        
        Args:
            symbol: Stock ticker
            stock_decision: BUY/HOLD/SELL decision
            conviction: Conviction level (1-10)
            current_price: Current stock price
            avg_cost: Average cost basis
            shares: Number of shares held
            risk_level: Risk level (low/moderate/high)
            volatility: Annual volatility percentage
            technical_score: Technical analysis score (1-10)
            fundamental_score: Fundamental analysis score (1-10)
            
        Returns:
            Options strategy recommendation with risk management
        """
        logger.info(f"Analyzing options strategies for {symbol}...")
        
        # Create comprehensive options analysis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert options strategist with deep knowledge of:
- Options pricing and Greeks
- Risk management and position sizing
- Market volatility analysis
- Income generation strategies
- Hedging techniques
- Take profit and stop loss strategies for options

Your task is to recommend the most suitable options strategy with detailed risk management including take profit and stop loss levels.

Language Requirement: All narrative content MUST be written in Chinese, while keeping the specified section headers in English.

**STRATEGY SELECTION CRITERIA:**
1. **BUY + High Conviction (8-10)**: Bull Call Spread, Long Call, Cash-Secured Put
2. **BUY + Moderate Conviction (5-7)**: Covered Call, Bull Call Spread
3. **HOLD + Any Conviction**: Covered Call, Protective Put, Collar
4. **SELL + High Conviction**: Bear Call Spread, Long Put, Protective Put
5. **SELL + Moderate Conviction**: Covered Call (if profitable), Cash-Secured Put

**RISK MANAGEMENT PRINCIPLES:**
1. Never risk more than 2-5% of portfolio on any single options trade
2. Set clear profit targets (typically 25-50% for income strategies, 100%+ for directional)
3. Use stop losses based on underlying price movement or options premium decay
4. Consider time decay and implied volatility in all decisions
5. Always have an exit plan before entering the trade"""),
            ("user", """Analyze {symbol} for options strategy:

**STOCK ANALYSIS:**
Symbol: {symbol}
Stock Decision: {stock_decision}
Conviction: {conviction}/10
Current Price: ${current_price}
Average Cost: ${avg_cost}
Shares Held: {shares}
Risk Level: {risk_level}
Volatility: {volatility}%
Technical Score: {technical_score}/10
Fundamental Score: {fundamental_score}/10

**POSITION ANALYSIS:**
Unrealized P&L: {unrealized_pl_percent:+.1f}%
Position Size: {position_size} shares
Sufficient for Covered Call: {covered_call_eligible}

**OPTIONS STRATEGY REQUIREMENTS:**
Provide a comprehensive options strategy recommendation with:

1. **STRATEGY SELECTION**: Choose the most appropriate strategy based on the stock analysis
2. **PARAMETERS**: Specific strike price (relative to current price) and expiration date
3. **TAKE PROFIT STRATEGY**: When to take profits (premium decay, underlying price targets, time-based)
4. **STOP LOSS STRATEGY**: When to cut losses (underlying price movement, premium increase, time stop)
5. **POSITION SIZING**: How many contracts to trade based on portfolio size
6. **ADJUSTMENT STRATEGY**: How to adjust if underlying moves against position
7. **EXIT CONDITIONS**: Specific conditions for closing the position

**OUTPUT FORMAT:**
```json
{{
    "strategy": "Strategy Name",
    "summary": "中文一句话总结",
    "parameters": "具体的期权参数建议（必须包含：行权价、到期日。例如：行权价$150 / 现价+5%，到期日30-45天）",
    "rationale": "详细理由中文",
    "risk_notes": "风险说明",
    "suitability": "适用场景",
    "take_profit_strategy": "止盈策略详细说明",
    "stop_loss_strategy": "止损策略详细说明",
    "profit_target_percent": 25.0,
    "loss_limit_percent": 50.0,
    "position_size_recommendation": "仓位建议",
    "adjustment_strategy": "调整策略",
    "exit_conditions": ["条件1", "条件2", "条件3"]
}}
```

**STRATEGY GUIDELINES:**
- For BUY decisions: Focus on bullish strategies (Covered Calls, Bull Spreads, Long Calls)
- For SELL decisions: Focus on bearish or protective strategies (Protective Puts, Bear Spreads)
- For HOLD decisions: Focus on income generation (Covered Calls) or protection (Protective Puts)
- Always consider the number of shares held for covered call eligibility
- Adjust profit targets and stop losses based on volatility and conviction""")
        ])
        
        # Calculate derived metrics
        unrealized_pl_percent = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        covered_call_eligible = shares >= 100
        
        try:
            # Invoke LLM with comprehensive data
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "stock_decision": stock_decision,
                "conviction": conviction,
                "current_price": current_price,
                "avg_cost": avg_cost,
                "shares": shares,
                "risk_level": risk_level,
                "volatility": volatility,
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "unrealized_pl_percent": unrealized_pl_percent,
                "position_size": shares,
                "covered_call_eligible": "是" if covered_call_eligible else "否"
            })
            
            # Parse the JSON response
            strategy_data = self._parse_options_strategy(response.content)
            
            logger.info(f"✓ {symbol}: Recommended {strategy_data.strategy} strategy")
            return strategy_data
            
        except Exception as e:
            logger.error(f"Error analyzing options for {symbol}: {e}")
            # Return default strategy
            return self._get_default_strategy(symbol, stock_decision, covered_call_eligible)
    
    def _parse_options_strategy(self, content: str) -> OptionsStrategy:
        """Parse LLM response into structured options strategy"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                strategy_data = json.loads(json_match.group(1))
            else:
                # Fallback: try to parse the entire content as JSON
                strategy_data = json.loads(content)
            
            return OptionsStrategy(
                strategy=strategy_data.get('strategy', 'Covered Call'),
                summary=strategy_data.get('summary', '基础期权策略'),
                parameters=strategy_data.get('parameters', '建议选择虚值期权，到期日30-45天'),
                rationale=strategy_data.get('rationale', '基于股票分析推荐'),
                risk_notes=strategy_data.get('risk_notes', '标准风险提示'),
                suitability=strategy_data.get('suitability', '适用于当前市场情况'),
                take_profit_strategy=strategy_data.get('take_profit_strategy', '达到目标利润时平仓'),
                stop_loss_strategy=strategy_data.get('stop_loss_strategy', '标的股票价格不利变动时止损'),
                profit_target_percent=float(strategy_data.get('profit_target_percent', 25)),
                loss_limit_percent=float(strategy_data.get('loss_limit_percent', 50)),
                position_size_recommendation=strategy_data.get('position_size_recommendation', '建议小仓位试探'),
                adjustment_strategy=strategy_data.get('adjustment_strategy', '根据市场变化调整'),
                exit_conditions=strategy_data.get('exit_conditions', ['达到止盈点', '达到止损点', '临近到期'])
            )
        except Exception as e:
            logger.error(f"Error parsing options strategy: {e}")
            return self._get_default_strategy("Unknown", "HOLD", False)
    
    def _get_default_strategy(self, symbol: str, stock_decision: str, covered_call_eligible: bool) -> OptionsStrategy:
        """Get default strategy when parsing fails"""
        if stock_decision == "BUY" and covered_call_eligible:
            strategy = "Covered Call"
            summary = "备兑看涨期权策略，持有股票同时卖出看涨期权"
        elif stock_decision == "SELL":
            strategy = "Protective Put"
            summary = "保护性看跌期权策略，为股票提供下跌保护"
        else:
            strategy = "Cash-Secured Put"
            summary = "现金担保看跌期权策略，逢低买入股票"
        
        return OptionsStrategy(
            strategy=strategy,
            summary=summary,
            parameters="选择略虚值期权，到期时间30-45天",
            rationale=f"基于{stock_decision}决策和当前市场情况推荐{strategy}",
            risk_notes="期权交易存在风险，请根据自身风险承受能力操作",
            suitability="适用于当前股票分析和市场环境",
            take_profit_strategy="获得50%权利金收入时考虑平仓",
            stop_loss_strategy="权利金翻倍或标的股票价格重大不利变动时止损",
            profit_target_percent=50.0,
            loss_limit_percent=100.0,
            position_size_recommendation="单只股票期权仓位不超过投资组合5%",
            adjustment_strategy="根据标的股票价格和波动率变化调整行权价和到期日",
            exit_conditions=["达到50%盈利目标", "权利金损失100%", "距离到期日不足10天"]
        )
    
    def run(self, all_agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - generates options strategies for all stocks
        
        Args:
            all_agent_data: Combined data from all previous agents including final recommendations
            
        Returns:
            Dictionary with options strategies for each stock
        """
        logger.info("=" * 50)
        logger.info("OPTIONS STRATEGY AGENT - Starting execution")
        logger.info("=" * 50)
        
        # Extract final recommendations and other data
        final_recommendations = all_agent_data.get('final_recommendations', {})
        recommendations = final_recommendations.get('recommendations', [])
        
        portfolio_data = all_agent_data.get('historical_data', {})
        comparisons = {pc['symbol']: pc for pc in portfolio_data.get('comparisons', [])}
        
        risk_data = all_agent_data.get('risk_data', {})
        position_risks = {pr['symbol']: pr for pr in risk_data.get('position_risks', [])}
        
        # Generate options strategies for each recommended stock
        options_strategies = []
        
        for rec in recommendations:
            try:
                symbol = rec['symbol']
                comparison = comparisons.get(symbol, {})
                position_risk = position_risks.get(symbol, {})
                
                strategy = self.analyze_stock_for_options(
                    symbol=symbol,
                    stock_decision=rec['decision'],
                    conviction=rec['conviction'],
                    current_price=comparison.get('current_price', 0),
                    avg_cost=comparison.get('avg_cost', 0),
                    shares=comparison.get('shares', 0),
                    risk_level=position_risk.get('risk_level', 'moderate'),
                    volatility=position_risk.get('volatility', 20),
                    technical_score=rec.get('technical_score', 5),
                    fundamental_score=rec.get('fundamental_score', 5)
                )
                
                options_strategies.append({
                    "symbol": symbol,
                    "stock_decision": rec['decision'],
                    "options_strategy": strategy.dict()
                })
                
                logger.info(f"✓ Generated options strategy for {symbol}: {strategy.strategy}")
                
            except Exception as e:
                logger.error(f"Error generating options strategy for {rec.get('symbol', 'Unknown')}: {e}")
        
        result = {
            "agent": "options_strategy",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "total_strategies_generated": len(options_strategies),
            "options_strategies": options_strategies,
            "status": "success"
        }
        
        logger.info(f"Options Strategy Agent completed - {len(options_strategies)} strategies generated")
        logger.info("=" * 50)
        
        return result


# Test function
def test_options_strategy_agent():
    """Test the options strategy agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Options Strategy Agent...")
        print("=" * 60)
        
        agent = OptionsStrategyAgent(openrouter_api_key=api_key)
        
        # Test individual stock analysis
        strategy = agent.analyze_stock_for_options(
            symbol="AAPL",
            stock_decision="BUY",
            conviction=8,
            current_price=175.0,
            avg_cost=150.0,
            shares=100,
            risk_level="moderate",
            volatility=25,
            technical_score=7,
            fundamental_score=8
        )
        
        print(f"\n✅ Test Results:")
        print(f"   - Strategy: {strategy.strategy}")
        print(f"   - Parameters: {strategy.parameters}")
        print(f"   - Take Profit: {strategy.take_profit_strategy}")
        print(f"   - Stop Loss: {strategy.stop_loss_strategy}")
        print(f"   - Profit Target: {strategy.profit_target_percent}%")
        print(f"   - Loss Limit: {strategy.loss_limit_percent}%")
        print("\n" + "=" * 60)
        print("✅ Options Strategy Agent test passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_options_strategy_agent()