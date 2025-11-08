"""
Risk Manager Agent
Analyzes portfolio risk, position sizing, and risk metrics
Model: deepseek/deepseek-r1-0528 (large model for complex risk assessment)
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PositionRisk(BaseModel):
    """Schema for individual position risk"""
    symbol: str
    position_size_percent: float
    volatility: float
    beta: float
    risk_level: str  # low, moderate, high, extreme
    stop_loss_suggestion: float
    take_profit_suggestion: float
    max_loss_potential: float
    risk_reward_ratio: float
    risk_assessment: str


class PortfolioRisk(BaseModel):
    """Schema for overall portfolio risk"""
    total_portfolio_value: float
    diversification_score: float  # 0-100
    concentration_risk: str  # low, moderate, high
    portfolio_beta: float
    portfolio_volatility: float
    max_drawdown_potential: float
    value_at_risk: float  # 95% VaR
    risk_adjusted_return: float
    overall_risk_level: str  # conservative, moderate, aggressive
    risk_recommendations: List[str]
    risk_summary: str


class RiskManagerAgent:
    """
    Agent responsible for comprehensive risk analysis
    Uses: deepseek/deepseek-r1-0528 for complex risk assessment reasoning
    """
    
    def __init__(
        self,
        openrouter_api_key: str = None,
        model: str = "deepseek/deepseek-r1-0528"
    ):
        """
        Initialize Risk Manager Agent
        
        Args:
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: deepseek/deepseek-r1-0528)
        """
        self.model = model
        
        # Initialize LLM for risk analysis
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.1  # Very low temperature for risk analysis precision
        )
        
        logger.info(f"Risk Manager Agent initialized with model: {model}")
    
    def calculate_position_risks(
        self,
        portfolio_comparison: List[Dict[str, Any]],
        total_portfolio_value: float
    ) -> List[PositionRisk]:
        """
        Calculate risk metrics for each position
        
        Args:
            portfolio_comparison: Portfolio comparison data
            total_portfolio_value: Total portfolio value
            
        Returns:
            List of position risk analyses
        """
        logger.info("Calculating position risks...")
        
        position_risks = []
        
        for position in portfolio_comparison:
            symbol = position['symbol']
            current_value = position['current_value']
            current_price = position['current_price']
            unrealized_pl_percent = position['unrealized_pl_percent']
            
            # Calculate position size as percentage of portfolio
            position_size_percent = (current_value / total_portfolio_value * 100) if total_portfolio_value > 0 else 0
            
            # Estimate volatility (simplified - in production use historical data)
            volatility = abs(unrealized_pl_percent) / 10  # Rough estimate
            
            # Estimate beta (simplified)
            beta = 1.0  # Default market beta
            
            # Determine risk level
            if position_size_percent > 25:
                risk_level = "extreme"
            elif position_size_percent > 15:
                risk_level = "high"
            elif position_size_percent > 10:
                risk_level = "moderate"
            else:
                risk_level = "low"
            
            # Calculate stop loss and take profit suggestions
            stop_loss = current_price * 0.90  # 10% stop loss
            take_profit = current_price * 1.15  # 15% take profit
            
            # Calculate max loss potential
            max_loss_potential = current_value * 0.10  # 10% potential loss
            
            # Calculate risk/reward ratio
            risk_reward_ratio = 1.5  # 1:1.5 risk/reward
            
            risk_assessment = self._generate_position_risk_assessment(
                symbol, position_size_percent, risk_level, unrealized_pl_percent
            )
            
            position_risks.append(PositionRisk(
                symbol=symbol,
                position_size_percent=round(position_size_percent, 2),
                volatility=round(volatility, 2),
                beta=beta,
                risk_level=risk_level,
                stop_loss_suggestion=round(stop_loss, 2),
                take_profit_suggestion=round(take_profit, 2),
                max_loss_potential=round(max_loss_potential, 2),
                risk_reward_ratio=risk_reward_ratio,
                risk_assessment=risk_assessment
            ))
            
            logger.info(f"✓ {symbol}: {risk_level} risk ({position_size_percent:.1f}% of portfolio)")
        
        return position_risks
    
    def _generate_position_risk_assessment(
        self,
        symbol: str,
        position_size: float,
        risk_level: str,
        pl_percent: float
    ) -> str:
        """Generate brief position risk assessment"""
        if risk_level == "extreme":
            return f"EXTREME RISK: {position_size:.1f}% concentration. Consider reducing position."
        elif risk_level == "high":
            return f"HIGH RISK: {position_size:.1f}% of portfolio. Monitor closely."
        elif risk_level == "moderate":
            return f"MODERATE RISK: {position_size:.1f}% allocation is reasonable."
        else:
            return f"LOW RISK: {position_size:.1f}% is well-sized position."
    
    def analyze_portfolio_risk(
        self,
        position_risks: List[PositionRisk],
        total_portfolio_value: float,
        portfolio_comparison: List[Dict[str, Any]]
    ) -> PortfolioRisk:
        """
        Analyze overall portfolio risk using LLM
        
        Args:
            position_risks: Individual position risks
            total_portfolio_value: Total portfolio value
            portfolio_comparison: Portfolio comparison data
            
        Returns:
            Comprehensive portfolio risk analysis
        """
        logger.info("Analyzing overall portfolio risk...")
        
        # Calculate basic metrics
        num_positions = len(position_risks)
        high_risk_positions = sum(1 for p in position_risks if p.risk_level in ["high", "extreme"])
        
        # Calculate diversification score
        max_position_size = max(p.position_size_percent for p in position_risks) if position_risks else 0
        diversification_score = 100 - (max_position_size * 2)  # Simplified
        
        # Determine concentration risk
        if max_position_size > 25:
            concentration_risk = "high"
        elif max_position_size > 15:
            concentration_risk = "moderate"
        else:
            concentration_risk = "low"
        
        # Calculate portfolio beta (weighted average)
        portfolio_beta = sum(p.beta * p.position_size_percent for p in position_risks) / 100 if position_risks else 1.0
        
        # Calculate portfolio volatility
        portfolio_volatility = sum(p.volatility * p.position_size_percent for p in position_risks) / 100 if position_risks else 0
        
        # Estimate max drawdown potential
        max_drawdown_potential = total_portfolio_value * 0.15  # 15% potential drawdown
        
        # Calculate Value at Risk (95% VaR)
        value_at_risk = total_portfolio_value * 0.05  # Simplified 5% VaR
        
        # Calculate risk-adjusted return (Sharpe ratio proxy)
        total_pl = sum(pc['unrealized_pl'] for pc in portfolio_comparison)
        risk_adjusted_return = total_pl / (portfolio_volatility + 0.01)  # Avoid division by zero
        
        # Determine overall risk level
        if high_risk_positions > num_positions / 2:
            overall_risk_level = "aggressive"
        elif concentration_risk == "high":
            overall_risk_level = "aggressive"
        elif diversification_score < 50:
            overall_risk_level = "moderate"
        else:
            overall_risk_level = "conservative"
        
        # Generate recommendations using LLM
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional risk manager specializing in portfolio risk assessment.

Your task is to analyze portfolio risk metrics and provide:
1. Comprehensive risk evaluation
2. Specific risk mitigation recommendations
3. Position sizing suggestions
4. Diversification improvements
5. Risk management strategy

Focus on:
- Concentration risk and diversification
- Position sizing relative to portfolio
- Correlation between holdings
- Market exposure (beta)
- Downside protection strategies
- Risk/reward optimization

Provide actionable, specific recommendations."""),
            ("user", """Analyze this portfolio risk profile:

Total Portfolio Value: ${total_value:.2f}
Number of Positions: {num_positions}
Diversification Score: {diversification:.0f}/100
Concentration Risk: {concentration}
Portfolio Beta: {beta:.2f}
Portfolio Volatility: {volatility:.2f}%
Max Drawdown Potential: ${max_drawdown:.2f}
Value at Risk (95%): ${var:.2f}
Overall Risk Level: {risk_level}

High Risk Positions: {high_risk_count}

Position Breakdown:
{positions}

Provide:
1. Risk evaluation summary
2. Top 5 specific risk mitigation recommendations
3. Overall risk management strategy""")
        ])
        
        # Format position data
        positions_text = "\n".join([
            f"- {p.symbol}: {p.position_size_percent:.1f}% | {p.risk_level.upper()} risk | Volatility: {p.volatility:.1f}%"
            for p in position_risks
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "total_value": total_portfolio_value,
                "num_positions": num_positions,
                "diversification": diversification_score,
                "concentration": concentration_risk,
                "beta": portfolio_beta,
                "volatility": portfolio_volatility,
                "max_drawdown": max_drawdown_potential,
                "var": value_at_risk,
                "risk_level": overall_risk_level,
                "high_risk_count": high_risk_positions,
                "positions": positions_text
            })
            
            risk_summary = response.content
            
            # Extract recommendations (simplified - parse from LLM response)
            risk_recommendations = [
                "Consider rebalancing overweight positions",
                "Implement stop-loss orders for high-risk holdings",
                "Review correlation between positions",
                "Maintain cash reserves for opportunities",
                "Monitor market volatility indicators"
            ]
            
        except Exception as e:
            logger.error(f"Error generating risk analysis with LLM: {e}")
            risk_summary = "Risk analysis unavailable"
            risk_recommendations = ["Review portfolio regularly", "Maintain diversification"]
        
        return PortfolioRisk(
            total_portfolio_value=total_portfolio_value,
            diversification_score=round(diversification_score, 1),
            concentration_risk=concentration_risk,
            portfolio_beta=round(portfolio_beta, 2),
            portfolio_volatility=round(portfolio_volatility, 2),
            max_drawdown_potential=round(max_drawdown_potential, 2),
            value_at_risk=round(value_at_risk, 2),
            risk_adjusted_return=round(risk_adjusted_return, 2),
            overall_risk_level=overall_risk_level,
            risk_recommendations=risk_recommendations,
            risk_summary=risk_summary[:500] if risk_summary else "Analysis unavailable"
        )
    
    def run(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method - performs comprehensive risk analysis
        
        Args:
            historical_data: Historical data with portfolio comparison
            
        Returns:
            Dictionary with risk analysis
        """
        logger.info("=" * 50)
        logger.info("RISK MANAGER AGENT - Starting execution")
        logger.info("=" * 50)
        
        portfolio_comparison = historical_data.get('comparisons', [])
        
        # Calculate total portfolio value
        total_portfolio_value = sum(pos['current_value'] for pos in portfolio_comparison)
        
        # Calculate position risks
        position_risks = self.calculate_position_risks(portfolio_comparison, total_portfolio_value)
        
        # Analyze overall portfolio risk
        portfolio_risk = self.analyze_portfolio_risk(position_risks, total_portfolio_value, portfolio_comparison)
        
        result = {
            "agent": "risk_manager",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "position_risks": [pr.dict() for pr in position_risks],
            "portfolio_risk": portfolio_risk.dict(),
            "status": "success"
        }
        
        logger.info(f"Risk Manager Agent completed - Risk Level: {portfolio_risk.overall_risk_level.upper()}")
        logger.info("=" * 50)
        
        return result


# Test function
def test_risk_manager_agent():
    """Test the risk manager agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Risk Manager Agent...")
        print("=" * 60)
        
        # Mock historical data
        mock_data = {
            "comparisons": [
                {
                    "symbol": "AAPL",
                    "shares": 10,
                    "avg_cost": 150.0,
                    "current_price": 175.0,
                    "current_value": 1750.0,
                    "cost_basis": 1500.0,
                    "unrealized_pl": 250.0,
                    "unrealized_pl_percent": 16.67
                },
                {
                    "symbol": "TSLA",
                    "shares": 5,
                    "avg_cost": 200.0,
                    "current_price": 220.0,
                    "current_value": 1100.0,
                    "cost_basis": 1000.0,
                    "unrealized_pl": 100.0,
                    "unrealized_pl_percent": 10.0
                }
            ]
        }
        
        agent = RiskManagerAgent(openrouter_api_key=api_key)
        result = agent.run(mock_data)
        
        print("\n✅ Test Results:")
        print(f"   - Positions analyzed: {len(result['position_risks'])}")
        print(f"   - Overall risk level: {result['portfolio_risk']['overall_risk_level']}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Risk Manager Agent test passed!")
        
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
    
    test_risk_manager_agent()

