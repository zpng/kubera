"""
Risk Manager Agent
Assesses portfolio risk, position sizing, and risk metrics
Model: deepseek/deepseek-r1-0528 (large reasoning model for complex risk assessment)
"""
import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np

from .base_agent import BaseAgent
from ..config import AGENT_MODELS

logger = logging.getLogger(__name__)


class RiskManagerAgent(BaseAgent):
    """
    Agent responsible for comprehensive risk assessment of the portfolio
    """
    
    def __init__(self):
        super().__init__(
            name="RiskManager",
            model=AGENT_MODELS["risk_manager"],
            role="Portfolio risk assessor and risk management specialist",
            temperature=0.4  # Lower temperature for more conservative risk assessment
        )
    
    def calculate_position_risk(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk metrics for a single position
        
        Args:
            stock_data: Combined data for a stock from all previous agents
            
        Returns:
            Risk metrics dictionary
        """
        try:
            symbol = stock_data.get("symbol")
            portfolio_data = stock_data.get("portfolio_position", {})
            historical = stock_data.get("historical_data", {})
            
            # Position size metrics
            shares = portfolio_data.get("shares", 0)
            avg_cost = portfolio_data.get("avg_cost", 0)
            current_price = historical.get("metrics", {}).get("current_price", avg_cost)
            position_value = shares * current_price
            cost_basis = shares * avg_cost
            
            # Volatility risk
            volatility = historical.get("metrics", {}).get("volatility", 0)
            
            # Calculate Value at Risk (simplified 1-day VaR at 95% confidence)
            # VaR = Position Value × Volatility × Z-score (1.65 for 95%)
            var_1day = position_value * (volatility / 100) * 1.65
            
            # Calculate maximum drawdown potential
            max_price = historical.get("metrics", {}).get("max_price", current_price)
            min_price = historical.get("metrics", {}).get("min_price", current_price)
            
            if max_price > 0:
                max_drawdown_pct = ((max_price - min_price) / max_price) * 100
            else:
                max_drawdown_pct = 0
            
            # Unrealized P&L risk
            unrealized_pl = position_value - cost_basis
            unrealized_pl_pct = (unrealized_pl / cost_basis * 100) if cost_basis > 0 else 0
            
            # Risk score (0-10, where 10 is highest risk)
            risk_score = min(10, (
                (volatility / 10) +  # Volatility component
                (max_drawdown_pct / 20) +  # Drawdown component
                (abs(unrealized_pl_pct) / 20)  # P&L volatility component
            ))
            
            return {
                "symbol": symbol,
                "position_value": round(position_value, 2),
                "cost_basis": round(cost_basis, 2),
                "unrealized_pl": round(unrealized_pl, 2),
                "unrealized_pl_pct": round(unrealized_pl_pct, 2),
                "volatility": volatility,
                "value_at_risk_1day": round(var_1day, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "risk_score": round(risk_score, 1),
                "risk_level": self._categorize_risk(risk_score)
            }
            
        except Exception as e:
            logger.error(f"Error calculating position risk: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "risk_score": 5,
                "risk_level": "Unknown"
            }
    
    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk based on score"""
        if risk_score < 3:
            return "Low"
        elif risk_score < 6:
            return "Medium"
        elif risk_score < 8:
            return "High"
        else:
            return "Very High"
    
    def assess_portfolio_risk(self, positions_risk: List[Dict]) -> Dict[str, Any]:
        """
        Assess overall portfolio risk
        
        Args:
            positions_risk: List of individual position risk metrics
            
        Returns:
            Portfolio-level risk assessment
        """
        if not positions_risk:
            return {}
        
        try:
            # Calculate portfolio metrics
            total_value = sum(p.get("position_value", 0) for p in positions_risk)
            total_var = sum(p.get("value_at_risk_1day", 0) for p in positions_risk)
            
            # Portfolio diversification
            position_count = len(positions_risk)
            
            # Calculate concentration risk (Herfindahl index)
            if total_value > 0:
                weights = [p.get("position_value", 0) / total_value for p in positions_risk]
                herfindahl_index = sum(w**2 for w in weights)
                concentration_score = herfindahl_index * 10  # Scale to 0-10
            else:
                concentration_score = 0
            
            # Average risk score
            avg_risk_score = np.mean([p.get("risk_score", 5) for p in positions_risk])
            
            # Portfolio VaR as percentage of total value
            portfolio_var_pct = (total_var / total_value * 100) if total_value > 0 else 0
            
            # Overall portfolio risk score
            portfolio_risk_score = (
                (avg_risk_score * 0.4) +  # Individual position risks
                (concentration_score * 0.3) +  # Concentration risk
                (min(10, portfolio_var_pct / 2) * 0.3)  # VaR risk
            )
            
            # Risk warnings
            warnings = []
            if concentration_score > 5:
                warnings.append("High concentration risk - portfolio not well diversified")
            if avg_risk_score > 7:
                warnings.append("High average position risk - consider reducing volatile positions")
            if portfolio_var_pct > 10:
                warnings.append(f"High Value at Risk ({portfolio_var_pct:.1f}%) - significant downside potential")
            
            # Find highest risk positions
            high_risk_positions = [
                p["symbol"] for p in positions_risk 
                if p.get("risk_score", 0) > 7
            ]
            
            return {
                "portfolio_value": round(total_value, 2),
                "portfolio_var_1day": round(total_var, 2),
                "portfolio_var_pct": round(portfolio_var_pct, 2),
                "position_count": position_count,
                "avg_risk_score": round(avg_risk_score, 2),
                "concentration_score": round(concentration_score, 2),
                "portfolio_risk_score": round(portfolio_risk_score, 2),
                "portfolio_risk_level": self._categorize_risk(portfolio_risk_score),
                "warnings": warnings,
                "high_risk_positions": high_risk_positions,
                "diversification_grade": self._grade_diversification(position_count, concentration_score)
            }
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return {}
    
    def _grade_diversification(self, position_count: int, concentration: float) -> str:
        """Grade portfolio diversification"""
        if position_count < 5:
            return "F"
        elif position_count < 10:
            grade = "D" if concentration > 6 else "C"
        elif position_count < 15:
            grade = "C" if concentration > 5 else "B"
        else:
            grade = "B" if concentration > 4 else "A"
        
        return grade
    
    def generate_risk_recommendations(self, portfolio_risk: Dict[str, Any], positions_risk: List[Dict]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        risk_level = portfolio_risk.get("portfolio_risk_level", "Medium")
        
        if risk_level in ["High", "Very High"]:
            recommendations.append(
                f"🔴 CRITICAL: Portfolio risk is {risk_level}. Consider reducing exposure to high-risk positions."
            )
        
        # Concentration recommendations
        if portfolio_risk.get("concentration_score", 0) > 6:
            recommendations.append(
                "⚠️ High concentration risk detected. Consider diversifying across more positions or sectors."
            )
        
        # VaR recommendations
        var_pct = portfolio_risk.get("portfolio_var_pct", 0)
        if var_pct > 10:
            recommendations.append(
                f"⚠️ High Value at Risk ({var_pct:.1f}%). Potential 1-day loss could be significant."
            )
        
        # High-risk position recommendations
        high_risk = portfolio_risk.get("high_risk_positions", [])
        if high_risk:
            recommendations.append(
                f"⚠️ High-risk positions identified: {', '.join(high_risk)}. Review these positions carefully."
            )
        
        # Positive feedback
        if risk_level == "Low" and len(recommendations) == 0:
            recommendations.append(
                "✅ Portfolio risk is well-managed. Continue monitoring positions regularly."
            )
        
        return recommendations
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method
        
        Args:
            state: Workflow state with all collected data
            
        Returns:
            Updated state with risk assessment
        """
        logger.info(f"[{self.name}] Starting risk assessment...")
        
        symbols = state.get("stock_symbols", [])
        portfolio_data = state.get("portfolio_data", {}).get("portfolio", {}).get("stocks", [])
        historical_data = state.get("historical_data", {})
        
        # Combine data for each symbol
        positions_risk = []
        
        for stock in portfolio_data:
            symbol = stock["symbol"]
            
            stock_data = {
                "symbol": symbol,
                "portfolio_position": stock,
                "historical_data": historical_data.get(symbol, {})
            }
            
            risk_metrics = self.calculate_position_risk(stock_data)
            positions_risk.append(risk_metrics)
        
        # Assess overall portfolio risk
        portfolio_risk = self.assess_portfolio_risk(positions_risk)
        
        # Generate recommendations
        recommendations = self.generate_risk_recommendations(portfolio_risk, positions_risk)
        
        # Update state
        state["risk_assessment"] = {
            "positions_risk": positions_risk,
            "portfolio_risk": portfolio_risk,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"[{self.name}] Risk assessment complete. Portfolio risk: {portfolio_risk.get('portfolio_risk_level', 'Unknown')}")
        return state


# Test function
def test_risk_manager_agent():
    """Test the RiskManagerAgent"""
    print("Testing Risk Manager Agent...")
    
    agent = RiskManagerAgent()
    
    # Test with sample state
    state = {
        "stock_symbols": ["AAPL", "TSLA"],
        "portfolio_data": {
            "portfolio": {
                "stocks": [
                    {"symbol": "AAPL", "shares": 10, "avg_cost": 150},
                    {"symbol": "TSLA", "shares": 5, "avg_cost": 200}
                ]
            }
        },
        "historical_data": {
            "AAPL": {
                "metrics": {"current_price": 175, "volatility": 25, "max_price": 180, "min_price": 140}
            },
            "TSLA": {
                "metrics": {"current_price": 250, "volatility": 45, "max_price": 280, "min_price": 180}
            }
        }
    }
    
    result_state = agent.process(state)
    
    print(f"\nRisk Assessment Results:")
    portfolio_risk = result_state["risk_assessment"]["portfolio_risk"]
    print(f"  Portfolio Value: ${portfolio_risk['portfolio_value']}")
    print(f"  Portfolio Risk Level: {portfolio_risk['portfolio_risk_level']}")
    print(f"  Average Risk Score: {portfolio_risk['avg_risk_score']}/10")
    print(f"  Value at Risk (1-day): ${portfolio_risk['portfolio_var_1day']}")
    
    print(f"\n  Position Risks:")
    for pos in result_state["risk_assessment"]["positions_risk"]:
        print(f"    {pos['symbol']}: {pos['risk_level']} ({pos['risk_score']}/10)")
    
    print(f"\n  Recommendations:")
    for rec in result_state["risk_assessment"]["recommendations"]:
        print(f"    {rec}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_risk_manager_agent()

