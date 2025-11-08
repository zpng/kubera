"""
Portfolio Loader Agent
Loads and validates portfolio.json data
Model: deepseek/deepseek-chat-v3.1 (fast, good with structured data)
"""
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from .base_agent import BaseAgent
from ..config import PORTFOLIO_PATH, AGENT_MODELS

logger = logging.getLogger(__name__)


class PortfolioLoaderAgent(BaseAgent):
    """
    Agent responsible for loading and validating portfolio data
    """
    
    def __init__(self):
        super().__init__(
            name="PortfolioLoader",
            model=AGENT_MODELS["portfolio_loader"],
            role="Portfolio data loader and validator",
            temperature=0.3  # Low temperature for structured tasks
        )
        self.portfolio_path = PORTFOLIO_PATH
    
    def load_portfolio(self) -> Dict[str, Any]:
        """Load portfolio from JSON file"""
        try:
            with open(self.portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            logger.info(f"Loaded portfolio from {self.portfolio_path}")
            return portfolio_data
        except FileNotFoundError:
            logger.error(f"Portfolio file not found: {self.portfolio_path}")
            return {"portfolio": {"stocks": []}}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in portfolio file: {e}")
            return {"portfolio": {"stocks": []}}
    
    def validate_portfolio(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Validate portfolio data structure"""
        errors = []
        
        if "portfolio" not in portfolio_data:
            errors.append("Missing 'portfolio' key")
            return errors
        
        if "stocks" not in portfolio_data["portfolio"]:
            errors.append("Missing 'stocks' key in portfolio")
            return errors
        
        stocks = portfolio_data["portfolio"]["stocks"]
        
        for idx, stock in enumerate(stocks):
            required_fields = ["symbol", "shares", "avg_cost"]
            for field in required_fields:
                if field not in stock:
                    errors.append(f"Stock {idx}: Missing required field '{field}'")
            
            # Validate data types
            if "shares" in stock and not isinstance(stock["shares"], (int, float)):
                errors.append(f"Stock {idx}: 'shares' must be numeric")
            
            if "avg_cost" in stock and not isinstance(stock["avg_cost"], (int, float)):
                errors.append(f"Stock {idx}: 'avg_cost' must be numeric")
        
        return errors
    
    def extract_stock_symbols(self, portfolio_data: Dict[str, Any]) -> List[str]:
        """Extract list of stock symbols from portfolio"""
        try:
            stocks = portfolio_data.get("portfolio", {}).get("stocks", [])
            symbols = [stock["symbol"] for stock in stocks if "symbol" in stock]
            logger.info(f"Extracted {len(symbols)} stock symbols: {symbols}")
            return symbols
        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")
            return []
    
    def get_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate portfolio summary statistics"""
        try:
            stocks = portfolio_data.get("portfolio", {}).get("stocks", [])
            
            total_cost_basis = sum(
                stock.get("shares", 0) * stock.get("avg_cost", 0)
                for stock in stocks
            )
            
            total_equity = sum(
                stock.get("shares", 0) * stock.get("current_price", stock.get("avg_cost", 0))
                for stock in stocks
            )
            
            summary = {
                "total_stocks": len(stocks),
                "stock_symbols": [s["symbol"] for s in stocks],
                "total_cost_basis": round(total_cost_basis, 2),
                "total_equity": round(total_equity, 2),
                "unrealized_pl": round(total_equity - total_cost_basis, 2),
                "timestamp": portfolio_data.get("portfolio", {}).get("date", datetime.now().isoformat())
            }
            
            logger.info(f"Portfolio summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {}
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method for the agent
        
        Args:
            state: Workflow state
            
        Returns:
            Updated state with portfolio data
        """
        logger.info(f"[{self.name}] Starting portfolio loading...")
        
        # Load portfolio
        portfolio_data = self.load_portfolio()
        
        # Validate
        validation_errors = self.validate_portfolio(portfolio_data)
        if validation_errors:
            logger.warning(f"Portfolio validation errors: {validation_errors}")
            state["errors"] = state.get("errors", []) + validation_errors
        
        # Extract symbols
        symbols = self.extract_stock_symbols(portfolio_data)
        
        # Generate summary
        summary = self.get_portfolio_summary(portfolio_data)
        
        # Update state
        state.update({
            "portfolio_data": portfolio_data,
            "stock_symbols": symbols,
            "portfolio_summary": summary,
            "portfolio_loaded": True,
            "portfolio_load_timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"[{self.name}] Successfully loaded {len(symbols)} stocks")
        return state


# Test function
def test_portfolio_loader():
    """Test the PortfolioLoaderAgent"""
    print("Testing Portfolio Loader Agent...")
    
    agent = PortfolioLoaderAgent()
    state = {}
    
    result_state = agent.process(state)
    
    print(f"\nPortfolio Summary:")
    print(json.dumps(result_state.get("portfolio_summary", {}), indent=2))
    
    print(f"\nStock Symbols: {result_state.get('stock_symbols', [])}")
    
    if result_state.get("errors"):
        print(f"\nValidation Errors: {result_state['errors']}")
    
    return result_state


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_portfolio_loader()

