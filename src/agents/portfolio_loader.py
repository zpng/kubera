"""
Portfolio Loader Agent
Reads portfolio JSON and provides current holdings data for analysis
Model: deepseek/deepseek-chat-v3.1 (fast, efficient data processing)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PortfolioStock(BaseModel):
    """Schema for portfolio stock"""
    symbol: str = Field(description="Stock ticker symbol")
    shares: float = Field(description="Number of shares owned")
    avg_cost: float = Field(description="Average cost per share")
    current_price: Optional[float] = Field(default=None, description="Current market price")
    equity: Optional[float] = Field(default=None, description="Current equity value")


class PortfolioData(BaseModel):
    """Schema for complete portfolio data"""
    stocks: List[PortfolioStock] = Field(description="List of portfolio holdings")
    total_equity: Optional[float] = Field(default=None, description="Total portfolio equity")
    last_updated: Optional[str] = Field(default=None, description="Last update timestamp")


class PortfolioLoaderAgent:
    """
    Agent responsible for loading and validating portfolio data
    Uses: deepseek/deepseek-chat-v3.1 for fast data processing and validation
    """
    
    def __init__(
        self,
        portfolio_path: str = None,
        openrouter_api_key: str = None,
        model: str = "deepseek/deepseek-chat-v3.1"
    ):
        """
        Initialize Portfolio Loader Agent
        
        Args:
            portfolio_path: Path to portfolio.json file
            openrouter_api_key: OpenRouter API key
            model: Model to use (default: deepseek/deepseek-chat-v3.1)
        """
        self.portfolio_path = portfolio_path or self._get_default_portfolio_path()
        self.model = model
        
        # Initialize LLM for validation and enrichment
        self.llm = ChatOpenAI(
            model=model,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=openrouter_api_key,
            temperature=0.1  # Low temperature for consistent data processing
        )
        
        # Set up JSON output parser
        self.output_parser = JsonOutputParser(pydantic_object=PortfolioData)
        
        logger.info(f"Portfolio Loader Agent initialized with model: {model}")
    
    def _get_default_portfolio_path(self) -> Path:
        """Get default portfolio path"""
        return Path(__file__).parent.parent.parent / "config" / "portfolio.json"
    
    def load_portfolio(self) -> Dict[str, Any]:
        """
        Load portfolio data from JSON file
        
        Returns:
            Dictionary containing portfolio holdings
        """
        try:
            with open(self.portfolio_path, 'r') as f:
                portfolio_data = json.load(f)
            
            logger.info(f"Successfully loaded portfolio from {self.portfolio_path}")
            return portfolio_data
        
        except FileNotFoundError:
            logger.error(f"Portfolio file not found: {self.portfolio_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in portfolio file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading portfolio: {e}")
            raise
    
    def extract_holdings(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract holdings from portfolio data
        
        Args:
            portfolio_data: Raw portfolio data
            
        Returns:
            List of portfolio holdings
        """
        holdings = portfolio_data.get('portfolio', {}).get('stocks', [])
        logger.info(f"Extracted {len(holdings)} holdings from portfolio")
        return holdings
    
    def validate_holdings(self, holdings: List[Dict[str, Any]]) -> PortfolioData:
        """
        Validate and structure holdings data using LLM
        
        Args:
            holdings: List of raw holdings data
            
        Returns:
            Validated PortfolioData object
        """
        try:
            # Create prompt for validation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a data validation agent. Your task is to validate and structure portfolio holdings data.
                
Validate that each stock has:
- symbol (ticker)
- shares (number of shares)
- avg_cost (average cost per share)
- current_price (optional)
- equity (optional)

Return the data in JSON format following the PortfolioData schema.
Calculate total_equity if current_price is available.
Add timestamp for last_updated."""),
                ("user", "Validate and structure this portfolio data:\n{holdings}")
            ])
            
            # Create chain
            chain = prompt | self.llm | self.output_parser
            
            # Execute validation
            result = chain.invoke({"holdings": json.dumps(holdings, indent=2)})
            
            logger.info("Portfolio holdings validated successfully")
            return PortfolioData(**result)
        
        except Exception as e:
            logger.error(f"Error validating holdings: {e}")
            # Fallback: return unvalidated data
            return PortfolioData(
                stocks=[PortfolioStock(**holding) for holding in holdings],
                last_updated=datetime.now().isoformat()
            )
    
    def get_stock_symbols(self, holdings: List[Dict[str, Any]]) -> List[str]:
        """
        Extract stock symbols from holdings
        
        Args:
            holdings: List of holdings data
            
        Returns:
            List of stock ticker symbols
        """
        symbols = [holding['symbol'] for holding in holdings if 'symbol' in holding]
        logger.info(f"Extracted {len(symbols)} stock symbols: {', '.join(symbols)}")
        return symbols
    
    def run(self) -> Dict[str, Any]:
        """
        Main execution method - loads and validates portfolio
        
        Returns:
            Dictionary with portfolio data and metadata
        """
        logger.info("=" * 50)
        logger.info("PORTFOLIO LOADER AGENT - Starting execution")
        logger.info("=" * 50)
        
        # Load portfolio data
        portfolio_data = self.load_portfolio()
        
        # Extract holdings
        holdings = self.extract_holdings(portfolio_data)
        
        # Validate holdings
        validated_data = self.validate_holdings(holdings)
        
        # Extract symbols
        symbols = self.get_stock_symbols(holdings)
        
        result = {
            "agent": "portfolio_loader",
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
            "holdings": holdings,
            "validated_data": validated_data.dict(),
            "symbols": symbols,
            "total_holdings": len(holdings),
            "status": "success"
        }
        
        logger.info(f"Portfolio Loader Agent completed - {len(holdings)} holdings processed")
        logger.info("=" * 50)
        
        return result


# Test function
def test_portfolio_loader():
    """Test the portfolio loader agent"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return
    
    try:
        print("\n🧪 Testing Portfolio Loader Agent...")
        print("=" * 60)
        
        agent = PortfolioLoaderAgent(openrouter_api_key=api_key)
        result = agent.run()
        
        print("\n✅ Test Results:")
        print(f"   - Holdings loaded: {result['total_holdings']}")
        print(f"   - Symbols: {', '.join(result['symbols'])}")
        print(f"   - Status: {result['status']}")
        print("\n" + "=" * 60)
        print("✅ Portfolio Loader Agent test passed!")
        
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
    
    test_portfolio_loader()

