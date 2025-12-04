import logging
import json
import re
from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage

from .base_agent import BaseAgent
from ..config import AGENT_MODELS

logger = logging.getLogger(__name__)


class OptionsStrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="OptionsStrategist",
            model=AGENT_MODELS.get("options_strategy", AGENT_MODELS["deep_researcher"]),
            role="Options strategy recommendation based on equity analysis",
            temperature=0.4,
            max_tokens=4000,
        )

    def aggregate_portfolio_data(self, symbol: str, state: Dict[str, Any]) -> Dict[str, Any]:
        portfolio_stocks = state.get("portfolio_data", {}).get("portfolio", {}).get("stocks", [])
        portfolio_pos = next((s for s in portfolio_stocks if s.get("symbol") == symbol), {})
        return {
            "symbol": symbol,
            "portfolio_position": portfolio_pos,
            "historical_data": state.get("historical_data", {}).get(symbol, {}),
            "news_data": state.get("news_data", {}).get(symbol, {}),
            "company_data": state.get("company_data", {}).get(symbol, {}),
            "twitter_sentiment": state.get("twitter_sentiment", {}).get(symbol, {}),
            "reddit_sentiment": state.get("reddit_sentiment", {}).get(symbol, {}),
            "risk_metrics": next((r for r in state.get("risk_assessment", {}).get("positions_risk", []) if r.get("symbol") == symbol), {}),
            "research_result": state.get("research_results", {}).get(symbol, {}),
            "available_cash_cny": state.get("portfolio_data", {}).get("portfolio", {}).get("available_cash_cny")
        }

    def create_prompt(self, symbol: str, all_data: Dict[str, Any]) -> str:
        """Create comprehensive options strategy prompt with take profit and stop loss guidance"""
        hist = all_data.get("historical_data", {})
        comp = all_data.get("company_data", {})
        risk = all_data.get("risk_metrics", {})
        pos = all_data.get("portfolio_position", {})
        research = all_data.get("research_result", {})
        cash_cny = all_data.get("available_cash_cny")
        trend = hist.get("trends", {}).get("trend", "Unknown")
        vol = hist.get("metrics", {}).get("volatility", "N/A")
        decision = research.get("decision", "HOLD")
        conviction = research.get("conviction", 5)
        risk_level = risk.get("risk_level", "N/A")
        
        # Extract price information
        current_price = hist.get("current_price", 0)
        avg_cost = pos.get("avg_cost", 0)
        shares = pos.get("shares", 0)
        unrealized_pl = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        
        prompt = f"""基于以下股票分析，推荐最适合的期权策略，包含详细的止盈止损建议：

**股票基本信息：**
- 代码：{symbol}
- 当前价格：${current_price}
- 持仓成本：${avg_cost}
- 持仓数量：{shares}股
- 未实现盈亏：{unrealized_pl:+.1f}%
- 趋势：{trend}
- 波动率：{vol}
- 风险等级：{risk_level}

**投资建议：**
- 决策：{decision}
- 信心度：{conviction}/10
- 目标价：${research.get('target_price', 0):.2f}
- 止损价：${research.get('stop_loss', 0):.2f}

**策略选择要求：**
1. 根据股票决策（{decision}）和风险等级选择最适合的期权策略
2. 提供具体的止盈止损建议，包括权利金目标和风险限制
3. 考虑持仓数量是否满足备兑开仓要求（≥100股）
4. 结合波动率和时间衰减因素
5. 必须提供明确的期权参数建议，包括具体的行权价（如$150）和到期日（如30天）

**输出格式（必须返回JSON）：**
```json
{{
    "strategy": "策略名称（如：备兑看涨期权）",
    "summary": "策略一句话总结",
    "parameters": "具体的期权参数建议（必须包含：行权价、到期日。例如：行权价$150 / 现价+5%，到期日30-45天）",
    "rationale": "详细选择理由（中文，分点说明）",
    "risk_notes": "风险提示和注意事项",
    "suitability": "适用投资者类型",
    "take_profit_strategy": "止盈策略（何时平仓、目标权利金等）",
    "stop_loss_strategy": "止损策略（何时止损、风险限制等）",
    "profit_target_percent": 50.0,
    "loss_limit_percent": 100.0,
    "position_size_recommendation": "仓位建议（最大仓位比例）",
    "adjustment_strategy": "调整策略（标的价格变动时的应对）",
    "exit_conditions": ["止盈条件1", "止损条件2", "时间条件3"]
}}
```

**策略指导原则：**
- BUY决策：优先考虑看涨策略（备兑开仓、牛市价差、买入看涨期权）
- SELL决策：优先考虑看跌或保护策略（保护性看跌期权、熊市价差）  
- HOLD决策：优先考虑收益增强或风险对冲策略
- 高波动率：选择卖方策略，低波动率：选择买方策略
- 必须提供具体的止盈止损百分比和操作条件"""
        
        return prompt

    def recommend_for_symbol(self, symbol: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
        system_msg = SystemMessage(content="你是期权策略专家，依据输入做稳健、可执行的策略推荐，中文输出，严格返回包含止盈止损的完整JSON。")
        user_msg = HumanMessage(content=self.create_prompt(symbol, all_data))
        try:
            response = self.invoke([system_msg, user_msg])
            text = response.content
            
            # Clean markdown code blocks
            clean_text = text.strip()
            if "```" in clean_text:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", clean_text, re.DOTALL)
                if match:
                    clean_text = match.group(1)
                else:
                    clean_text = clean_text.replace("```json", "").replace("```", "")
            
            try:
                obj = json.loads(clean_text)
                # Ensure all required fields are present
                default_strategy = {
                    "strategy": "N/A",
                    "summary": "模型未返回结构化JSON",
                    "parameters": "建议选择虚值期权（OTM），到期日30-45天",
                    "rationale": ["分析未完成"],
                    "risk_notes": "",
                    "suitability": "",
                    "take_profit_strategy": "达到目标利润时平仓",
                    "stop_loss_strategy": "权利金损失50%时止损",
                    "profit_target_percent": 50.0,
                    "loss_limit_percent": 50.0,
                    "position_size_recommendation": "建议小仓位试探",
                    "adjustment_strategy": "根据市场变化调整",
                    "exit_conditions": ["达到止盈点", "达到止损点", "临近到期"]
                }
                
                # Merge with defaults to ensure all fields exist
                for key, default_value in default_strategy.items():
                    if key not in obj:
                        obj[key] = default_value
                
            except Exception:
                obj = {
                    "strategy": "N/A",
                    "summary": "模型未返回结构化 JSON，以下为原始说明",
                    "parameters": "建议暂缓期权操作",
                    "rationale": [text[:300]],
                    "risk_notes": "",
                    "suitability": "",
                    "take_profit_strategy": "达到目标利润时平仓",
                    "stop_loss_strategy": "权利金损失50%时止损",
                    "profit_target_percent": 50.0,
                    "loss_limit_percent": 50.0,
                    "position_size_recommendation": "建议小仓位试探",
                    "adjustment_strategy": "根据市场变化调整",
                    "exit_conditions": ["达到止盈点", "达到止损点", "临近到期"]
                }
            
            if not isinstance(obj.get("rationale", []), list):
                obj["rationale"] = [str(obj.get("rationale", ""))]
            obj["timestamp"] = datetime.now().isoformat()
            return obj
        except Exception as e:
            logger.error(f"Options strategy failed for {symbol}: {e}")
            return {
                "strategy": "N/A",
                "summary": "期权策略生成失败，建议使用现货管理或等待条件改善",
                "parameters": "建议暂缓期权操作",
                "rationale": [str(e)],
                "risk_notes": "",
                "suitability": "",
                "take_profit_strategy": "系统错误，请手动设置止盈",
                "stop_loss_strategy": "系统错误，请手动设置止损",
                "profit_target_percent": 25.0,
                "loss_limit_percent": 50.0,
                "position_size_recommendation": "暂时不建议期权交易",
                "adjustment_strategy": "等待系统恢复",
                "exit_conditions": ["系统错误，建议观望"],
                "timestamp": datetime.now().isoformat()
            }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        symbols = state.get("stock_symbols", [])
        if not symbols:
            return state
        results = state.get("research_results", {})
        for symbol in symbols:
            all_data = self.aggregate_portfolio_data(symbol, state)
            strat = self.recommend_for_symbol(symbol, all_data)
            if symbol in results:
                results[symbol]["options_strategy"] = strat
        state["research_results"] = results
        state["options_timestamp"] = datetime.now().isoformat()
        return state

def test_options_strategy_agent():
    agent = OptionsStrategyAgent()
    state = {
        "stock_symbols": ["AAPL"],
        "portfolio_data": {"portfolio": {"stocks": [{"symbol": "AAPL", "shares": 120, "avg_cost": 120.0, "current_price": 140.0}]}},
        "historical_data": {"AAPL": {"metrics": {"volatility": 2.5}, "trends": {"trend": "bullish"}}},
        "company_data": {"AAPL": {"financial_metrics": {"beta": 1.2}}},
        "risk_assessment": {"positions_risk": [{"symbol": "AAPL", "risk_level": "Medium"}]},
        "research_results": {"AAPL": {"decision": "HOLD", "conviction": 7}}
    }
    out = agent.process(state)
    print(json.dumps(out.get("research_results", {}).get("AAPL", {}).get("options_strategy", {}), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    test_options_strategy_agent()