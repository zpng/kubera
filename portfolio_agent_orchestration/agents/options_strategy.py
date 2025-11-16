import logging
import json
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
        beta = comp.get("financial_metrics", {}).get("beta", "N/A")
        shares = pos.get("shares", 0)
        prompt = (
            f"你是资深期权策略师。基于股票 {symbol} 的综合数据（价格趋势: {trend}，波动率: {vol}，风险等级: {risk_level}，Beta: {beta}，研究决策: {decision}，信心: {conviction}/10，持股数: {shares}），为该标的给出一个最适合的期权策略建议。\n\n"
            f"要求：\n"
            f"- 仅在适合时推荐策略，若不适合期权则明确说明原因。\n"
            f"- 不给出精确行权价或保证金数值，用相对表达（如“略 OTM”“近月/30–45 天”）。\n"
            f"- 输出必须为 JSON，字段：strategy, summary, parameters, rationale(数组), risk_notes, suitability。\n"
            f"- strategy 必须在集合内：['Covered Call','Cash-Secured Put','Protective Put','Collar','Bull Call Spread','Bear Put Spread','Iron Condor','Long Call','Long Put','N/A']。\n"
            f"- 中文输出。\n"
            f"- 资金约束：可用现金(人民币)：{cash_cny if cash_cny is not None else '未知'}；请在 summary 与 suitability 中说明在该资金约束下的可执行性，不给出精确金额。\n"
        )
        return prompt

    def recommend_for_symbol(self, symbol: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
        system_msg = SystemMessage(content="你是期权策略专家，依据输入做稳健、可执行的策略推荐，中文输出，严格返回 JSON。")
        user_msg = HumanMessage(content=self.create_prompt(symbol, all_data))
        try:
            response = self.invoke([system_msg, user_msg])
            text = response.content
            try:
                obj = json.loads(text)
            except Exception:
                obj = {
                    "strategy": "N/A",
                    "summary": "模型未返回结构化 JSON，以下为原始说明",
                    "parameters": "",
                    "rationale": [text[:300]],
                    "risk_notes": "",
                    "suitability": ""
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
                "parameters": "",
                "rationale": [str(e)],
                "risk_notes": "",
                "suitability": "",
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