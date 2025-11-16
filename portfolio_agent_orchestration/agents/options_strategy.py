"""
Options Strategy Agent
Generates rule-based options strategies for the portfolio under cash and risk constraints
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class OptionsStrategyAgent:
    """
    Produce options recommendations (cash-secured puts, call spreads, protective puts,
    collars, straddles, and covered calls when underlying is held)
    """

    def __init__(self, cash_budget_rmb: int | None = None):
        # Read cash budget from env with default
        env_budget = os.getenv("OPTIONS_CASH_BUDGET_RMB")
        try:
            env_budget_val = int(env_budget) if env_budget else None
        except ValueError:
            env_budget_val = None

        self.cash_budget_rmb = cash_budget_rmb if cash_budget_rmb is not None else (env_budget_val if env_budget_val is not None else 200_000)
        logger.info(f"OptionsStrategyAgent initialized with cash budget: {self.cash_budget_rmb} RMB")

    def _is_cn_etf(self, symbol: str) -> bool:
        return symbol.isdigit() and len(symbol) == 6

    def _default_cn_etfs(self) -> List[str]:
        return ["510300", "510050"]

    def _choose_expiry(self, months: int) -> str:
        # Represent expiry in a relative form to avoid fabricating exact dates
        if months <= 1:
            return "next_month"
        elif months <= 2:
            return "second_month"
        elif months <= 3:
            return "next_quarter"
        else:
            return f"{months}_months"

    def _recommend_cash_secured_put(self, symbol: str, price_rmb: float, budget_rmb: int) -> Dict[str, Any]:
        multiplier = 10_000 if self._is_cn_etf(symbol) else 100
        moneyness = "OTM_8to10pct"
        margin_per_contract = price_rmb * multiplier
        max_contracts = max(int(budget_rmb // margin_per_contract), 0)
        contracts = max_contracts if max_contracts > 0 else 0

        return {
            "symbol": symbol,
            "strategy": "cash_secured_put",
            "expiry": self._choose_expiry(3),
            "legs": [{"type": "sell_put", "moneyness": moneyness, "contracts": contracts}],
            "budget_rmb": min(budget_rmb, contracts * margin_per_contract),
            "margin_rmb": contracts * margin_per_contract,
            "notes": "以较低波动赚取权利金；可能被指派则接收ETF。"
        }

    def _recommend_call_spread(self, symbol: str, price_rmb: float, budget_rmb: int) -> Dict[str, Any]:
        # Budget-driven sizing without fabricating premiums
        unit_budget = 6000
        units = max(int(budget_rmb // unit_budget), 1)
        return {
            "symbol": symbol,
            "strategy": "bull_call_spread",
            "expiry": self._choose_expiry(3),
            "legs": [
                {"type": "buy_call", "moneyness": "ATM_or_ITM_minor", "units": units},
                {"type": "sell_call", "moneyness": "OTM_8to12pct", "units": units}
            ],
            "budget_rmb": units * unit_budget,
            "notes": "受控成本参与上行，最大亏损受限于净支出。"
        }

    def _recommend_protective_put(self, symbol: str, price_rmb: float, budget_rmb: int) -> Dict[str, Any]:
        unit_budget = 5000
        units = max(int(budget_rmb // unit_budget), 1)
        return {
            "symbol": symbol,
            "strategy": "protective_put",
            "expiry": self._choose_expiry(2),
            "legs": [{"type": "buy_put", "moneyness": "OTM_5to10pct", "units": units}],
            "budget_rmb": units * unit_budget,
            "notes": "限制下行风险，适用于高权重持仓。"
        }

    def _recommend_collar(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "strategy": "collar",
            "expiry": self._choose_expiry(3),
            "legs": [
                {"type": "buy_put", "moneyness": "OTM_5to10pct"},
                {"type": "sell_call", "moneyness": "OTM_5to10pct"}
            ],
            "notes": "用部分上行换取下行保护，净成本低。"
        }

    def _recommend_straddle(self, symbol: str, budget_rmb: int) -> Dict[str, Any]:
        unit_budget = 8000
        units = max(int(budget_rmb // unit_budget), 1)
        return {
            "symbol": symbol,
            "strategy": "long_straddle",
            "expiry": self._choose_expiry(1),
            "legs": [
                {"type": "buy_call", "moneyness": "ATM", "units": units},
                {"type": "buy_put", "moneyness": "ATM", "units": units}
            ],
            "budget_rmb": units * unit_budget,
            "notes": "事件驱动波动博弈，小额谨慎配置。"
        }

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate options recommendations and put them into workflow state
        """
        logger.info("Generating options strategy recommendations...")

        # Try to get symbols and current prices from state
        symbols: List[str] = state.get("stock_symbols", [])
        comparisons = {c.get("symbol"): c for c in state.get("historical_data", {}).get("comparisons", [])}
        base_symbols = symbols

        # Allocate budgets per strategy
        total_budget = self.cash_budget_rmb
        csp_budget = int(total_budget * 0.6)
        spread_budget = int(total_budget * 0.15)
        protective_budget = int(total_budget * 0.1)
        straddle_budget = int(total_budget * 0.1)

        recommendations: List[Dict[str, Any]] = []

        if not base_symbols:
            state["options_recommendations"] = []
            logger.info("No holdings found, skip options recommendations")
            return state

        chosen_csp_symbol = base_symbols[0]
        price = comparisons.get(chosen_csp_symbol, {}).get("current_price", 4.0)
        recommendations.append(self._recommend_cash_secured_put(chosen_csp_symbol, price, csp_budget))

        # Bull call spread on another ETF
        chosen_spread_symbol = base_symbols[1] if len(base_symbols) > 1 else base_symbols[0]
        price2 = comparisons.get(chosen_spread_symbol, {}).get("current_price", 3.0)
        recommendations.append(self._recommend_call_spread(chosen_spread_symbol, price2, spread_budget))

        # Protective put on highest-risk or first holding if available
        target_symbol = base_symbols[0]
        price3 = comparisons.get(target_symbol, {}).get("current_price", 4.0)
        recommendations.append(self._recommend_protective_put(target_symbol, price3, protective_budget))

        # Small straddle position for event-driven exposure
        recommendations.append(self._recommend_straddle(chosen_spread_symbol, straddle_budget))

        state["options_recommendations"] = recommendations
        logger.info(f"✓ Options recommendations generated: {len(recommendations)} strategies")
        return state