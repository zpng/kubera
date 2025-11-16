"""
Options Strategy Agent (legacy pipeline)
Generates rule-based options strategies for the portfolio under cash and risk constraints
"""

import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class OptionsStrategyAgent:
    def __init__(self, cash_budget_rmb: int | None = None):
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
        contracts = max(int(budget_rmb // margin_per_contract), 0)
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

    def generate(self, symbols: List[str], comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sym_list = symbols
        comp_map = {c.get("symbol"): c for c in comparisons}
        total_budget = self.cash_budget_rmb
        csp_budget = int(total_budget * 0.6)
        spread_budget = int(total_budget * 0.15)
        protective_budget = int(total_budget * 0.1)
        straddle_budget = int(total_budget * 0.1)

        recs: List[Dict[str, Any]] = []
        if not sym_list:
            return recs
        csp_symbol = sym_list[0]
        p1 = comp_map.get(csp_symbol, {}).get("current_price", 4.0)
        recs.append(self._recommend_cash_secured_put(csp_symbol, p1, csp_budget))

        spread_symbol = sym_list[1] if len(sym_list) > 1 else sym_list[0]
        p2 = comp_map.get(spread_symbol, {}).get("current_price", 3.0)
        recs.append(self._recommend_call_spread(spread_symbol, p2, spread_budget))

        target_symbol = sym_list[0]
        p3 = comp_map.get(target_symbol, {}).get("current_price", 4.0)
        recs.append(self._recommend_protective_put(target_symbol, p3, protective_budget))

        recs.append(self._recommend_straddle(spread_symbol, straddle_budget))
        return recs