"""Trade execution against Kalshi (phase 6).

LAST phase, and the most dangerous: a bug here loses real money. Guardrails are
mandatory and enforced *before* any order is sent:
  - kill_switch blocks everything when set,
  - mode must be "live" (dry_run / paper never place real orders),
  - per-market and total exposure caps,
  - the model must have cleared the backtest CLV gate.
"""

from __future__ import annotations

from ..config import Config
from ..data.kalshi import KalshiClient
from ..types import TradeIdea


class Trader:
    def __init__(self, config: Config, kalshi: KalshiClient):
        self.config = config
        self.kalshi = kalshi

    def _blocked_reason(self, idea: TradeIdea, current_exposure: float) -> str | None:
        ex = self.config.execution
        if ex.kill_switch:
            return "kill_switch engaged"
        if ex.mode != "live":
            return f"mode={ex.mode} (not live)"
        if idea.stake > self.config.sizing.max_per_market:
            return "exceeds max_per_market"
        if current_exposure + idea.stake > self.config.sizing.max_total_exposure:
            return "exceeds max_total_exposure"
        return None

    def place(self, idea: TradeIdea, current_exposure: float = 0.0) -> dict:
        """Place (or simulate) a single trade.

        In dry_run / paper this only logs the intended order. In live mode it
        will sign and POST to Kalshi's order endpoint.
        """
        reason = self._blocked_reason(idea, current_exposure)
        if reason is not None:
            return {"status": "blocked", "reason": reason, "idea": idea}

        # TODO(phase6): build and POST the Kalshi order (signed request),
        # using limit prices derived from idea.price and orderbook depth.
        raise NotImplementedError("phase6: implement live Kalshi order placement")
