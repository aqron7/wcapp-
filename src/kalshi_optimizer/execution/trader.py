"""Order execution against Kalshi (phase 6) — guarded.

Safety is enforced *before* any order is sent:
  - kill_switch blocks everything,
  - only execution.mode == "live" actually sends; dry_run/paper simulate,
  - per-order stake cap.

A single bet maps to one limit order: staking $S at cost ``c`` per $1 buys
floor(S/c) contracts at limit price ``round(c*100)`` cents. True parlays can't
be replicated by independent orders (you'd need to roll winnings), so parlays
are tracking-only — execute their legs individually if you want.
"""

from __future__ import annotations

import uuid

from ..config import Config
from ..data.kalshi import KalshiClient


class Trader:
    def __init__(self, config: Config, kalshi: KalshiClient):
        self.config = config
        self.kalshi = kalshi

    def _blocked_reason(self, stake: float) -> str | None:
        ex = self.config.execution
        if ex.kill_switch:
            return "kill_switch engaged"
        if stake > self.config.sizing.max_per_market:
            return f"stake ${stake:.2f} exceeds max_per_market ${self.config.sizing.max_per_market:.2f}"
        return None

    def execute_single(self, leg: dict, stake: float) -> dict:
        """Place (or simulate) one single-leg order from a bet-slip leg."""
        price = float(leg.get("price") or 0)
        side = leg.get("side")
        ticker = leg.get("market_id")
        if price <= 0 or side not in ("yes", "no") or not ticker:
            return {"status": "error", "reason": "bad leg (need market_id, side, price)"}

        price_cents = round(price * 100)
        if price_cents < 1:
            return {"status": "error", "reason": "price below 1 cent"}
        count = int((stake * 100) // price_cents)   # integer-cent math avoids float error
        if count < 1:
            return {"status": "error", "reason": f"stake ${stake:.2f} too small for price {price:.2f}"}

        order = {"ticker": ticker, "side": side, "count": count,
                 "price_cents": price_cents, "est_cost": round(count * price_cents / 100, 2)}

        reason = self._blocked_reason(stake)
        if reason or self.config.execution.mode != "live":
            return {"status": "simulated", "reason": reason or f"mode={self.config.execution.mode}",
                    "would_place": order}

        resp = self.kalshi.place_order(ticker, side, count, order["price_cents"],
                                       client_order_id=str(uuid.uuid4()))
        return {"status": "placed", "order": order, "kalshi": resp}
