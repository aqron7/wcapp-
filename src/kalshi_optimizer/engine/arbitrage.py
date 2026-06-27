"""Cross-platform arbitrage scanner (strategy C).

Needs no model — just consistent ``event_key``s across platforms. Built in
phase 1 to validate the data pipeline and ship a useful tool early.

Two arb shapes:
  1. Cross-platform: buy Yes on the cheaper platform, buy No (or the opposing
     outcome) on the other, when the two prices sum to < $1 minus fees.
  2. Internal multi-outcome: a mutually-exclusive set whose cheapest-to-cover
     prices sum to < $1.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import MarketQuote


@dataclass
class ArbOpportunity:
    event_key: str
    legs: list[tuple[str, str, float]]   # (platform, side, price)
    guaranteed_profit: float             # fraction of $1 staked, after fees
    description: str


def scan_cross_platform(
    quotes: list[MarketQuote],
    min_profit: float = 0.01,
    fee: float = 0.0,
) -> list[ArbOpportunity]:
    """Find two-leg arbs across platforms for the same ``event_key``.

    TODO(phase1):
      - Group quotes by event_key.
      - For each Yes outcome on platform A, find the opposing No (= 1 - yes_ask)
        on platform B. If yes_ask_A + no_ask_B < 1 - fee, it's an arb.
      - Account for orderbook depth so the profit is actually fillable.
    """
    by_event: dict[str, list[MarketQuote]] = {}
    for q in quotes:
        if q.event_key:
            by_event.setdefault(q.event_key, []).append(q)

    opportunities: list[ArbOpportunity] = []
    # TODO(phase1): implement the pairwise comparison described above.
    _ = (by_event, min_profit, fee)
    return opportunities
