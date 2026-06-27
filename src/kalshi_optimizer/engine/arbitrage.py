"""Cross-platform arbitrage scanner (strategy C).

Needs no model — just consistent ``event_key``s across platforms (see
normalize.py). Built in phase 1 to validate the data pipeline and ship a useful
tool early.

The core insight for binary contracts: a "Yes" on platform A and the opposing
"No" on platform B cover both states of the world. Buying Yes costs
``yes_ask_A``; buying No costs ``1 - yes_bid_B`` (you sell Yes / buy No against
the bid). If those two costs sum to less than $1 (minus fees), you have locked a
guaranteed profit regardless of outcome.

    cost = yes_ask_A + (1 - yes_bid_B)
    profit_fraction = 1 - cost - fee
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import MarketQuote, Side


@dataclass
class ArbOpportunity:
    event_key: str
    legs: list[tuple[str, str, float]]   # (platform, side, price) each leg
    guaranteed_profit: float             # fraction of $1 staked, after fees
    description: str


def _pair_profit(buy_yes: MarketQuote, buy_no_against: MarketQuote, fee: float) -> float | None:
    """Profit from buying Yes on one market and No on another (None if not fillable)."""
    if buy_yes.yes_ask is None or buy_no_against.yes_bid is None:
        return None
    cost = buy_yes.yes_ask + (1.0 - buy_no_against.yes_bid)
    return 1.0 - cost - fee


def scan_cross_platform(
    quotes: list[MarketQuote],
    min_profit: float = 0.01,
    fee: float = 0.0,
) -> list[ArbOpportunity]:
    """Find two-leg arbs across platforms for the same ``event_key``.

    For every event, considers both directions (Yes on A / No on B, and vice
    versa) for each cross-platform pair, keeping only those clearing
    ``min_profit``. Markets without an event_key are ignored (never mismatched).
    """
    by_event: dict[str, list[MarketQuote]] = {}
    for q in quotes:
        if q.event_key:
            by_event.setdefault(q.event_key, []).append(q)

    opportunities: list[ArbOpportunity] = []
    for ekey, group in by_event.items():
        for i in range(len(group)):
            for j in range(len(group)):
                if i == j:
                    continue
                a, b = group[i], group[j]
                if a.platform == b.platform:
                    continue  # cross-platform only
                profit = _pair_profit(a, b, fee)
                if profit is None or profit < min_profit:
                    continue
                opportunities.append(
                    ArbOpportunity(
                        event_key=ekey,
                        legs=[
                            (a.platform, Side.YES.value, a.yes_ask),
                            (b.platform, Side.NO.value, round(1.0 - b.yes_bid, 4)),
                        ],
                        guaranteed_profit=round(profit, 4),
                        description=(
                            f"Buy YES @{a.yes_ask:.2f} on {a.platform} + "
                            f"NO @{1.0 - b.yes_bid:.2f} on {b.platform} "
                            f"=> {profit * 100:.1f}% locked"
                        ),
                    )
                )
    # Best first, and de-duplicate symmetric pairs by keeping the top per event.
    opportunities.sort(key=lambda o: o.guaranteed_profit, reverse=True)
    return opportunities
