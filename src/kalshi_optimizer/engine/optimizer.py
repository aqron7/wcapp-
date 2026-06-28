"""Cross-sport / cross-market parlay optimizer.

Given the current single-leg edges, find the parlays with the best expected
return. A parlay's payoff is multiplicative, so the metric is expected ROI:

    roi = (product of leg win-probs) / (product of leg costs) - 1

Only +EV legs are used (a parlay of -EV legs is worse than its parts), legs are
de-duplicated to one per game and never share a game (independence), and leg
count is capped because variance compounds with every leg.
"""

from __future__ import annotations

from itertools import combinations
from math import prod


def optimize_parlays(edges: list[dict], max_legs: int = 3, min_leg_edge: float = 0.03,
                     top: int = 5) -> list[dict]:
    """Return the top parlays by expected ROI.

    ``edges`` items need: event_ticker, market_id, side, price, fair_side, edge,
    title (anything else is passed through on the legs).
    """
    pool_by_event: dict[str, dict] = {}
    for e in sorted((e for e in edges if e["edge"] >= min_leg_edge),
                    key=lambda x: x["edge"], reverse=True):
        pool_by_event.setdefault(e["event_ticker"], e)  # best edge per game
    pool = list(pool_by_event.values())

    out: list[dict] = []
    for r in range(2, min(max_legs, len(pool)) + 1):
        for combo in combinations(pool, r):
            fair = prod(l["fair_side"] for l in combo)
            cost = prod(l["price"] for l in combo)
            if cost <= 0:
                continue
            roi = fair / cost - 1.0
            out.append({
                "legs": list(combo),
                "n": r,
                "fair": round(fair, 4),
                "cost": round(cost, 4),
                "roi": round(roi, 4),
                "payout_mult": round(1.0 / cost, 2),
            })
    out.sort(key=lambda x: x["roi"], reverse=True)
    return out[:top]
