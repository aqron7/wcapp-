"""Poisson totals model: P(total goals/runs over a line).

Kalshi totals are a ladder of "Over X.5" binaries, each carrying ``floor_strike``
(the line). YES wins if the game total exceeds it. Modeling the total as Poisson:

    P(total > line) = P(X >= floor(line)+1) = 1 - CDF(floor(line))

The mean total (lambda) is seeded at the league average and nudged by how
lopsided the matchup is (mismatches tend to produce a few more goals/runs).
Seeds are approximations — validate via the backtester before trusting. A real
upgrade uses team attack/defence strengths (Dixon-Coles) instead of a flat mean.
"""

from __future__ import annotations

from math import exp, factorial, floor

from ..types import MarketQuote, Prediction

# League-average game totals (both teams combined).
AVG_TOTAL = {"soccer": 2.75, "mlb": 8.6}


def poisson_cdf(lmbda: float, k: int) -> float:
    return sum(exp(-lmbda) * lmbda ** i / factorial(i) for i in range(0, k + 1))


def prob_over(lmbda: float, line: float) -> float:
    """P(total > line) for a half-integer line (e.g. 2.5 -> P(X >= 3))."""
    k = int(floor(line))
    return max(0.0, min(1.0, 1.0 - poisson_cdf(lmbda, k)))


def totals_predictions(quotes: list[MarketQuote], sport: str,
                       lmbda: float | None = None) -> list[Prediction]:
    """Predictions for total-type markets, keyed by (event_ticker, outcome)."""
    lmbda = lmbda or AVG_TOTAL.get(sport)
    if not lmbda:
        return []
    preds: list[Prediction] = []
    for q in quotes:
        if q.market_type != "total" or q.strike is None or not q.outcome or not q.event_key:
            continue
        preds.append(Prediction(q.event_key, q.outcome, prob_over(lmbda, q.strike),
                                sport, "poisson_total_v1"))
    return preds
