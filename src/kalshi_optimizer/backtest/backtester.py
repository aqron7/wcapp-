"""Backtester — the validation gate (phase 3).

No real or automated trades until the model clears this bar:
  - good calibration (low Brier score / log-loss), and
  - positive **closing-line value (CLV)** — our entry consistently beats the
    market's closing price, the single best predictor of long-run edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error of probabilistic predictions (lower is better)."""
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: list[float], outcomes: list[int], eps: float = 1e-9) -> float:
    """Negative log-likelihood (lower is better)."""
    if not probs:
        return float("nan")
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(1 - eps, max(eps, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(probs)


def closing_line_value(entry_prices: list[float], closing_prices: list[float]) -> float:
    """Average edge of our entry vs the closing price (higher is better).

    Positive mean CLV is the go/no-go signal for trading real money.
    """
    if not entry_prices:
        return float("nan")
    return sum(c - e for e, c in zip(entry_prices, closing_prices)) / len(entry_prices)


@dataclass
class BacktestResult:
    n: int
    brier: float
    log_loss: float
    mean_clv: float

    @property
    def passes_gate(self) -> bool:
        """Conservative gate: positive CLV and a calibrated Brier score."""
        return self.mean_clv > 0 and self.brier < 0.25


def run_backtest(
    probs: list[float],
    outcomes: list[int],
    entry_prices: list[float],
    closing_prices: list[float],
) -> BacktestResult:
    """Compute the full validation report over historical predictions."""
    return BacktestResult(
        n=len(probs),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        mean_clv=closing_line_value(entry_prices, closing_prices),
    )


def score_from_db(db_path: str | None = None, min_edge: float = 0.03) -> BacktestResult:
    """Build the validation report from the snapshot DB.

    For each market it derives an entry (first liquid snapshot with a model
    fair value), a closing price (last liquid snapshot), and the settled
    result. Only markets where the model had an edge >= ``min_edge`` at entry
    count toward the score. Prices are expressed in terms of the side we'd have
    bought, so positive mean CLV means our entry beat the closing line.
    """
    from collections import defaultdict

    from .. import storage

    conn = storage.connect(db_path or storage.DEFAULT_DB)
    cur = conn.execute(
        "SELECT market_id, ts, yes_bid, yes_ask, model_fair, status, result "
        "FROM snapshots ORDER BY ts"
    )
    markets: dict[str, dict] = defaultdict(lambda: {"entries": [], "result": None})
    for market_id, ts, yes_bid, yes_ask, fair, status, result in cur:
        m = markets[market_id]
        if status == "settled" and result in ("yes", "no"):
            m["result"] = result
        elif status == "active" and yes_bid is not None and yes_ask is not None and fair is not None:
            m["entries"].append((ts, (yes_bid + yes_ask) / 2.0, fair))

    probs: list[float] = []
    outcomes: list[int] = []
    entry_prices: list[float] = []
    closing_prices: list[float] = []
    for m in markets.values():
        if m["result"] is None or not m["entries"]:
            continue
        m["entries"].sort()
        _, entry_yes, fair = m["entries"][0]
        _, closing_yes, _ = m["entries"][-1]
        if abs(fair - entry_yes) < min_edge:
            continue  # we wouldn't have bet this market
        bet_yes = fair >= entry_yes
        probs.append(fair)
        outcomes.append(1 if m["result"] == "yes" else 0)
        # Express prices for the side we bought (YES price, or the NO price).
        entry_prices.append(entry_yes if bet_yes else 1.0 - entry_yes)
        closing_prices.append(closing_yes if bet_yes else 1.0 - closing_yes)

    return run_backtest(probs, outcomes, entry_prices, closing_prices)
