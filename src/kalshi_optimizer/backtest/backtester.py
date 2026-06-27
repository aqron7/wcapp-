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
    """Compute the full validation report over historical predictions.

    TODO(phase3):
      - Load historical model predictions + actual results + Kalshi closing
        prices from the snapshot DB.
      - Optionally bucket by sport / market type for per-segment CLV.
    """
    return BacktestResult(
        n=len(probs),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        mean_clv=closing_line_value(entry_prices, closing_prices),
    )
