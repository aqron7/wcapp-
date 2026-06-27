"""Position sizing: fractional Kelly with exposure caps."""

from __future__ import annotations

from ..config import SizingConfig


def kelly_fraction(fair_prob: float, price: float) -> float:
    """Full-Kelly fraction of bankroll for a binary contract bought at ``price``.

    Net odds b = (1 - price) / price  (win $1-price per $price staked).
    Kelly f* = (b*p - q) / b, where p = fair_prob, q = 1 - p.
    Clamped to [0, 1]; 0 means no edge.
    """
    p = fair_prob
    q = 1.0 - p
    if price <= 0 or price >= 1:
        return 0.0
    b = (1.0 - price) / price
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))


def stake(
    fair_prob: float,
    price: float,
    bankroll: float,
    cfg: SizingConfig,
    current_total_exposure: float = 0.0,
) -> float:
    """USD stake after fractional Kelly and exposure caps."""
    f = kelly_fraction(fair_prob, price) * cfg.kelly_fraction
    raw = f * bankroll
    raw = min(raw, cfg.max_per_market)
    headroom = max(0.0, cfg.max_total_exposure - current_total_exposure)
    return round(min(raw, headroom), 2)
