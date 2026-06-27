"""Fair-value blending.

Combine our model's probability with free market consensus (Polymarket / The
Odds API) into a single fair probability per outcome. Since we avoid a paid
sharp feed, the model is primary and market prices act as a calibration anchor.
"""

from __future__ import annotations


def blend(
    model_prob: float,
    market_prob: float | None,
    model_weight: float = 0.6,
) -> float:
    """Weighted blend of model and market probability.

    With no market reference available, returns the model probability.
    ``model_weight`` should drop as the model proves itself (or rise as it does,
    depending on measured calibration — tune via the backtester).
    """
    if market_prob is None:
        return model_prob
    w = max(0.0, min(1.0, model_weight))
    return w * model_prob + (1.0 - w) * market_prob
