"""Calibration: de-vig market prices and adjust model dispersion.

Two fixes for the "only ever fades the favorite" failure mode:

1. De-vig — a multi-outcome market's YES prices sum to >1 (the house margin).
   Comparing a model (which sums to 1) against raw prices makes *every* outcome
   look overpriced, biasing toward "buy NO" on the favorite. Normalizing the
   event's outcome prices to sum to 1 removes that structural bias.

2. Sharpen — a seeded/compressed model is under-confident on favorites, a
   one-directional error. Raising probabilities away from 0.5 (temperature < 1,
   i.e. gamma > 1) makes the model's confidence match reality, so residual
   disagreements become two-sided. gamma should ultimately be *fit* from settled
   results (log-loss); until then it's a modest documented default.
"""

from __future__ import annotations


def devig(outcome_probs: dict[str, float]) -> dict[str, float]:
    """Normalize an event's per-outcome implied probabilities to sum to 1."""
    total = sum(v for v in outcome_probs.values() if v is not None)
    if total <= 0:
        return outcome_probs
    return {k: (v / total if v is not None else None) for k, v in outcome_probs.items()}


def sharpen(p: float, gamma: float) -> float:
    """Push a probability away from (gamma>1) or toward (gamma<1) 0.5.

    Treats ``p`` as a binary this-outcome-vs-not probability (consistent with how
    each Kalshi market is evaluated). gamma == 1 is a no-op.
    """
    if gamma == 1.0 or p <= 0.0 or p >= 1.0:
        return p
    a = p ** gamma
    b = (1.0 - p) ** gamma
    return a / (a + b)
