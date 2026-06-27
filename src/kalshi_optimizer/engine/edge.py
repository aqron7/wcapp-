"""Edge / expected-value calculation for binary contracts."""

from __future__ import annotations

from ..types import Side


def expected_value(fair_prob: float, price: float, side: Side, fee: float = 0.0) -> float:
    """EV per $1 contract.

    Buying Yes at ``price`` pays out $1 if the event happens. Ignoring fees the
    EV simplifies to (fair_prob - price). Fee is subtracted as a flat cost.
    """
    if side is Side.YES:
        ev = fair_prob - price
    else:  # buying No at (1 - price) implied
        ev = (1.0 - fair_prob) - (1.0 - price)
    return ev - fee


def best_side(fair_prob: float, price: float, fee: float = 0.0) -> tuple[Side, float]:
    """Return the side with positive edge (if any) and its EV."""
    yes_ev = expected_value(fair_prob, price, Side.YES, fee)
    no_ev = expected_value(fair_prob, price, Side.NO, fee)
    if yes_ev >= no_ev:
        return Side.YES, yes_ev
    return Side.NO, no_ev
