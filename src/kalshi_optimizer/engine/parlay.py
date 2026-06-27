"""Parlay math.

A parlay wins only if every leg wins. Under an independence assumption:
  combined fair prob  = product of each leg's (bought-side) win probability
  combined cost/price = product of each leg's (bought-side) entry price
  combined edge       = combined_fair - combined_price

Kalshi has no native parlay product, so a parlay here is a basket of single
contracts: buying all legs replicates the parlay payout (stake / combined cost
if every leg wins). Independence is an approximation — correlated legs (e.g. two
outcomes in the same match) violate it, so the UI warns when legs share a game.
"""

from __future__ import annotations

from math import prod


def combine(legs: list[dict]) -> dict:
    """Combine legs each having 'price' (bought-side cost) and 'fair_side' (win prob)."""
    prices = [float(leg["price"]) for leg in legs]
    fairs = [float(leg["fair_side"]) for leg in legs]
    price = prod(prices) if prices else 0.0
    fair = prod(fairs) if fairs else 0.0
    return {"price": round(price, 4), "fair": round(fair, 4), "edge": round(fair - price, 4)}


def has_correlated_legs(legs: list[dict]) -> bool:
    """True if two legs reference the same game (event_ticker) — independence breaks."""
    events = [leg.get("event_ticker") for leg in legs if leg.get("event_ticker")]
    return len(events) != len(set(events))
