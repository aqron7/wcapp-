"""Tests for the cross-market parlay optimizer."""

from kalshi_optimizer.engine.optimizer import optimize_parlays

EDGES = [
    {"event_ticker": "E1", "market_id": "E1-A", "side": "yes", "price": 0.5, "fair_side": 0.6, "edge": 0.08, "title": "A"},
    {"event_ticker": "E2", "market_id": "E2-B", "side": "yes", "price": 0.4, "fair_side": 0.5, "edge": 0.08, "title": "B"},
    {"event_ticker": "E3", "market_id": "E3-C", "side": "yes", "price": 0.3, "fair_side": 0.35, "edge": 0.04, "title": "C"},
]


def test_optimize_ranks_by_roi_and_keeps_legs_independent():
    res = optimize_parlays(EDGES, max_legs=3, min_leg_edge=0.03)
    assert res
    top = res[0]
    assert top["roi"] > 0
    evs = [l["event_ticker"] for l in top["legs"]]
    assert len(evs) == len(set(evs))   # no two legs from the same game


def test_low_edge_excluded_and_event_deduped():
    edges = [
        {"event_ticker": "E1", "market_id": "E1-A", "side": "yes", "price": 0.5, "fair_side": 0.6, "edge": 0.08, "title": "A"},
        {"event_ticker": "E1", "market_id": "E1-B", "side": "no", "price": 0.5, "fair_side": 0.55, "edge": 0.05, "title": "A2"},
        {"event_ticker": "E2", "market_id": "E2-C", "side": "yes", "price": 0.4, "fair_side": 0.42, "edge": 0.02, "title": "C"},
    ]
    # E1 dedups to one leg, E2 is below min edge -> no 2-leg parlay possible.
    assert optimize_parlays(edges, max_legs=3, min_leg_edge=0.03) == []
