"""Tests for market parsing, event-key matching, and the arb scanner.

All run against recorded fixtures — no network — mirroring the real Kalshi and
Polymarket API response shapes.
"""

import json
from pathlib import Path

from kalshi_optimizer.data import kalshi, polymarket
from kalshi_optimizer.engine.arbitrage import scan_cross_platform
from kalshi_optimizer.normalize import event_key

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


def test_event_key_is_order_independent():
    a = event_key("Will the New York Yankees beat the Boston Red Sox?", "mlb", None)
    b = event_key("Red Sox vs Yankees", "mlb", None)
    assert a is not None and a.endswith("BOS+NYY")
    # date differs (None here) but team-pair portion matches
    assert a.split("|")[-1] == b.split("|")[-1]


def test_event_key_none_without_two_teams():
    assert event_key("Will it rain in Seattle?", "mlb", None) is None


def test_kalshi_parse_skips_settled_and_converts_cents():
    quotes = kalshi.parse_markets(_load("kalshi_markets.json"), "mlb")
    assert len(quotes) == 2  # settled Cubs/Reds market dropped
    yanks = next(q for q in quotes if "Yankees" in q.title)
    assert yanks.yes_ask == 0.56  # 56 cents -> 0.56
    assert yanks.event_key.endswith("BOS+NYY")


def test_polymarket_parse_filters_to_sport():
    quotes = polymarket.parse_markets(_load("polymarket_markets.json"), "mlb")
    titles = [q.title for q in quotes]
    assert len(quotes) == 2  # weather market filtered out
    assert any("Yankees" in t for t in titles)


def test_cross_platform_arb_detected():
    quotes = kalshi.parse_markets(_load("kalshi_markets.json"), "mlb")
    quotes += polymarket.parse_markets(_load("polymarket_markets.json"), "mlb")

    arbs = scan_cross_platform(quotes, min_profit=0.01, fee=0.0)
    # Yankees: YES @0.56 (kalshi) + NO @0.40 (poly) = 0.96 -> 4% locked.
    assert len(arbs) == 1
    arb = arbs[0]
    assert arb.event_key.endswith("BOS+NYY")
    assert round(arb.guaranteed_profit, 2) == 0.04
    platforms = {leg[0] for leg in arb.legs}
    assert platforms == {"kalshi", "polymarket"}


def test_fee_erodes_marginal_arb():
    quotes = kalshi.parse_markets(_load("kalshi_markets.json"), "mlb")
    quotes += polymarket.parse_markets(_load("polymarket_markets.json"), "mlb")
    # A 5% fee wipes out the 4% edge.
    assert scan_cross_platform(quotes, min_profit=0.01, fee=0.05) == []
