"""Tests for market parsing and the arb scanner.

All run against recorded fixtures / synthetic quotes — no network.
"""

import json
from pathlib import Path

from kalshi_optimizer.data import kalshi, polymarket
from kalshi_optimizer.engine.arbitrage import scan_cross_platform
from kalshi_optimizer.types import MarketQuote

FIX = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIX / name).read_text())


def test_kalshi_parse_reads_dollar_prices_and_outcome():
    quotes = kalshi.parse_markets(_load("kalshi_markets.json"), "mlb")
    assert len(quotes) == 4  # settled game dropped
    nyy = next(q for q in quotes if q.outcome == "NYY")
    assert nyy.yes_ask == 0.56            # parsed from yes_ask_dollars string
    assert nyy.event_key == "KXMLBGAME-26JUN271905NYYBOS"  # shared event_ticker
    assert nyy.outcome_label == "New York Y"


def test_kalshi_zero_price_is_no_liquidity():
    payload = {"markets": [{
        "ticker": "KXMLBGAME-X-AAA", "event_ticker": "KXMLBGAME-X",
        "title": "t", "status": "active",
        "yes_bid_dollars": "0.0000", "yes_ask_dollars": "0.0000",
        "yes_sub_title": "AAA",
    }]}
    q = kalshi.parse_markets(payload, "mlb")[0]
    assert q.yes_bid is None and q.yes_ask is None


def test_polymarket_parse_filters_to_sport():
    quotes = polymarket.parse_markets(_load("polymarket_markets.json"), "mlb")
    assert len(quotes) == 2  # weather market filtered out
    assert any("Yankees" in q.title for q in quotes)


def test_cross_platform_arb_logic():
    # Same event on two platforms (shared event_key), YES cheap on A vs bid on B.
    quotes = [
        MarketQuote("kalshi", "k1", "game", yes_bid=0.54, yes_ask=0.56, event_key="E"),
        MarketQuote("polymarket", "p1", "game", yes_bid=0.60, yes_ask=0.62, event_key="E"),
    ]
    arbs = scan_cross_platform(quotes, min_profit=0.01, fee=0.0)
    # Buy YES @0.56 (kalshi) + NO @0.40 (poly) = 0.96 -> 4% locked.
    assert len(arbs) == 1
    assert round(arbs[0].guaranteed_profit, 2) == 0.04
    assert {leg[0] for leg in arbs[0].legs} == {"kalshi", "polymarket"}


def test_fee_erodes_marginal_arb():
    quotes = [
        MarketQuote("kalshi", "k1", "game", yes_bid=0.54, yes_ask=0.56, event_key="E"),
        MarketQuote("polymarket", "p1", "game", yes_bid=0.60, yes_ask=0.62, event_key="E"),
    ]
    assert scan_cross_platform(quotes, min_profit=0.01, fee=0.05) == []
