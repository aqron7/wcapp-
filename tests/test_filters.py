"""Tests for the live-only filters: drop ended games and 0-chance markets."""

from kalshi_optimizer.config import Config
from kalshi_optimizer.data.kalshi import is_ended, parse_markets
from kalshi_optimizer.engine.value import find_value_edges
from kalshi_optimizer.types import MarketQuote, Prediction


def test_is_ended():
    assert is_ended("2020-01-01T00:00:00Z")       # long past -> ended
    assert not is_ended("2999-01-01T00:00:00Z")   # future -> live
    assert not is_ended(None)


def test_parse_captures_game_time():
    payload = {"markets": [{
        "ticker": "KXMLBGAME-26JUN281415MIASTL-STL", "event_ticker": "KXMLBGAME-26JUN281415MIASTL",
        "title": "x", "status": "active", "yes_bid_dollars": "0.5", "yes_ask_dollars": "0.52",
        "yes_sub_title": "St. Louis", "expected_expiration_time": "2026-06-28T21:15:00Z"}]}
    q = parse_markets(payload, "mlb", "winner")[0]
    assert q.game_time == "2026-06-28T21:15:00Z"


def test_zero_chance_market_skipped():
    quotes = [
        MarketQuote("kalshi", "E-A", "g", 0.01, 0.02, sport="soccer", event_key="E",
                    outcome="A", market_type="winner"),     # ~1c, near-decided
        MarketQuote("kalshi", "E-B", "g", 0.40, 0.42, sport="soccer", event_key="E",
                    outcome="B", market_type="winner"),
    ]
    preds = [Prediction("E", "A", 0.20, "soccer", "m"), Prediction("E", "B", 0.55, "soccer", "m")]
    ideas = find_value_edges(quotes, preds, Config())
    assert all(i.market_id != "E-A" for i in ideas)   # the 1c longshot is not a pick
