"""Tests for the real-time book logic (no network)."""

from kalshi_optimizer.config import Config
from kalshi_optimizer.realtime import LiveBook, _to_prob, handle_message
from kalshi_optimizer.types import MarketQuote, Prediction


def test_to_prob_cents():
    assert _to_prob(56) == 0.56
    assert _to_prob(0) is None      # no resting order
    assert _to_prob(None) is None


def test_handle_ticker_updates_book():
    book = LiveBook()
    book.markets["M-AAA"] = MarketQuote("kalshi", "M-AAA", "g", None, None,
                                        sport="soccer", event_key="M", outcome="AAA")
    handle_message({"type": "ticker_v2",
                    "msg": {"market_ticker": "M-AAA", "yes_bid": 54, "yes_ask": 56}}, book)
    assert book.markets["M-AAA"].yes_bid == 0.54
    assert book.markets["M-AAA"].yes_ask == 0.56


def test_seed_preserves_live_prices():
    book = LiveBook()
    book.markets["M-AAA"] = MarketQuote("kalshi", "M-AAA", "g", 0.5, 0.52,
                                        sport="soccer", event_key="M", outcome="AAA")
    fresh = MarketQuote("kalshi", "M-AAA", "g", None, None,
                        sport="soccer", event_key="M", outcome="AAA")
    book.seed([fresh], [])
    assert book.markets["M-AAA"].yes_bid == 0.5  # kept the live price


def test_book_edges_from_live_prices():
    book = LiveBook()
    book.markets["KXWCGAME-E-AAA"] = MarketQuote(
        "kalshi", "KXWCGAME-E-AAA", "A vs B Winner? [A]", 0.54, 0.56,
        sport="soccer", event_key="KXWCGAME-E", outcome="AAA")
    book.preds = [Prediction("KXWCGAME-E", "AAA", 0.75, "soccer", "m")]
    ideas = book.edges(Config())
    assert len(ideas) == 1 and ideas[0].edge > 0.03
