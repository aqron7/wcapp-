"""Tests for de-vig + sharpen, and that edges are no longer one-directional."""

from kalshi_optimizer.config import Config
from kalshi_optimizer.engine.calibration import devig, sharpen
from kalshi_optimizer.engine.value import find_value_edges
from kalshi_optimizer.types import MarketQuote, Prediction, Side


def test_devig_normalizes_to_one():
    out = devig({"A": 0.55, "B": 0.30, "TIE": 0.25})  # sums to 1.10 (10% vig)
    assert abs(sum(out.values()) - 1.0) < 1e-9
    assert out["A"] < 0.55  # inflated price pulled down


def test_sharpen_raises_favorites_lowers_dogs():
    assert sharpen(0.85, 1.25) > 0.85
    assert sharpen(0.15, 1.25) < 0.15
    assert sharpen(0.5, 1.25) == 0.5
    assert sharpen(0.7, 1.0) == 0.7  # no-op


def test_edges_can_back_an_underdog():
    # Market makes A the underdog (0.30); the model thinks A is much stronger.
    quotes = [
        MarketQuote("kalshi", "E-A", "A vs B Winner?", 0.29, 0.31,
                    sport="soccer", event_key="E", outcome="A"),
        MarketQuote("kalshi", "E-B", "A vs B Winner?", 0.69, 0.71,
                    sport="soccer", event_key="E", outcome="B"),
    ]
    preds = [Prediction("E", "A", 0.55, "soccer", "m"),
             Prediction("E", "B", 0.45, "soccer", "m")]
    ideas = find_value_edges(quotes, preds, Config())
    assert ideas, "expected an edge"
    top = ideas[0]
    # The recommendation backs the underdog A with YES — not a favorite-fade.
    assert top.market_id == "E-A" and top.side is Side.YES
