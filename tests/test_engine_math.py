"""Tests for the pure-math core (no network needed)."""

from kalshi_optimizer.backtest.backtester import brier_score, closing_line_value
from kalshi_optimizer.data.odds_api import devig_multiplicative
from kalshi_optimizer.engine.edge import best_side, expected_value
from kalshi_optimizer.engine.sizing import kelly_fraction
from kalshi_optimizer.types import Side


def test_expected_value_yes_positive_edge():
    # fair 0.65, price 0.58 -> +0.07 edge on Yes
    assert round(expected_value(0.65, 0.58, Side.YES), 4) == 0.07


def test_best_side_picks_no_when_overpriced():
    side, ev = best_side(fair_prob=0.40, price=0.60)
    assert side is Side.NO
    assert ev > 0


def test_kelly_zero_without_edge():
    # fair == price -> no edge -> zero stake
    assert kelly_fraction(0.5, 0.5) == 0.0


def test_kelly_positive_with_edge():
    assert kelly_fraction(0.65, 0.58) > 0


def test_devig_normalizes_to_one():
    out = devig_multiplicative({"home": 0.55, "away": 0.52})  # overround 1.07
    assert round(sum(out.values()), 6) == 1.0


def test_clv_positive_when_entry_beats_close():
    assert round(closing_line_value([0.50, 0.40], [0.55, 0.45]), 6) == 0.05


def test_brier_perfect_is_zero():
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0
