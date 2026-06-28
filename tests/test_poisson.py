"""Tests for the Poisson totals model + totals edge path."""

from kalshi_optimizer.config import Config
from kalshi_optimizer.engine.value import find_value_edges, predictions_for_sport
from kalshi_optimizer.models.poisson import prob_over, totals_predictions
from kalshi_optimizer.types import MarketQuote, Side


def test_prob_over_monotonic_decreasing():
    assert prob_over(2.75, 1.5) > prob_over(2.75, 2.5) > prob_over(2.75, 5.5)
    assert 0 < prob_over(2.75, 2.5) < 1


def _total_quote(strike, suffix, bid, ask):
    return MarketQuote("kalshi", f"KXWCTOTAL-E-{suffix}", "Will over X be scored?",
                       bid, ask, sport="soccer", event_key="KXWCTOTAL-E",
                       outcome=suffix, outcome_label=f"Over {strike}",
                       market_type="total", strike=strike)


def test_totals_predictions_use_strike():
    quotes = [_total_quote(2.5, "3", 0.49, 0.52), _total_quote(3.5, "4", 0.29, 0.31)]
    preds = {p.outcome: p.fair_prob for p in totals_predictions(quotes, "soccer")}
    assert preds["3"] > preds["4"]                 # over 2.5 likelier than over 3.5
    assert abs(preds["3"] - prob_over(2.75, 2.5)) < 1e-9


def test_totals_edge_when_market_underprices_over():
    # Model P(over 2.5) ~ 0.52 at lambda 2.75; market sells YES well below fair.
    quotes = [_total_quote(2.5, "3", 0.36, 0.38)]
    preds = predictions_for_sport("soccer", quotes)
    ideas = find_value_edges(quotes, preds, Config())
    assert ideas and ideas[0].side is Side.YES
    assert "KXWCTOTAL-E-3" == ideas[0].market_id
