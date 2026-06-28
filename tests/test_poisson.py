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
    # MLB totals use flat Poisson (soccer totals now go through Dixon-Coles).
    # Model P(over 8.5) ~ 0.54 at lambda 8.6; market sells YES well below fair.
    q = MarketQuote("kalshi", "KXMLBTOTAL-E-9", "Total Runs?", 0.36, 0.38,
                    sport="mlb", event_key="KXMLBTOTAL-E", outcome="9",
                    outcome_label="Over 8.5", market_type="total", strike=8.5)
    preds = predictions_for_sport("mlb", [q])
    ideas = find_value_edges([q], preds, Config())
    assert ideas and ideas[0].side is Side.YES
    assert ideas[0].market_id == "KXMLBTOTAL-E-9"
