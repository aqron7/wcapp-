"""Tests for the empirical MLB run-total distribution."""

from kalshi_optimizer.engine.value import mlb_totals_predictions
from kalshi_optimizer.models.mlb_totals import fit_distribution, prob_over
from kalshi_optimizer.types import MarketQuote


def test_fit_distribution_empirical_rates():
    settled = (
        [{"floor_strike": 7.5, "result": "yes"}] * 30
        + [{"floor_strike": 7.5, "result": "no"}] * 10      # 75% over 7.5
        + [{"floor_strike": 9.5, "result": "yes"}] * 10
        + [{"floor_strike": 9.5, "result": "no"}] * 30      # 25% over 9.5
    )
    d = fit_distribution(settled, min_n=20)
    assert round(d[7.5], 2) == 0.75
    assert round(d[9.5], 2) == 0.25
    assert prob_over(8.5, d) < d[7.5] and prob_over(8.5, d) > d[9.5]  # interpolated


def test_prob_over_falls_back_to_poisson_when_unfit():
    p = prob_over(8.5, {})       # no distribution -> Poisson(8.6)
    assert 0.4 < p < 0.6


def test_mlb_totals_predictions_uses_model():
    q = MarketQuote("kalshi", "KXMLBTOTAL-E-9", "Total Runs?", 0.5, 0.52, sport="mlb",
                    event_key="KXMLBTOTAL-E", outcome="9", outcome_label="Over 8.5",
                    market_type="total", strike=8.5)
    preds = mlb_totals_predictions([q])
    assert len(preds) == 1 and preds[0].outcome == "9"
