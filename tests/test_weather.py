"""Tests for weather totals adjustment and recent-form blending."""

from kalshi_optimizer.engine.value import mlb_totals_predictions
from kalshi_optimizer.models.props import _blend_recent
from kalshi_optimizer.types import MarketQuote
from kalshi_optimizer.weather import PARK_COORDS, totals_multiplier


def test_totals_multiplier_temp_and_rain():
    assert totals_multiplier(90, 0) > 1.0       # hot -> more runs
    assert totals_multiplier(50, 0) < 1.0       # cold -> fewer
    assert totals_multiplier(75, 2.0) < totals_multiplier(75, 0)  # rain suppresses


def test_parks_present():
    assert "COL" in PARK_COORDS and "NYY" in PARK_COORDS


def test_blend_recent():
    assert _blend_recent(10.0, 14.0) == 12.0    # halfway
    assert _blend_recent(10.0, None) == 10.0    # no recent -> season
    assert _blend_recent(10.0, 0) == 10.0       # zero recent ignored


def test_weather_totals_override_scales_with_multiplier():
    def w(code):
        return MarketQuote("kalshi", f"KXMLBGAME-26JUN281415MIASTL-{code}",
                           "Miami vs St. Louis Winner?", 0.5, 0.52, sport="mlb",
                           event_key="KXMLBGAME-26JUN281415MIASTL", outcome=code,
                           outcome_label={"MIA": "Miami", "STL": "St. Louis"}[code],
                           market_type="winner")
    total = MarketQuote("kalshi", "KXMLBTOTAL-26JUN281415MIASTL-9", "Total Runs?",
                        0.5, 0.52, sport="mlb", event_key="KXMLBTOTAL-26JUN281415MIASTL",
                        outcome="9", outcome_label="Over 8.5", market_type="total", strike=8.5)
    quotes = [w("MIA"), w("STL"), total]

    hot = {p.outcome: p.fair_prob for p in mlb_totals_predictions(quotes, lambda c, d, h: 1.08)}
    cold = {p.outcome: p.fair_prob for p in mlb_totals_predictions(quotes, lambda c, d, h: 0.92)}
    assert hot["9"] > cold["9"]   # warmer -> higher P(over)
