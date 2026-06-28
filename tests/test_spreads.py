"""Tests for spread (handicap / run-line) modeling."""

from kalshi_optimizer.engine.value import spread_predictions
from kalshi_optimizer.types import MarketQuote


def _winner(code, label):
    return MarketQuote("kalshi", f"KXWCGAME-G-{code}", "Spain vs Austria Winner?",
                       0.5, 0.52, sport="soccer", event_key="KXWCGAME-G",
                       outcome=code, outcome_label=label, market_type="winner")


def _spread(code, strike):
    return MarketQuote("kalshi", f"KXWCSPREAD-G-{code}", f"{code} wins by more?",
                       0.3, 0.32, sport="soccer", event_key="KXWCSPREAD-G",
                       outcome=code, outcome_label=f"{code} handicap", market_type="spread",
                       strike=strike)


def test_spread_favors_stronger_team():
    quotes = [_winner("ESP", "Spain"), _winner("AUT", "Austria"),
              _spread("ESP2", 1.5), _spread("AUT2", 1.5)]
    preds = {p.outcome: p.fair_prob for p in spread_predictions(quotes, "soccer")}
    assert "ESP2" in preds and "AUT2" in preds
    assert 0 < preds["AUT2"] < preds["ESP2"] < 1   # Spain likelier to cover -1.5


def test_mlb_spread_uses_run_matrix():
    def w(code):
        return MarketQuote("kalshi", f"KXMLBGAME-G-{code}", "LAD vs COL Winner?",
                           0.5, 0.52, sport="mlb", event_key="KXMLBGAME-G",
                           outcome=code, outcome_label=code, market_type="winner")

    def s(code, strike):
        return MarketQuote("kalshi", f"KXMLBSPREAD-G-{code}", "spread", 0.3, 0.32,
                           sport="mlb", event_key="KXMLBSPREAD-G", outcome=code,
                           outcome_label=code, market_type="spread", strike=strike)

    quotes = [w("LAD"), w("COL"), s("LAD2", 1.5), s("COL2", 1.5)]
    preds = {p.outcome: p.fair_prob for p in spread_predictions(quotes, "mlb")}
    # LAD (much stronger) covers -1.5 more often than COL.
    assert preds["LAD2"] > preds["COL2"]
