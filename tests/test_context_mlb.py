"""Tests for MLB ratings, ballpark altitude, and the recent-form engine."""

from kalshi_optimizer import form, storage
from kalshi_optimizer.models.baseball import BaseballModel
from kalshi_optimizer.models.context import BALLPARK_ALTITUDE_M


def test_mlb_ratings_seeded_and_ordered():
    m = BaseballModel()
    assert m.elo.rating("LAD") > m.elo.rating("COL")  # strong vs weak


def test_coors_is_high_altitude():
    assert BALLPARK_ALTITUDE_M["COL"] > 1000
    assert BALLPARK_ALTITUDE_M["MIA"] < 100


def test_recent_form_from_db(tmp_path):
    conn = storage.connect(str(tmp_path / "f.db"))
    storage.insert_snapshots(conn, [
        ("t1", "mlb", "E1", "E1-NYY", "NYY", None, None, None, "settled", "yes"),
        ("t2", "mlb", "E2", "E2-NYY", "NYY", None, None, None, "settled", "yes"),
        ("t3", "mlb", "E3", "E3-BOS", "BOS", None, None, None, "settled", "no"),
    ])
    f = form.recent_form(conn, "mlb")
    assert f["NYY"] == 1.0   # won both
    assert f["BOS"] == 0.0   # lost


def test_form_shifts_mlb_prediction():
    from kalshi_optimizer.engine.value import predictions_for_sport
    from kalshi_optimizer.types import MarketQuote

    quotes = [
        MarketQuote("kalshi", "KXMLBGAME-E-NYY", "New York Y vs Boston Winner?",
                    0.5, 0.52, sport="mlb", event_key="KXMLBGAME-E", outcome="NYY",
                    outcome_label="New York Y"),
        MarketQuote("kalshi", "KXMLBGAME-E-BOS", "New York Y vs Boston Winner?",
                    0.48, 0.5, sport="mlb", event_key="KXMLBGAME-E", outcome="BOS",
                    outcome_label="Boston"),
    ]
    base = {p.outcome: p.fair_prob for p in predictions_for_sport("mlb", quotes)}
    hot = {p.outcome: p.fair_prob
           for p in predictions_for_sport("mlb", quotes, form={"NYY": 1.0, "BOS": 0.0})}
    assert hot["NYY"] > base["NYY"]  # strong recent form lifts NYY
