"""Tests for parlay math, the bet ledger, and context adjustments."""

from kalshi_optimizer import ledger, storage
from kalshi_optimizer.engine.parlay import combine, has_correlated_legs
from kalshi_optimizer.models.context import MatchContext, adjust
from kalshi_optimizer.models.soccer import SoccerModel


def test_parlay_combine_multiplies():
    legs = [{"price": 0.5, "fair_side": 0.6}, {"price": 0.4, "fair_side": 0.5}]
    c = combine(legs)
    assert c["price"] == 0.2 and c["fair"] == 0.3
    assert round(c["edge"], 4) == 0.1


def test_correlated_legs_detected():
    legs = [{"event_ticker": "E1"}, {"event_ticker": "E1"}]
    assert has_correlated_legs(legs)
    assert not has_correlated_legs([{"event_ticker": "E1"}, {"event_ticker": "E2"}])


def _seed_settled(conn, market_id, result):
    storage.insert_snapshots(conn, [
        ("t1", "soccer", "E", market_id, "X", 0.60, 0.62, 0.7, "active", None),
        ("t2", "soccer", "E", market_id, "X", None, None, None, "settled", result),
    ])


def test_single_bet_wins_and_pnl(tmp_path):
    conn = storage.connect(str(tmp_path / "b.db"))
    _seed_settled(conn, "KX-GAME-AAA", "yes")
    leg = {"market_id": "KX-GAME-AAA", "event_ticker": "KX-GAME",
           "outcome": "AAA", "side": "yes", "price": 0.50, "fair_side": 0.70,
           "label": "AAA"}
    ledger.record_bet(conn, [leg], stake=10.0, sport="soccer")

    assert ledger.settle_pending(conn) == 1
    s = ledger.summary(conn)
    # $10 at 0.50 -> 20 contracts, win pays $20, pnl +$10.
    assert s["pnl"] == 10.0
    assert s["n_settled"] == 1 and s["win_rate"] == 1.0


def test_single_bet_loses(tmp_path):
    conn = storage.connect(str(tmp_path / "b.db"))
    _seed_settled(conn, "KX-GAME-AAA", "no")  # we bet YES, result NO
    leg = {"market_id": "KX-GAME-AAA", "side": "yes", "price": 0.50,
           "fair_side": 0.70, "label": "AAA", "event_ticker": "KX-GAME"}
    ledger.record_bet(conn, [leg], stake=10.0)
    ledger.settle_pending(conn)
    assert ledger.summary(conn)["pnl"] == -10.0


def test_parlay_needs_all_legs(tmp_path):
    conn = storage.connect(str(tmp_path / "b.db"))
    _seed_settled(conn, "M1", "yes")
    _seed_settled(conn, "M2", "yes")
    legs = [
        {"market_id": "M1", "side": "yes", "price": 0.5, "fair_side": 0.6, "label": "a", "event_ticker": "E1"},
        {"market_id": "M2", "side": "yes", "price": 0.5, "fair_side": 0.6, "label": "b", "event_ticker": "E2"},
    ]
    ledger.record_bet(conn, legs, stake=10.0)
    ledger.settle_pending(conn)
    s = ledger.summary(conn)
    # cost 0.25 -> $10 pays $40 -> pnl +$30
    assert s["pnl"] == 30.0


def test_context_altitude_and_host():
    # Mexico (host, altitude-acclimatised) vs a lowland team in Mexico City.
    adj_mex, adj_other, why = adjust("Mexico", "Norway",
                                     MatchContext(venue_city="Mexico City"))
    assert adj_mex > adj_other  # Mexico advantaged by altitude + host
    factors = {f for f, _, _ in why}
    assert "altitude" in factors and "host" in factors


def test_context_none_is_neutral():
    a, b, why = adjust("Brazil", "Japan", None)
    assert a == 0 and b == 0


def test_contextual_probs_shift_toward_host():
    model = SoccerModel()
    base = model.match_probs("Mexico", "Germany")[0]
    (p_mex, _, _), _ = model.contextual_match_probs(
        "Mexico", "Germany", MatchContext(venue_city="Mexico City"))
    assert p_mex > base  # altitude + host help Mexico
