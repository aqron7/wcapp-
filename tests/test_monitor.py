"""Tests for calibration backtest, live positions, and the calendar."""

from kalshi_optimizer import ledger, storage
from kalshi_optimizer.backtest.model_backtest import report, totals_calibration, walk_forward
from kalshi_optimizer.models.poisson import prob_over


def test_totals_calibration_scores_settled_overs():
    # Two settled MLB totals: over 8.5 (went over) and over 11.5 (stayed under).
    settled = [
        {"floor_strike": 8.5, "result": "yes", "event_ticker": "E1"},
        {"floor_strike": 11.5, "result": "no", "event_ticker": "E2"},
        {"floor_strike": 0, "result": "", "event_ticker": "E3"},  # unsettled -> skipped
    ]
    m = totals_calibration(settled, lambda line, ev: prob_over(8.6, line))
    assert m["n"] == 2
    assert "brier" in m and 0 <= m["brier"] <= 1


def test_walk_forward_beats_coin_flip_when_signal_exists():
    # 'a' always beats 'b' -> model should learn it and beat the 0.5 baseline.
    games = [("a", "b", 1.0)] * 40
    r = report(games, {"a": 1500, "b": 1500}, k=20)
    assert r["n"] == 40
    assert r["brier"] < r["baseline_brier"]


def test_walk_forward_predicts_before_update():
    preds, outs = walk_forward([("a", "b", 1.0), ("a", "b", 1.0)], {"a": 1500, "b": 1500})
    assert preds[0] == 0.5            # first game predicted from equal priors
    assert preds[1] > 0.5            # after a win, 'a' favored


def test_live_positions_mark_to_market(tmp_path):
    conn = storage.connect(str(tmp_path / "b.db"))
    leg = {"market_id": "M1", "event_ticker": "E", "side": "yes",
           "price": 0.50, "fair_side": 0.6, "label": "AAA"}
    ledger.record_bet(conn, [leg], stake=10.0)
    pos = ledger.live_positions(conn, {"M1": 0.60})   # price rose 0.50 -> 0.60
    assert len(pos) == 1
    assert pos[0]["current"] == 0.60
    assert pos[0]["unrealized"] == 2.0               # $10 * (0.60/0.50 - 1)


def test_calendar_groups_by_date(tmp_path):
    conn = storage.connect(str(tmp_path / "b.db"))
    storage.insert_bet(conn, {"placed_ts": "2026-06-28T10:00", "kind": "single",
                              "sport": "mlb", "description": "x", "legs_json": "[]",
                              "stake": 10, "entry_price": 0.5, "fair_prob": 0.6, "edge": 0.1,
                              "status": "won", "payout": 20, "pnl": 10,
                              "closing_price": None, "result_ts": None})
    storage.insert_bet(conn, {"placed_ts": "2026-06-28T12:00", "kind": "single",
                              "sport": "mlb", "description": "y", "legs_json": "[]",
                              "stake": 5, "entry_price": 0.5, "fair_prob": 0.6, "edge": 0.1,
                              "status": "pending", "payout": None, "pnl": None,
                              "closing_price": None, "result_ts": None})
    cal = ledger.calendar(conn)
    assert len(cal) == 1
    assert cal[0]["count"] == 2 and cal[0]["pnl"] == 10 and cal[0]["pending"] == 1
