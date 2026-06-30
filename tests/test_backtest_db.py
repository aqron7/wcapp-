"""Tests for snapshot storage and backtester DB scoring (phase 3)."""

from kalshi_optimizer import storage
from kalshi_optimizer.backtest.backtester import (grouped_score_from_db,
                                                  realized_from_db, score_from_db)


def _seed(path):
    conn = storage.connect(path)
    # Market we bet (model 0.75 vs entry mid 0.55 -> edge), line moved our way
    # to 0.70, and it resolved YES.
    rows = [
        ("2026-06-27T10:00", "soccer", "E1", "M1", "AAA", 0.54, 0.56, 0.75, "active", None),
        ("2026-06-27T18:00", "soccer", "E1", "M1", "AAA", 0.69, 0.71, 0.75, "active", None),
        ("2026-06-28T00:00", "soccer", "E1", "M1", "AAA", None, None, None, "settled", "yes"),
        # Market with no edge (model ~ price) -> excluded from scoring.
        ("2026-06-27T10:00", "soccer", "E2", "M2", "BBB", 0.49, 0.51, 0.50, "active", None),
        ("2026-06-28T00:00", "soccer", "E2", "M2", "BBB", None, None, None, "settled", "no"),
    ]
    storage.insert_snapshots(conn, rows)


def test_score_from_db_computes_clv_and_brier(tmp_path):
    db = str(tmp_path / "snap.db")
    _seed(db)
    result = score_from_db(db_path=db, min_edge=0.03)

    assert result.n == 1  # only the market with an edge counts
    # Bought YES at mid 0.55, closed at 0.70 -> +0.15 CLV.
    assert round(result.mean_clv, 2) == 0.15
    # Model said 0.75, outcome YES(1): Brier = 0.0625.
    assert round(result.brier, 4) == 0.0625


def test_grouped_breakdown_splits_by_sport_and_type(tmp_path):
    db = str(tmp_path / "snap.db")
    _seed(db)
    groups = dict(grouped_score_from_db(db_path=db, min_edge=0.03))
    # The one edge market is soccer, default market type 'winner'.
    assert groups["soccer"].n == 1
    assert groups["soccer/winner"].n == 1
    assert round(groups["soccer/winner"].mean_clv, 2) == 0.15


def test_realized_return_fills_at_ask_net_of_fee(tmp_path):
    db = str(tmp_path / "snap.db")
    _seed(db)
    rr = realized_from_db(db_path=db, min_edge=0.03, fee=0.01)
    assert rr["n"] == 1 and rr["win_rate"] == 1.0
    # Bought YES at ask 0.56 (mid 0.55), won -> 1/0.56-1; mid fill flatters it.
    assert round(rr["roi_ask"], 4) == round(1 / 0.56 - 1, 4)
    assert rr["roi_mid"] > rr["roi_ask"]            # spread costs
    assert round(rr["roi_net"], 4) == round(rr["roi_ask"] - 0.01, 4)
    assert round(rr["avg_spread"], 4) == 0.01


def test_bucket_gate_requires_clv_above_cost():
    from kalshi_optimizer.backtest.backtester import BacktestResult
    # Like mlb/prop: big sample, calibrated, but CLV below the ~1% fee -> not armed.
    flat = BacktestResult(n=1028, brier=0.184, log_loss=0.5, mean_clv=0.0008)
    assert not flat.passes_bucket_gate(min_clv=0.01)
    # Like mlb/spread: clears samples, calibration, and beats cost -> armed.
    edge = BacktestResult(n=136, brier=0.130, log_loss=0.5, mean_clv=0.0319)
    assert edge.passes_bucket_gate(min_clv=0.01)
    # Too few samples never arms, however large the CLV.
    tiny = BacktestResult(n=43, brier=0.20, log_loss=0.5, mean_clv=0.12)
    assert not tiny.passes_bucket_gate(min_clv=0.01)


def test_empty_db_is_zero(tmp_path):
    db = str(tmp_path / "empty.db")
    storage.connect(db)
    assert score_from_db(db_path=db).n == 0
