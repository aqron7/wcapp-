"""Tests for snapshot storage and backtester DB scoring (phase 3)."""

from kalshi_optimizer import storage
from kalshi_optimizer.backtest.backtester import score_from_db


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


def test_empty_db_is_zero(tmp_path):
    db = str(tmp_path / "empty.db")
    storage.connect(db)
    assert score_from_db(db_path=db).n == 0
