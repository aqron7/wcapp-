"""Tests for the crypto catalyst agent (pure parse/dedup/score logic)."""

from datetime import datetime, timedelta, timezone

from kalshi_optimizer import storage
from kalshi_optimizer.crypto import catalysts, score


def test_parse_signals_validates_and_normalizes():
    valid = {"a1", "a2"}
    text = ('junk [{"article_id":"a1","asset":"btc","direction":"UP","magnitude":"huge",'
            '"horizon_hours":48,"confidence":1.7,"catalyst_type":"unlock","thesis":"t"},'
            '{"article_id":"zz","asset":"ETH","direction":"down"},'          # bad id -> dropped
            '{"article_id":"a2","asset":"","direction":"up"}] tail')          # no asset -> dropped
    sigs = catalysts.parse_signals(text, valid)
    assert len(sigs) == 1
    s = sigs[0]
    assert s["asset"] == "BTC" and s["direction"] == "up"
    assert s["magnitude"] == "small"      # 'huge' normalized to default
    assert s["confidence"] == 1.0         # clamped
    assert catalysts.parse_signals("no json", valid) == []


def test_dedup_keeps_highest_confidence():
    sigs = [
        {"asset": "BTC", "catalyst_type": "unlock", "confidence": 0.4},
        {"asset": "BTC", "catalyst_type": "unlock", "confidence": 0.8},
        {"asset": "ETH", "catalyst_type": "listing", "confidence": 0.5},
    ]
    out = catalysts.dedup(sigs)
    assert len(out) == 2
    btc = next(s for s in out if s["asset"] == "BTC")
    assert btc["confidence"] == 0.8


def test_signed_move_and_correctness():
    # up call, +3% move -> correct, signed move +0.03
    assert score._is_correct("up", 0.03)
    assert score._signed_move("up", 0.03) == 0.03
    # down call, price rose -> wrong, signed move negative
    assert not score._is_correct("down", 0.03)
    assert score._signed_move("down", 0.03) == -0.03
    # tiny move within the flat band -> only 'neutral' is right
    assert score._is_correct("neutral", 0.005)
    assert not score._is_correct("up", 0.005)


def test_score_matured_grades_after_horizon(tmp_path, monkeypatch):
    conn = storage.connect(str(tmp_path / "c.db"))
    flagged = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    storage.insert_crypto_signal(conn, {
        "ts": flagged, "asset": "BTC", "direction": "up", "magnitude": "medium",
        "horizon_hours": 24, "confidence": 0.7, "catalyst_type": "listing",
        "thesis": "t", "title": "x", "url": "", "source": "s", "published": 0,
        "entry_price": 100.0, "status": "pending"})
    # BTC rose 5% by scoring time.
    monkeypatch.setattr(score, "spot_price", lambda *a, **k: 105.0)
    assert score.score_matured(conn) == 1
    lb = score.leaderboard(conn)
    row = next(r for r in lb["leaderboard"] if r["group"] == "listing")
    assert row["n"] == 1 and row["hit_rate"] == 1.0
    assert row["mean_signed_move"] == 0.05


def test_score_skips_immature_signals(tmp_path):
    conn = storage.connect(str(tmp_path / "c2.db"))
    fresh = datetime.now(timezone.utc).isoformat()
    storage.insert_crypto_signal(conn, {
        "ts": fresh, "asset": "ETH", "direction": "down", "magnitude": "small",
        "horizon_hours": 24, "confidence": 0.6, "catalyst_type": "macro",
        "thesis": "t", "title": "x", "url": "", "source": "s", "published": 0,
        "entry_price": 50.0, "status": "pending"})
    assert score.score_matured(conn) == 0          # horizon not elapsed
    assert score.leaderboard(conn)["pending"] == 1
