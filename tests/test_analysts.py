"""Tests for the AI analyst engine (pure parts: parse, packets, grading)."""

from kalshi_optimizer import analysts, storage
from kalshi_optimizer.types import MarketQuote


def test_parse_pick_valid_and_invalid():
    valid = {"E-A"}
    ok = analysts.parse_pick('{"market_id":"E-A","side":"no","confidence":0.7,"rationale":"x"}', valid)
    assert ok == {"market_id": "E-A", "side": "no", "confidence": 0.7, "rationale": "x"}
    # market not offered -> rejected; junk -> None; pass -> None
    assert analysts.parse_pick('{"market_id":"E-Z","side":"yes"}', valid) is None
    assert analysts.parse_pick("no json here", valid) is None
    assert analysts.parse_pick('{"market_id":null}', valid) is None


def test_build_packets_groups_and_filters():
    quotes = [
        MarketQuote("kalshi", "KXWCGAME-G-ESP", "Spain vs Austria Winner?", 0.5, 0.52,
                    sport="soccer", event_key="KXWCGAME-G", outcome="ESP",
                    outcome_label="Spain", market_type="winner"),
        MarketQuote("kalshi", "KXWCGAME-G-AUT", "Spain vs Austria Winner?", 0.46, 0.48,
                    sport="soccer", event_key="KXWCGAME-G", outcome="AUT",
                    outcome_label="Austria", market_type="winner"),
        MarketQuote("kalshi", "KXWCGAME-G-LONG", "Spain vs Austria Winner?", 0.01, 0.02,
                    sport="soccer", event_key="KXWCGAME-G", outcome="LONG",
                    outcome_label="x", market_type="winner"),  # near-decided -> filtered
    ]
    packets = analysts.build_packets(quotes, "soccer")
    g = packets["G"]
    assert g["label"] == "Spain vs Austria"
    ids = {m["market_id"] for m in g["markets"]}
    assert ids == {"KXWCGAME-G-ESP", "KXWCGAME-G-AUT"}   # longshot dropped


def test_grade_and_leaderboard(tmp_path):
    conn = storage.connect(str(tmp_path / "a.db"))
    storage.insert_snapshots(conn, [
        ("t", "soccer", "E", "E-A", "A", None, None, None, "settled", "yes"),
    ])
    storage.insert_analyst_pick(conn, {"ts": "t", "date": "2026-06-28", "sport": "soccer",
                                       "analyst": "Quant", "game_key": "E", "game_label": "A v B",
                                       "market_id": "E-A", "side": "yes", "price": 0.5,
                                       "confidence": 0.7, "rationale": "x"})
    lb = analysts.grade_and_leaderboard(conn)["leaderboard"]
    assert lb[0]["analyst"] == "Quant"
    assert lb[0]["wins"] == 1 and lb[0]["pnl"] == 10.0   # $10 at 0.5 -> +$10
