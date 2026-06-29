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


def test_parse_plays_types_and_validation():
    valid = {"E-A", "E-B", "E-C"}
    personas = {"Quant", "Contrarian"}
    text = (
        '[{"analyst":"Quant","play_type":"single","legs":[{"market_id":"E-A","side":"yes","stake":1}],'
        '"confidence":0.6,"rationale":"r1"},'
        '{"analyst":"Ghost","play_type":"single","legs":[{"market_id":"E-B","side":"yes"}]},'  # bad analyst
        '{"analyst":"Contrarian","play_type":"single","legs":[{"market_id":"E-Z","side":"no"}]},'  # bad id -> no legs
        '{"analyst":"Contrarian","play_type":"hedge","legs":['
        '{"market_id":"E-B","side":"yes","stake":0.7},{"market_id":"E-C","side":"no","stake":0.3}],'
        '"confidence":0.8,"rationale":"main + insurance"}]')
    plays = analysts.parse_plays(text, valid, personas)
    assert [(p["analyst"], p["play_type"], len(p["legs"])) for p in plays] == [
        ("Quant", "single", 1), ("Contrarian", "hedge", 2)]
    assert plays[1]["legs"][0]["stake"] == 0.7
    assert analysts.parse_plays("no json", valid, personas) == []


def test_parse_plays_multi_leg_defaults_to_parlay():
    plays = analysts.parse_plays(
        '[{"analyst":"Quant","legs":[{"market_id":"E-A","side":"yes"},'
        '{"market_id":"E-B","side":"yes"}]}]', {"E-A", "E-B"}, {"Quant"})
    assert plays[0]["play_type"] == "parlay"   # 2 legs, no type given


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


def _pick(**kw):
    base = {"ts": "t", "date": "2026-06-28", "sport": "soccer", "game_key": "E",
            "game_label": "A v B", "confidence": 0.7, "rationale": "x",
            "play_type": "single", "role": "main", "stake": 1.0, "leg_label": "l"}
    base.update(kw)
    return base


def test_grade_single_play(tmp_path):
    conn = storage.connect(str(tmp_path / "a.db"))
    storage.insert_snapshots(conn, [
        ("t", "soccer", "E", "E-A", "A", None, None, None, "settled", "yes"),
    ])
    storage.insert_analyst_pick(conn, _pick(analyst="Quant", market_id="E-A", side="yes",
                                            price=0.5, play_id="p1"))
    lb = analysts.grade_and_leaderboard(conn)["leaderboard"]
    assert lb[0]["analyst"] == "Quant"
    assert lb[0]["wins"] == 1 and lb[0]["pnl"] == 10.0   # $10 at 0.5 -> +$10


def test_grade_parlay_all_must_hit(tmp_path):
    conn = storage.connect(str(tmp_path / "p.db"))
    storage.insert_snapshots(conn, [
        ("t", "soccer", "E", "E-A", "A", None, None, None, "settled", "yes"),
        ("t", "soccer", "E", "E-B", "B", None, None, None, "settled", "no"),
    ])
    # Two-leg parlay; one leg loses -> whole play loses the single $10 stake.
    storage.insert_analyst_pick(conn, _pick(analyst="Quant", market_id="E-A", side="yes",
                                            price=0.5, play_id="par", play_type="parlay"))
    storage.insert_analyst_pick(conn, _pick(analyst="Quant", market_id="E-B", side="yes",
                                            price=0.5, play_id="par", play_type="parlay", role="leg"))
    lb = analysts.grade_and_leaderboard(conn)["leaderboard"]
    assert lb[0]["picks"] == 1 and lb[0]["settled"] == 1   # one play, two legs
    assert lb[0]["wins"] == 0 and lb[0]["pnl"] == -10.0


def test_grade_hedge_insures_the_loss(tmp_path):
    conn = storage.connect(str(tmp_path / "h.db"))
    storage.insert_snapshots(conn, [
        ("t", "soccer", "E", "E-A", "A", None, None, None, "settled", "no"),   # main loses
        ("t", "soccer", "E", "E-B", "B", None, None, None, "settled", "yes"),  # insurance hits
    ])
    storage.insert_analyst_pick(conn, _pick(analyst="Quant", market_id="E-A", side="yes",
                                            price=0.5, stake=0.6, play_id="h", play_type="hedge"))
    storage.insert_analyst_pick(conn, _pick(analyst="Quant", market_id="E-B", side="yes",
                                            price=0.4, stake=0.4, play_id="h", play_type="hedge", role="leg"))
    lb = analysts.grade_and_leaderboard(conn)["leaderboard"]
    # main: -$6 ; insurance: $4/0.4-$4 = +$6 ; net 0 -> cushioned, not a win
    assert lb[0]["settled"] == 1 and lb[0]["pnl"] == 0.0
