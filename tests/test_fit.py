"""Tests for fitting ratings from settled games."""

from kalshi_optimizer.models.fit import fit_ratings, games_from_settled


def _settled(event, code, label, result, ts):
    return {"ticker": f"{event}-{code}", "event_ticker": event, "yes_sub_title": label,
            "result": result, "close_time": ts}


def test_games_from_settled_reads_winner():
    raw = [
        _settled("KXWCGAME-D1ESPAUT", "ESP", "Reg Time: Spain", "yes", "t1"),
        _settled("KXWCGAME-D1ESPAUT", "AUT", "Reg Time: Austria", "no", "t1"),
        _settled("KXWCGAME-D1ESPAUT", "TIE", "Reg Time: Tie", "no", "t1"),
    ]
    games = games_from_settled(raw, "soccer")
    assert games == [("spain", "austria", 1.0)]   # Spain won, keyed by name


def test_draw_is_half():
    raw = [
        _settled("E", "ESP", "Reg Time: Spain", "no", "t1"),
        _settled("E", "AUT", "Reg Time: Austria", "no", "t1"),
        _settled("E", "TIE", "Reg Time: Tie", "yes", "t1"),
    ]
    assert games_from_settled(raw, "soccer")[0][2] == 0.5


def test_fit_raises_winner_rating():
    init = {"a": 1500, "b": 1500}
    out = fit_ratings([("a", "b", 1.0)] * 5, init, k=20)
    assert out["a"] > 1500 > out["b"]


def test_fit_orders_chronologically():
    raw = [
        _settled("G2", "BBB", "Reg Time: Bbb", "yes", "2026-02-01"),
        _settled("G2", "AAA", "Reg Time: Aaa", "no", "2026-02-01"),
        _settled("G1", "AAA", "Reg Time: Aaa", "yes", "2026-01-01"),
        _settled("G1", "CCC", "Reg Time: Ccc", "no", "2026-01-01"),
    ]
    games = games_from_settled(raw, "soccer")
    assert games[0][:2] == ("aaa", "ccc")   # Jan game first
