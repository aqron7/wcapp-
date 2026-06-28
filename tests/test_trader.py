"""Tests for guarded order execution."""

from kalshi_optimizer.config import Config, ExecutionConfig
from kalshi_optimizer.execution.trader import Trader

LEG = {"market_id": "KXWCGAME-E-GER", "side": "no", "price": 0.07}


class FakeKalshi:
    def __init__(self):
        self.calls = []

    def place_order(self, ticker, side, count, price_cents, client_order_id):
        self.calls.append((ticker, side, count, price_cents))
        return {"order_id": "abc", "status": "resting"}


def _cfg(mode, kill):
    c = Config()
    c.execution = ExecutionConfig(mode=mode, kill_switch=kill)
    return c


def test_kill_switch_blocks_and_simulates():
    fk = FakeKalshi()
    r = Trader(_cfg("live", True), fk).execute_single(LEG, 21.0)
    assert r["status"] == "simulated" and "kill_switch" in r["reason"]
    assert not fk.calls  # nothing sent


def test_dry_run_simulates_with_order_math():
    fk = FakeKalshi()
    r = Trader(_cfg("dry_run", False), fk).execute_single(LEG, 21.0)
    assert r["status"] == "simulated"
    # $21 at 0.07 -> 300 contracts at 7 cents
    assert r["would_place"]["count"] == 300
    assert r["would_place"]["price_cents"] == 7
    assert not fk.calls


def test_live_sends_real_order():
    fk = FakeKalshi()
    r = Trader(_cfg("live", False), fk).execute_single(LEG, 21.0)
    assert r["status"] == "placed"
    assert fk.calls == [("KXWCGAME-E-GER", "no", 300, 7)]


def test_stake_too_small_errors():
    r = Trader(_cfg("live", False), FakeKalshi()).execute_single(LEG, 0.02)
    assert r["status"] == "error"
