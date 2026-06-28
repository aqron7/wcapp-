"""Bet ledger: record bets, settle them from snapshot results, summarize P&L.

Payout model (Kalshi binary contracts): staking $S on a contract bought at cost
``c`` per $1 buys S/c contracts, each paying $1 on a win. So a winning bet pays
``S/c`` (profit ``S/c - S``); a loser pays 0 (profit ``-S``). A parlay's cost is
the product of its legs' costs and it pays only if every leg wins.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import prod

from . import storage
from .engine.parlay import combine


def record_bet(conn, legs: list[dict], stake: float, sport: str = "") -> int:
    """Persist a new pending bet (single if one leg, else parlay)."""
    c = combine(legs)
    kind = "single" if len(legs) == 1 else "parlay"
    desc = " + ".join(f"{leg.get('side', '').upper()} {leg.get('label', leg.get('outcome'))}"
                      for leg in legs)
    bet = {
        "placed_ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind, "sport": sport, "description": desc,
        "legs_json": json.dumps(legs), "stake": float(stake),
        "entry_price": c["price"], "fair_prob": c["fair"], "edge": c["edge"],
        "status": "pending", "payout": None, "pnl": None,
        "closing_price": None, "result_ts": None,
    }
    return storage.insert_bet(conn, bet)


def _leg_won(leg: dict, result: str) -> bool:
    return (leg["side"] == "yes" and result == "yes") or (leg["side"] == "no" and result == "no")


def settle_pending(conn) -> int:
    """Settle pending bets whose every leg has a result. Returns count settled."""
    settled = 0
    for bet in storage.list_bets(conn):
        if bet["status"] != "pending":
            continue
        legs = json.loads(bet["legs_json"])
        results = [storage.result_for_market(conn, leg["market_id"]) for leg in legs]
        if any(r is None for r in results):
            continue  # not all legs settled yet

        won = all(_leg_won(leg, r) for leg, r in zip(legs, results))
        cost = bet["entry_price"] or prod(float(leg["price"]) for leg in legs)
        payout = (bet["stake"] / cost) if (won and cost) else 0.0
        pnl = payout - bet["stake"]

        # Closing line of the basket (for CLV), in bought-side price terms.
        closes = []
        for leg in legs:
            mid = storage.closing_mid(conn, leg["market_id"])
            if mid is None:
                closes = None
                break
            closes.append(mid if leg["side"] == "yes" else 1.0 - mid)
        closing_price = round(prod(closes), 4) if closes else None

        storage.update_bet(
            conn, bet["id"],
            status="won" if won else "lost",
            payout=round(payout, 2), pnl=round(pnl, 2),
            closing_price=closing_price,
            result_ts=datetime.now(timezone.utc).isoformat(),
        )
        settled += 1
    return settled


def live_positions(conn, price_map: dict[str, float]) -> list[dict]:
    """Pending bets marked to market. ``price_map`` is market_id -> live YES mid."""
    out = []
    for b in storage.list_bets(conn):
        if b["status"] != "pending":
            continue
        legs = json.loads(b["legs_json"])
        cost, ok, cur_legs = 1.0, True, []
        for leg in legs:
            mid = price_map.get(leg["market_id"])
            cur = None if mid is None else (mid if leg["side"] == "yes" else 1.0 - mid)
            if cur is None:
                ok = False
            else:
                cost *= cur
            cur_legs.append({"label": leg.get("label"), "side": leg["side"],
                             "entry": leg.get("price"),
                             "current": None if cur is None else round(cur, 2)})
        current = round(cost, 4) if ok else None
        entry = b["entry_price"]
        unreal = round(b["stake"] * (current / entry - 1), 2) if (ok and entry) else None
        out.append({"id": b["id"], "description": b["description"], "kind": b["kind"],
                    "stake": b["stake"], "entry": entry, "current": current,
                    "unrealized": unreal, "legs": cur_legs})
    return out


def calendar(conn) -> list[dict]:
    """Bets grouped by placed date: count, staked, settled P&L, pending count."""
    days: dict[str, dict] = {}
    for b in storage.list_bets(conn):
        date = (b["placed_ts"] or "")[:10]
        if not date:
            continue
        d = days.setdefault(date, {"date": date, "count": 0, "staked": 0.0,
                                   "pnl": 0.0, "pending": 0})
        d["count"] += 1
        d["staked"] += b["stake"]
        if b["status"] == "pending":
            d["pending"] += 1
        else:
            d["pnl"] += b["pnl"] or 0
    for d in days.values():
        d["staked"] = round(d["staked"], 2)
        d["pnl"] = round(d["pnl"], 2)
    return sorted(days.values(), key=lambda x: x["date"])


def summary(conn) -> dict:
    """Aggregate ledger stats + a cumulative P&L curve."""
    bets = storage.list_bets(conn)
    settled = [b for b in bets if b["status"] in ("won", "lost")]
    staked = sum(b["stake"] for b in settled)
    pnl = sum(b["pnl"] or 0 for b in settled)
    wins = sum(1 for b in settled if b["status"] == "won")
    # CLV in bought-side cost terms: we beat the close if we paid LESS than it
    # closed at, i.e. closing_price > entry_price -> positive value.
    clvs = [b["closing_price"] - b["entry_price"]
            for b in settled
            if b["entry_price"] is not None and b["closing_price"] is not None]

    curve, running = [], 0.0
    for b in sorted(settled, key=lambda x: x["result_ts"] or ""):
        running += b["pnl"] or 0
        curve.append(round(running, 2))

    return {
        "n_total": len(bets),
        "n_pending": sum(1 for b in bets if b["status"] == "pending"),
        "n_settled": len(settled),
        "staked": round(staked, 2),
        "pnl": round(pnl, 2),
        "roi": round(pnl / staked, 4) if staked else None,
        "win_rate": round(wins / len(settled), 4) if settled else None,
        "mean_clv": round(sum(clvs) / len(clvs), 4) if clvs else None,
        "curve": curve,
    }
