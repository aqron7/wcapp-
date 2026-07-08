"""Scoring gate for crypto catalyst signals.

The crypto analog of the sports CLV gate: each signal is logged with the asset's
entry price; once its horizon elapses we compare to the price then and score
whether the predicted direction was right and how far it moved. The agent isn't
trusted for trading until its *signed move in the predicted direction* is
positive across a real sample — direction accuracy alone can be a coin flip.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import storage
from .news import spot_price

# Moves smaller than this are treated as flat (noise), so "neutral" can be right
# and tiny wiggles don't count as directional hits.
FLAT_BAND = 0.01   # 1%


def _matured(ts_iso: str, horizon_hours: int, now: datetime) -> bool:
    try:
        flagged = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    except ValueError:
        return False
    return now >= flagged + timedelta(hours=horizon_hours or 0)


def _signed_move(direction: str, move_pct: float) -> float:
    """Return the move in the direction we predicted (up:+move, down:-move)."""
    if direction == "up":
        return move_pct
    if direction == "down":
        return -move_pct
    return -abs(move_pct)   # 'neutral' profits only if it stayed flat


def _is_correct(direction: str, move_pct: float, band: float = FLAT_BAND) -> bool:
    if direction == "up":
        return move_pct > band
    if direction == "down":
        return move_pct < -band
    return abs(move_pct) <= band


def score_matured(conn, api_key: str = "", now: datetime | None = None) -> int:
    """Score pending signals whose horizon has elapsed. Returns count scored."""
    now = now or datetime.now(timezone.utc)
    scored = 0
    for s in storage.list_crypto_signals(conn):
        if s["status"] != "pending" or not _matured(s["ts"], s["horizon_hours"] or 0, now):
            continue
        if not s["entry_price"]:
            storage.update_crypto_signal(conn, s["id"], status="skipped")
            continue
        after = spot_price(s["asset"], api_key)
        if after is None:
            continue
        move = (after - s["entry_price"]) / s["entry_price"]
        storage.update_crypto_signal(
            conn, s["id"], status="scored", price_after=after,
            move_pct=round(move, 5), correct=1 if _is_correct(s["direction"], move) else 0,
            scored_ts=now.isoformat())
        scored += 1
    return scored


def leaderboard(conn) -> dict:
    """Per catalyst_type accuracy + mean signed move (the edge signal), plus overall."""
    rows = [s for s in storage.list_crypto_signals(conn) if s["status"] == "scored"]
    groups: dict[str, list[dict]] = {}
    for s in rows:
        groups.setdefault(s["catalyst_type"] or "other", []).append(s)
        groups.setdefault("ALL", []).append(s)

    def summarize(name, items):
        n = len(items)
        hits = sum(1 for s in items if s["correct"])
        edge = sum(_signed_move(s["direction"], s["move_pct"] or 0.0) for s in items) / n
        return {"group": name, "n": n, "hit_rate": round(hits / n, 3),
                "mean_signed_move": round(edge, 4)}

    board = [summarize(k, v) for k, v in groups.items() if v]
    board.sort(key=lambda x: (x["group"] != "ALL", -x["n"]))
    return {"leaderboard": board,
            "pending": sum(1 for s in storage.list_crypto_signals(conn) if s["status"] == "pending")}
