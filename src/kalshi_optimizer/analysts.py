"""AI analyst "perspectives" — opinionated bet takes per game.

Each persona is an LLM prompt that reads a game's data packet (candidate markets
with prices + model fair, plus context) and either picks one bet or passes. These
are *ideas from a point of view*, separate from the CLV-validated model edges, and
each analyst's picks are logged and graded so you can see whose perspective wins.

Reasoning is over the data we provide + the model's general sports knowledge — no
live injury/news/web access.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone

from .engine.value import predictions_with_context
from .providers import llm_complete
from .types import MarketQuote

PERSONAS: dict[str, str] = {
    "Quant": "a disciplined quantitative bettor who trusts the model fair value and "
             "ratings, and only bets when the price clearly diverges from fair value.",
    "Contrarian": "a contrarian who fades public favorites and inflated chalk, hunting "
                  "value on underdogs and unders.",
    "Trend-rider": "a momentum bettor who backs recent form, hot streaks, and in-form "
                   "teams and players.",
    "Situational": "a situational handicapper who weighs weather, altitude, rest, travel, "
                   "and park/venue factors heavily.",
    "Handicapper": "a seasoned generalist handicapper who blends team quality, matchups, "
                   "playing styles, motivation, and broad sports knowledge into one best play.",
}


def _game_key(event_ticker: str) -> str:
    return event_ticker.split("-", 1)[1] if "-" in event_ticker else event_ticker


def build_packets(quotes: list[MarketQuote], sport: str, min_price: float = 0.05) -> dict[str, dict]:
    """Per game: a label + candidate markets (id, description, price, model fair)."""
    fair = {(p.event_key, p.outcome): p.fair_prob for p in predictions_with_context(sport, quotes)}
    games: dict[str, dict] = {}
    for q in quotes:
        if q.platform != "kalshi" or not q.event_key or not q.outcome:
            continue
        if not q.yes_bid or not q.yes_ask or not (min_price <= (q.yes_mid or 0) <= 1 - min_price):
            continue
        gk = _game_key(q.event_key)
        g = games.setdefault(gk, {"game_key": gk, "sport": sport, "label": None, "markets": []})
        if q.market_type == "winner" and not g["label"]:
            g["label"] = q.title.split(" Winner")[0]
        g["markets"].append({
            "market_id": q.market_id,
            "desc": f"{q.title} [{q.outcome_label or q.outcome}]",
            "price": round(q.yes_mid, 2),
            "fair": round(fair.get((q.event_key, q.outcome), 0.0), 2),
        })
    for g in games.values():
        if not g["label"] and g["markets"]:
            g["label"] = g["markets"][0]["desc"]
    return {gk: g for gk, g in games.items() if g["markets"]}


def build_prompt(persona_desc: str, packet: dict) -> str:
    lines = "\n".join(f"- {m['market_id']} | {m['desc']} | price {m['price']} | model_fair {m['fair']}"
                      for m in packet["markets"][:12])
    return (
        f"Game: {packet['label']} ({packet['sport']})\n"
        f"Bets available (id | description | price | model_fair):\n{lines}\n\n"
        f"You are {persona_desc}\n"
        "Pick AT MOST ONE bet to place from your perspective, or none. 'side' is "
        "\"yes\" to back the listed outcome, \"no\" to bet against it.\n"
        'Reply ONLY with JSON: {"market_id":"<id or null>","side":"yes",'
        '"confidence":0.0-1.0,"rationale":"one sentence"}'
    )


def parse_pick(text: str, valid_ids: set[str]) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    mid = d.get("market_id")
    if not mid or mid not in valid_ids:
        return None
    side = "no" if str(d.get("side", "yes")).lower() == "no" else "yes"
    try:
        conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    return {"market_id": mid, "side": side, "confidence": conf,
            "rationale": str(d.get("rationale", ""))[:240]}


STAKE_UNIT = 10.0  # hypothetical flat stake per analyst pick, for grading


def grade_and_leaderboard(conn) -> dict:
    """Settle gradeable analyst picks and return per-analyst records + open picks."""
    from . import storage

    for p in storage.list_analyst_picks(conn):
        if p["status"] != "pending":
            continue
        result = storage.result_for_market(conn, p["market_id"])
        if result is None:
            continue
        outcome = p["market_id"].rsplit("-", 1)[-1]  # unused but explicit
        won = (p["side"] == "yes" and result == "yes") or (p["side"] == "no" and result == "no")
        price = p["price"] or 0
        pnl = (STAKE_UNIT / price - STAKE_UNIT) if (won and price) else -STAKE_UNIT
        storage.update_analyst_pick(conn, p["id"], status="won" if won else "lost",
                                    result=result, pnl=round(pnl, 2))

    board: dict[str, dict] = {}
    for p in storage.list_analyst_picks(conn):
        b = board.setdefault(p["analyst"], {"analyst": p["analyst"], "picks": 0,
                                            "settled": 0, "wins": 0, "pnl": 0.0, "pending": 0})
        b["picks"] += 1
        if p["status"] == "pending":
            b["pending"] += 1
        else:
            b["settled"] += 1
            b["wins"] += 1 if p["status"] == "won" else 0
            b["pnl"] += p["pnl"] or 0
    for b in board.values():
        b["pnl"] = round(b["pnl"], 2)
        b["roi"] = round(b["pnl"] / (b["settled"] * STAKE_UNIT), 3) if b["settled"] else None
        b["win_rate"] = round(b["wins"] / b["settled"], 3) if b["settled"] else None
    leaderboard = sorted(board.values(), key=lambda x: (x["pnl"], x["settled"]), reverse=True)
    return {"leaderboard": leaderboard}


def generate_takes(quotes: list[MarketQuote], sport: str, secrets,
                   personas: list[str] | None = None, max_games: int = 8) -> list[dict]:
    """Call the LLM for each persona on each game; return logged-pick dicts."""
    packets = list(build_packets(quotes, sport).values())[:max_games]
    chosen = personas or list(PERSONAS)
    ts = datetime.now(timezone.utc).isoformat()
    date = ts[:10]
    out: list[dict] = []
    for packet in packets:
        valid = {m["market_id"] for m in packet["markets"]}
        price_by_id = {m["market_id"]: m["price"] for m in packet["markets"]}
        for persona in chosen:
            try:
                text = llm_complete(build_prompt(PERSONAS[persona], packet),
                                    system="You are a sharp sports betting analyst.", secrets=secrets)
            except Exception:  # noqa: BLE001
                continue
            pick = parse_pick(text, valid)
            if not pick:
                continue
            yes_mid = price_by_id[pick["market_id"]]
            entry = yes_mid if pick["side"] == "yes" else round(1 - yes_mid, 2)
            out.append({"ts": ts, "date": date, "sport": sport, "analyst": persona,
                        "game_key": packet["game_key"], "game_label": packet["label"],
                        "market_id": pick["market_id"], "side": pick["side"],
                        "price": entry, "confidence": pick["confidence"],
                        "rationale": pick["rationale"]})
            time.sleep(0.2)  # be gentle with free-tier rate limits
    return out
