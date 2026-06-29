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


def build_panel_prompt(personas: dict[str, str], packets: list[dict]) -> str:
    """One prompt covering every persona AND every game (1 LLM call per run).

    Each analyst returns a *play* per game they like: a single bet, a parlay
    (several legs that all must hit, for a bigger payout), or a hedged single (a
    main bet plus a smaller insurance leg so a loss is cushioned).
    """
    blocks = []
    for pk in packets:
        lines = "\n".join(f"  - {m['market_id']} | {m['desc']} | price {m['price']} | model_fair {m['fair']}"
                          for m in pk["markets"][:10])
        blocks.append(f"GAME {pk['game_key']}: {pk['label']} ({pk['sport']})\n{lines}")
    roster = "\n".join(f"- {name}: {desc}" for name, desc in personas.items())
    return (
        "You are a panel of distinct sports-betting analysts. Each has a perspective:\n"
        f"{roster}\n\n"
        "Today's games and the markets available for each "
        "(market_id | description | price | model_fair). "
        "price is the cost of YES; betting \"no\" costs (1 - price):\n\n"
        + "\n\n".join(blocks) + "\n\n"
        "For EACH analyst, build their best plays for this slate. For any game an "
        "analyst likes, give ONE play with a play_type:\n"
        '- "parlay": 2+ legs (markets, even across games) that all must win, for a '
        "bigger payout — use only with real conviction on each leg.\n"
        '- "single": one leg.\n'
        '- "hedge": a main leg PLUS a smaller insurance leg on a different or '
        "opposing outcome, so if the main loses the insurance cushions it and you "
        "don't lose much, while a main win still nets a profit.\n"
        "Skip games an analyst has no edge on. Each leg is "
        '{"market_id","side","stake"} where side is "yes" or "no" and stake is a '
        "relative weight 0-1 (parlay: use 1 for every leg; hedge: main larger, e.g. "
        "0.7, insurance smaller, e.g. 0.3).\n"
        "In the rationale, NAME the actual bet in plain words (e.g. \"take the over "
        "2.5 goals and Spain to win\" or \"back the Yankees but insure with the "
        "under\"), don't just cite ids.\n"
        "Reply ONLY with a JSON array of plays: "
        '[{"analyst":"<exact name>","play_type":"single|parlay|hedge",'
        '"legs":[{"market_id":"<id>","side":"yes","stake":1}],'
        '"confidence":0.0-1.0,"rationale":"one or two sentences"}]. '
        "Use analyst names and market_id values exactly as listed."
    )


def parse_plays(text: str, valid_ids: set[str], valid_personas: set[str]) -> list[dict]:
    """Parse the panel response into validated plays (analyst + typed legs)."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except ValueError:
        return []
    out: list[dict] = []
    for d in arr if isinstance(arr, list) else []:
        if not isinstance(d, dict):
            continue
        analyst = str(d.get("analyst", ""))
        if analyst not in valid_personas:
            continue
        legs = []
        for leg in d.get("legs", []) if isinstance(d.get("legs"), list) else []:
            if not isinstance(leg, dict):
                continue
            mid = leg.get("market_id")
            if not mid or mid not in valid_ids:
                continue
            side = "no" if str(leg.get("side", "yes")).lower() == "no" else "yes"
            try:
                stake = float(leg.get("stake", 1.0))
            except (TypeError, ValueError):
                stake = 1.0
            stake = max(0.0, min(1.0, stake)) or 1.0
            legs.append({"market_id": mid, "side": side, "stake": stake})
        if not legs:
            continue
        ptype = str(d.get("play_type", "single")).lower()
        if ptype not in ("single", "parlay", "hedge"):
            ptype = "single"
        if len(legs) == 1:
            ptype = "single"          # one leg can't be a parlay/hedge
        elif ptype == "single":
            ptype = "parlay"          # multiple legs with no type -> treat as parlay
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.6))))
        except (TypeError, ValueError):
            conf = 0.6
        out.append({"analyst": analyst, "play_type": ptype, "legs": legs,
                    "confidence": conf, "rationale": str(d.get("rationale", ""))[:300]})
    return out


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


STAKE_UNIT = 10.0  # hypothetical stake per play, for grading


def _leg_won(side: str, result: str | None) -> bool:
    return (side == "yes" and result == "yes") or (side == "no" and result == "no")


def _grade_play(legs: list[dict], results: dict) -> tuple[bool, float]:
    """Return (won, pnl) for one settled play. legs are pick rows; results maps id->'yes'/'no'.

    parlay: one STAKE_UNIT, all legs must win, payout multiplies the leg odds.
    single/hedge: STAKE_UNIT split across legs by stake weight; each leg settles
    independently and the play nets the sum (insurance cushions a main-leg loss).
    """
    ptype = legs[0]["play_type"] or "single"
    if ptype == "parlay":
        won_all = all(_leg_won(l["side"], results[l["id"]]) for l in legs)
        if not won_all:
            return False, -STAKE_UNIT
        mult = 1.0
        for l in legs:
            mult *= 1.0 / (l["price"] or 1.0)
        return True, STAKE_UNIT * mult - STAKE_UNIT
    total_w = sum(l["stake"] or 1.0 for l in legs) or 1.0
    pnl = 0.0
    for l in legs:
        stake = STAKE_UNIT * (l["stake"] or 1.0) / total_w
        if _leg_won(l["side"], results[l["id"]]) and l["price"]:
            pnl += stake / l["price"] - stake
        else:
            pnl -= stake
    return pnl > 0, pnl


def _play_id(p: dict) -> str:
    return p.get("play_id") or f"legacy-{p['id']}"


def grade_and_leaderboard(conn) -> dict:
    """Settle gradeable analyst plays and return a per-analyst leaderboard."""
    from . import storage

    plays: dict[str, list[dict]] = {}
    for p in storage.list_analyst_picks(conn):
        plays.setdefault(_play_id(p), []).append(p)

    for legs in plays.values():
        if any(l["status"] != "pending" for l in legs):
            continue  # already graded
        results = {l["id"]: storage.result_for_market(conn, l["market_id"]) for l in legs}
        if any(r is None for r in results.values()):
            continue  # not all legs settled yet
        won, pnl = _grade_play(legs, results)
        for i, l in enumerate(legs):
            storage.update_analyst_pick(conn, l["id"], status="won" if won else "lost",
                                        result=results[l["id"]],
                                        pnl=round(pnl, 2) if i == 0 else 0.0)

    # Re-read so the board reflects the grades just written, not stale rows.
    plays = {}
    for p in storage.list_analyst_picks(conn):
        plays.setdefault(_play_id(p), []).append(p)
    board: dict[str, dict] = {}
    for legs in plays.values():
        analyst = legs[0]["analyst"]
        b = board.setdefault(analyst, {"analyst": analyst, "picks": 0, "settled": 0,
                                       "wins": 0, "pnl": 0.0, "pending": 0})
        b["picks"] += 1
        if any(l["status"] == "pending" for l in legs):
            b["pending"] += 1
            continue
        b["settled"] += 1
        pnl = sum(l["pnl"] or 0 for l in legs)
        b["pnl"] += pnl
        b["wins"] += 1 if pnl > 0 else 0
    for b in board.values():
        b["pnl"] = round(b["pnl"], 2)
        b["roi"] = round(b["pnl"] / (b["settled"] * STAKE_UNIT), 3) if b["settled"] else None
        b["win_rate"] = round(b["wins"] / b["settled"], 3) if b["settled"] else None
    leaderboard = sorted(board.values(), key=lambda x: (x["pnl"], x["settled"]), reverse=True)
    return {"leaderboard": leaderboard}


def generate_takes(quotes: list[MarketQuote], sport: str, secrets,
                   personas: list[str] | None = None, max_games: int = 8) -> list[dict]:
    """One LLM call for the whole panel; return per-leg rows grouped into plays."""
    import uuid

    packets = list(build_packets(quotes, sport).values())[:max_games]
    if not packets:
        return []
    chosen = personas or list(PERSONAS)
    roster = {name: PERSONAS[name] for name in chosen if name in PERSONAS}
    # market_id -> (packet, yes_mid, desc) so the response maps back to games.
    meta = {m["market_id"]: (pk, m["price"], m["desc"])
            for pk in packets for m in pk["markets"]}
    valid = set(meta)
    ts = datetime.now(timezone.utc).isoformat()
    date = ts[:10]

    # One LLM call for the whole panel — cheapest on the free daily quota.
    try:
        text = llm_complete(build_panel_prompt(roster, packets),
                            system="You are a panel of sharp sports betting analysts.",
                            secrets=secrets)
    except Exception:  # noqa: BLE001
        return []

    out: list[dict] = []
    for play in parse_plays(text, valid, set(roster)):
        pid = f"{play['analyst']}-{uuid.uuid4().hex[:8]}"
        labels = {meta[l["market_id"]][0]["label"] for l in play["legs"]}
        game_label = ("Parlay: " + "; ".join(sorted(labels)))[:90] if len(labels) > 1 \
            else next(iter(labels))
        game_key = meta[play["legs"][0]["market_id"]][0]["game_key"]
        for i, leg in enumerate(play["legs"]):
            pk, yes_mid, desc = meta[leg["market_id"]]
            entry = yes_mid if leg["side"] == "yes" else round(1 - yes_mid, 2)
            out.append({"ts": ts, "date": date, "sport": sport, "analyst": play["analyst"],
                        "game_key": game_key, "game_label": game_label,
                        "market_id": leg["market_id"], "side": leg["side"], "price": entry,
                        "confidence": play["confidence"], "rationale": play["rationale"],
                        "play_id": pid, "play_type": play["play_type"],
                        "role": "main" if i == 0 else "leg", "stake": leg["stake"],
                        "leg_label": desc})
    return out
