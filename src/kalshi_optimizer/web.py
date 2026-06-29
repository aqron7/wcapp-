"""FastAPI backend for the web dashboard (phase 5).

Serves the single-page frontend plus JSON endpoints that reuse the same engine
the CLI uses. Run with:  python -m kalshi_optimizer dashboard

Needs the optional dashboard dependency:  pip install -e ".[dashboard]"
"""

from __future__ import annotations

from pathlib import Path

import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import ledger, realtime, storage
from .backtest.backtester import score_from_db
from .config import Config
from .data.kalshi import KalshiClient
from .engine.parlay import combine, has_correlated_legs
from .engine.value import find_value_edges, predictions_with_context

app = FastAPI(title="Kalshi Edge")
_INDEX = Path(__file__).with_name("static") / "index.html"
LIVE = realtime.LiveBook()


def _market_type(market_id: str) -> str:
    from .data.kalshi import SERIES_TYPE
    return SERIES_TYPE.get(market_id.split("-", 1)[0], "winner")


def _game_key(market_id: str) -> str:
    ev = market_id.rsplit("-", 1)[0]
    return ev.split("-", 1)[1] if "-" in ev else ev


def _edge_dict(idea, sport: str) -> dict:
    side_fair = idea.fair_prob if idea.side.value == "yes" else 1 - idea.fair_prob
    return {
        "sport": sport, "title": idea.title, "market_id": idea.market_id,
        "event_ticker": idea.market_id.rsplit("-", 1)[0],
        "outcome": idea.market_id.rsplit("-", 1)[-1],
        "market_type": _market_type(idea.market_id),
        "game_key": _game_key(idea.market_id),
        "side": idea.side.value, "price": round(idea.price, 2),
        "fair": round(idea.fair_prob, 2), "fair_side": round(side_fair, 2),
        "edge": round(idea.edge, 4), "stake": idea.stake, "why": idea.rationale,
    }


def _all_edges(config) -> list[dict]:
    if LIVE.connected and LIVE.markets:
        return [_edge_dict(i, _sport_of(i.market_id)) for i in LIVE.edges(config)]
    kalshi = KalshiClient(config.secrets)
    edges: list[dict] = []
    for sport in ("soccer", "mlb"):
        try:
            quotes = kalshi.get_sports_markets(sport)
            edges += [_edge_dict(i, sport)
                      for i in find_value_edges(quotes, predictions_with_context(sport, quotes), config)]
        except Exception:  # noqa: BLE001
            pass
    return edges


@app.on_event("startup")
async def _start_realtime() -> None:
    asyncio.create_task(realtime.maintain(LIVE, Config.load()))


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX.read_text(encoding="utf-8")


def _sport_of(market_id: str) -> str:
    return "mlb" if "MLB" in market_id else "soccer"


def _config(min_edge: float, kelly: float, bankroll: float) -> Config:
    c = Config.load()
    c.bankroll = bankroll
    c.edge.min_edge = min_edge
    c.sizing.kelly_fraction = kelly
    return c


@app.get("/api/edges")
def api_edges(sports: str = "soccer,mlb", min_edge: float = 0.03,
              kelly: float = 0.25, bankroll: float = 1000.0) -> dict:
    config = _config(min_edge, kelly, bankroll)
    wanted = [s for s in sports.split(",") if s]

    # Prefer the live ws book when it's connected and populated.
    if LIVE.connected and LIVE.markets:
        edges = [_edge_dict(i, _sport_of(i.market_id)) for i in LIVE.edges(config)
                 if _sport_of(i.market_id) in wanted]
        edges.sort(key=lambda e: e["edge"], reverse=True)
        return {"edges": edges, "errors": [], "live": True}

    # Fallback: fetch fresh over REST.
    kalshi = KalshiClient(config.secrets)
    edges, errors = [], []
    for sport in wanted:
        try:
            quotes = kalshi.get_sports_markets(sport)
            preds = predictions_with_context(sport, quotes)
            edges += [_edge_dict(i, sport) for i in find_value_edges(quotes, preds, config)]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sport}: {exc}")
    edges.sort(key=lambda e: e["edge"], reverse=True)
    return {"edges": edges, "errors": errors, "live": False}


@app.get("/api/history")
def api_history(market_id: str) -> dict:
    """Price/model time series for one market (for the chart), from snapshots."""
    conn = storage.connect()
    rows = conn.execute(
        "SELECT ts, yes_bid, yes_ask, model_fair FROM snapshots "
        "WHERE market_id=? AND status='active' ORDER BY ts",
        (market_id,),
    ).fetchall()
    points = [
        {"ts": ts,
         "mid": (yb + ya) / 2 if yb is not None and ya is not None else None,
         "fair": fair}
        for ts, yb, ya, fair in rows
    ]
    return {"market_id": market_id, "points": points}


@app.get("/api/account")
def api_account() -> dict:
    try:
        config = Config.load()
        bal = KalshiClient(config.secrets).auth_check()
        dollars = bal.get("balance_dollars")
        if dollars is None and "balance" in bal:
            dollars = bal["balance"] / 100.0
        return {"ok": True, "balance": dollars}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/api/games")
def api_games(min_edge: float = 0.03) -> dict:
    """Group edges by game and list the recommended picks per market for each."""
    config = _config(min_edge, 0.25, 1000.0)
    games: dict[str, dict] = {}
    for e in _all_edges(config):
        g = games.setdefault(e["game_key"], {
            "game_key": e["game_key"], "sport": e["sport"],
            "label": None, "picks": [], "best": 0.0})
        g["picks"].append(e)
        g["best"] = max(g["best"], e["edge"])
        if e["market_type"] == "winner" and not g["label"]:
            g["label"] = e["title"].split(" Winner")[0].split(" [")[0]
    out = list(games.values())
    for g in out:
        if not g["label"]:
            g["label"] = g["picks"][0]["title"]
        g["picks"].sort(key=lambda p: p["edge"], reverse=True)
    out.sort(key=lambda g: g["best"], reverse=True)
    return {"games": out}


@app.get("/api/positions/live")
def api_positions_live() -> dict:
    """Pending bets marked to market against live WS prices."""
    price_map = {mid: q.yes_mid for mid, q in LIVE.markets.items() if q.yes_mid}
    conn = storage.connect()
    return {"live": bool(LIVE.connected and price_map),
            "positions": ledger.live_positions(conn, price_map)}


@app.get("/api/calendar")
def api_calendar() -> dict:
    return {"days": ledger.calendar(storage.connect())}


@app.get("/api/analysts")
def api_analysts() -> dict:
    from .analysts import grade_and_leaderboard
    from .providers import provider_name

    conn = storage.connect()
    lb = grade_and_leaderboard(conn)
    rows = storage.list_analyst_picks(conn)
    # Only show the most recent generation's pending plays (today's slate), so
    # stale picks from earlier runs don't clutter the tab.
    pending = [p for p in rows if p["status"] == "pending"]
    latest = max((p["date"] for p in pending if p["date"]), default=None)
    # Group pending legs into plays, then plays into games.
    plays: dict[str, dict] = {}
    for p in pending:
        if latest and p["date"] != latest:
            continue
        pid = p["play_id"] or f"legacy-{p['id']}"
        play = plays.setdefault(pid, {
            "analyst": p["analyst"], "game_key": p["game_key"], "label": p["game_label"],
            "sport": p["sport"], "play_type": p["play_type"] or "single",
            "confidence": p["confidence"], "rationale": p["rationale"], "legs": []})
        play["legs"].append({"label": p["leg_label"] or p["market_id"],
                             "side": p["side"], "price": p["price"], "stake": p["stake"]})
    games: dict[str, dict] = {}
    for play in plays.values():
        g = games.setdefault(play["game_key"], {"game_key": play["game_key"],
                                                "label": play["label"], "sport": play["sport"],
                                                "takes": []})
        g["takes"].append(play)
    return {"games": list(games.values()), "leaderboard": lb["leaderboard"],
            "provider": provider_name(Config.load().secrets)}


@app.post("/api/analysts/generate")
def api_analysts_generate() -> dict:
    from .analysts import generate_takes
    from .providers import provider_name

    config = Config.load()
    if not provider_name(config.secrets):
        return {"ok": False, "error": "No LLM key set — add GEMINI_API_KEY (free) to .env"}
    conn = storage.connect()
    logged = 0
    for sport in config.sports:
        try:
            if LIVE.connected and LIVE.markets:
                quotes = [q for q in LIVE.quotes() if q.sport == sport]
            else:
                quotes = KalshiClient(config.secrets).get_sports_markets(sport)
            for pick in generate_takes(quotes, sport, config.secrets):
                storage.insert_analyst_pick(conn, pick)
                logged += 1
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc), "logged": logged}
    return {"ok": True, "logged": logged}


@app.get("/api/calibration")
def api_calibration(sport: str = "mlb") -> dict:
    """Walk-forward calibration of the winner model from settled games."""
    from .backtest.model_backtest import report
    from .models.baseball import DEFAULT_MLB_RATINGS
    from .models.fit import FIT_K, games_from_settled
    from .models.soccer import DEFAULT_RATINGS

    winner_series = {"soccer": "KXWCGAME", "mlb": "KXMLBGAME"}.get(sport)
    if not winner_series:
        return {"n": 0}
    try:
        kalshi = KalshiClient(Config.load().secrets)
        raw = list(kalshi.iter_raw_markets(sport, status="settled", series_list=[winner_series]))
        games = games_from_settled(raw, sport)
        init = ({k.lower(): v for k, v in DEFAULT_RATINGS.items()} if sport == "soccer"
                else dict(DEFAULT_MLB_RATINGS))
        return report(games, init, k=FIT_K.get(sport, 20.0))
    except Exception as exc:  # noqa: BLE001
        return {"n": 0, "error": str(exc)}


@app.get("/api/live/status")
def api_live_status() -> dict:
    return {"connected": LIVE.connected, "markets": len(LIVE.markets),
            "last_update": LIVE.last_update, "error": LIVE.error}


@app.websocket("/ws")
async def ws_edges(sock: WebSocket) -> None:
    """Push recomputed edges to the browser ~1.5s while the live book is fed."""
    await sock.accept()
    config = _config(0.03, 0.25, 1000.0)
    try:
        while True:
            if LIVE.connected and LIVE.markets:
                edges = [_edge_dict(i, _sport_of(i.market_id)) for i in LIVE.edges(config)]
                edges.sort(key=lambda e: e["edge"], reverse=True)
                await sock.send_json({"edges": edges, "live": True,
                                      "last_update": LIVE.last_update})
            else:
                await sock.send_json({"edges": [], "live": False, "error": LIVE.error})
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        return


@app.get("/api/backtest")
def api_backtest(min_edge: float = 0.03) -> dict:
    r = score_from_db(min_edge=min_edge)
    return {
        "n": r.n,
        "brier": None if r.n == 0 else round(r.brier, 4),
        "log_loss": None if r.n == 0 else round(r.log_loss, 4),
        "clv": None if r.n == 0 else round(r.mean_clv, 4),
        "passes": bool(r.n and r.passes_gate),
    }


@app.get("/api/stats")
def api_stats() -> dict:
    conn = storage.connect()
    total, markets, last = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT market_id), MAX(ts) FROM snapshots"
    ).fetchone()
    settled = conn.execute(
        "SELECT COUNT(DISTINCT market_id) FROM snapshots WHERE status='settled'"
    ).fetchone()[0]
    return {"rows": total or 0, "markets": markets or 0,
            "settled": settled or 0, "last": last}


@app.post("/api/snapshot")
def api_snapshot() -> dict:
    from .logger import run_snapshot
    try:
        n = run_snapshot(Config.load())
        return {"ok": True, "rows": n}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class BetIn(BaseModel):
    legs: list[dict]
    stake: float
    sport: str = ""


@app.post("/api/parlay/quote")
def api_parlay_quote(bet: BetIn) -> dict:
    c = combine(bet.legs)
    c["correlated"] = has_correlated_legs(bet.legs)
    c["legs"] = len(bet.legs)
    return c


@app.get("/api/parlay/optimize")
def api_parlay_optimize(max_legs: int = 3, min_leg_edge: float = 0.03) -> dict:
    """Top parlays by expected ROI, built from the current single-leg edges."""
    from .engine.optimizer import optimize_parlays

    config = _config(min_leg_edge, 0.25, 1000.0)
    if LIVE.connected and LIVE.markets:
        edges = [_edge_dict(i, _sport_of(i.market_id)) for i in LIVE.edges(config)]
    else:
        kalshi = KalshiClient(config.secrets)
        edges = []
        for sport in ("soccer", "mlb"):
            try:
                quotes = kalshi.get_sports_markets(sport)
                edges += [_edge_dict(i, sport)
                          for i in find_value_edges(quotes, predictions_with_context(sport, quotes), config)]
            except Exception:  # noqa: BLE001
                pass
    return {"parlays": optimize_parlays(edges, max_legs=max_legs, min_leg_edge=min_leg_edge)}


@app.get("/api/bets")
def api_bets() -> dict:
    conn = storage.connect()
    import json as _json
    bets = []
    for b in storage.list_bets(conn):
        b["legs"] = _json.loads(b.pop("legs_json"))
        bets.append(b)
    return {"bets": bets, "summary": ledger.summary(conn)}


@app.post("/api/bets")
def api_place_bet(bet: BetIn) -> dict:
    if not bet.legs or bet.stake <= 0:
        return {"ok": False, "error": "need legs and a positive stake"}
    conn = storage.connect()
    bet_id = ledger.record_bet(conn, bet.legs, bet.stake, bet.sport)
    return {"ok": True, "id": bet_id}


@app.post("/api/bets/settle")
def api_settle() -> dict:
    conn = storage.connect()
    n = ledger.settle_pending(conn)
    return {"settled": n, "summary": ledger.summary(conn)}


class OrderIn(BaseModel):
    leg: dict
    stake: float


@app.post("/api/order")
def api_order(o: OrderIn) -> dict:
    """Send a single-leg order to Kalshi (or simulate when guarded/non-live)."""
    from .execution.trader import Trader

    config = Config.load()
    try:
        return Trader(config, KalshiClient(config.secrets)).execute_single(o.leg, o.stake)
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": str(exc)}


@app.get("/api/exec/status")
def api_exec_status() -> dict:
    ex = Config.load().execution
    return {"mode": ex.mode, "kill_switch": ex.kill_switch,
            "armed": (not ex.kill_switch) and ex.mode == "live"}
