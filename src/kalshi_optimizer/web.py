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
from .engine.value import find_value_edges, predictions_for_sport

app = FastAPI(title="Kalshi Edge")
_INDEX = Path(__file__).with_name("static") / "index.html"
LIVE = realtime.LiveBook()


def _edge_dict(idea, sport: str) -> dict:
    side_fair = idea.fair_prob if idea.side.value == "yes" else 1 - idea.fair_prob
    return {
        "sport": sport, "title": idea.title, "market_id": idea.market_id,
        "event_ticker": idea.market_id.rsplit("-", 1)[0],
        "outcome": idea.market_id.rsplit("-", 1)[-1],
        "side": idea.side.value, "price": round(idea.price, 2),
        "fair": round(idea.fair_prob, 2), "fair_side": round(side_fair, 2),
        "edge": round(idea.edge, 4), "stake": idea.stake, "why": idea.rationale,
    }


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
            preds = predictions_for_sport(sport, quotes)
            edges += [_edge_dict(i, sport) for i in find_value_edges(quotes, preds, config)]
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sport}: {exc}")
    edges.sort(key=lambda e: e["edge"], reverse=True)
    return {"edges": edges, "errors": errors, "live": False}


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
