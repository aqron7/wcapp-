"""FastAPI backend for the web dashboard (phase 5).

Serves the single-page frontend plus JSON endpoints that reuse the same engine
the CLI uses. Run with:  python -m kalshi_optimizer dashboard

Needs the optional dashboard dependency:  pip install -e ".[dashboard]"
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import ledger, storage
from .backtest.backtester import score_from_db
from .config import Config
from .data.kalshi import KalshiClient
from .engine.parlay import combine, has_correlated_legs
from .engine.value import find_value_edges, predictions_for_sport

app = FastAPI(title="Kalshi Edge")
_INDEX = Path(__file__).with_name("static") / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX.read_text(encoding="utf-8")


@app.get("/api/edges")
def api_edges(sports: str = "soccer,mlb", min_edge: float = 0.03,
              kelly: float = 0.25, bankroll: float = 1000.0) -> dict:
    config = Config.load()
    config.bankroll = bankroll
    config.edge.min_edge = min_edge
    config.sizing.kelly_fraction = kelly

    kalshi = KalshiClient(config.secrets)
    edges, errors = [], []
    for sport in [s for s in sports.split(",") if s]:
        try:
            quotes = kalshi.get_sports_markets(sport)
            preds = predictions_for_sport(sport, quotes)
            for idea in find_value_edges(quotes, preds, config):
                side_fair = idea.fair_prob if idea.side.value == "yes" else 1 - idea.fair_prob
                edges.append({
                    "sport": sport,
                    "title": idea.title,
                    "market_id": idea.market_id,
                    "event_ticker": idea.market_id.rsplit("-", 1)[0],
                    "outcome": idea.market_id.rsplit("-", 1)[-1],
                    "side": idea.side.value,
                    "price": round(idea.price, 2),
                    "fair": round(idea.fair_prob, 2),
                    "fair_side": round(side_fair, 2),
                    "edge": round(idea.edge, 4),
                    "stake": idea.stake,
                    "why": idea.rationale,
                })
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sport}: {exc}")
    edges.sort(key=lambda e: e["edge"], reverse=True)
    return {"edges": edges, "errors": errors}


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
