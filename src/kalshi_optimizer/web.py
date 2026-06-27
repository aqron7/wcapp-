"""FastAPI backend for the web dashboard (phase 5).

Serves the single-page frontend plus JSON endpoints that reuse the same engine
the CLI uses. Run with:  python -m kalshi_optimizer dashboard

Needs the optional dashboard dependency:  pip install -e ".[dashboard]"
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from . import storage
from .backtest.backtester import score_from_db
from .config import Config
from .data.kalshi import KalshiClient
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
                edges.append({
                    "sport": sport,
                    "title": idea.title,
                    "side": idea.side.value,
                    "price": round(idea.price, 2),
                    "fair": round(idea.fair_prob, 2),
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
