"""Snapshot logger: record live market prices + model fair values, and capture
settled results. Run on a schedule so backtest history accumulates.

This is time-sensitive: closing prices and outcomes cannot be backfilled, so the
sooner this runs regularly, the sooner the backtester gate has data.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import storage
from .config import Config
from .data.kalshi import KalshiClient
from .engine.value import predictions_with_context


def run_snapshot(config: Config, db_path: str = storage.DEFAULT_DB) -> int:
    """Capture one snapshot across configured sports. Returns rows written."""
    kalshi = KalshiClient(config.secrets)
    conn = storage.connect(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    rows: list[tuple] = []

    for sport in config.sports:
        # Active markets: record prices + the model's current fair value.
        quotes = kalshi.get_sports_markets(sport)
        fair = {(p.event_key, p.outcome): p.fair_prob
                for p in predictions_with_context(sport, quotes)}
        for q in quotes:
            rows.append((
                ts, sport, q.event_key, q.market_id, q.outcome,
                q.yes_bid, q.yes_ask, fair.get((q.event_key, q.outcome)),
                "active", None,
            ))

        # Settled markets: record the result so outcomes can be scored later.
        for m in kalshi.iter_raw_markets(sport, status="settled"):
            ticker = m.get("ticker", "")
            outcome = ticker.rsplit("-", 1)[-1] if "-" in ticker else None
            rows.append((
                ts, sport, m.get("event_ticker"), ticker, outcome,
                None, None, None, "settled", m.get("result") or None,
            ))

    return storage.insert_snapshots(conn, rows)
