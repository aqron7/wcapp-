"""SQLite storage for market snapshots (the backtest data source).

Every snapshot run appends one row per market: its prices, the model's fair
probability at that moment, and (once the market settles) the result. Over time
this accumulates the entry prices, closing prices, and outcomes the backtester
needs to measure calibration and closing-line value.

The DB lives under the project-root ``data/`` dir (gitignored) so it is never
committed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = "data/snapshots.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts            TEXT NOT NULL,
    sport         TEXT,
    event_ticker  TEXT,
    market_id     TEXT NOT NULL,
    outcome       TEXT,
    yes_bid       REAL,
    yes_ask       REAL,
    model_fair    REAL,
    status        TEXT,
    result        TEXT
);
CREATE INDEX IF NOT EXISTS idx_snap_market ON snapshots(market_id);
CREATE INDEX IF NOT EXISTS idx_snap_ts ON snapshots(ts);
"""


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_snapshots(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """Insert snapshot rows: (ts, sport, event_ticker, market_id, outcome,
    yes_bid, yes_ask, model_fair, status, result)."""
    conn.executemany(
        "INSERT INTO snapshots "
        "(ts, sport, event_ticker, market_id, outcome, yes_bid, yes_ask, "
        " model_fair, status, result) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    return len(rows)
