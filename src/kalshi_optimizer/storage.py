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

CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_ts     TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- 'single' | 'parlay'
    sport         TEXT,
    description   TEXT,
    legs_json     TEXT NOT NULL,          -- [{market_id,event_ticker,outcome,side,price,fair_side,label}]
    stake         REAL NOT NULL,
    entry_price   REAL,                   -- combined cost per $1 (product of legs)
    fair_prob     REAL,                   -- combined model probability
    edge          REAL,
    status        TEXT NOT NULL,          -- 'pending' | 'won' | 'lost'
    payout        REAL,
    pnl           REAL,
    closing_price REAL,                   -- combined closing cost (for CLV)
    result_ts     TEXT
);
"""


def connect(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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


# --------------------------------------------------------------------------- #
# Bets ledger
# --------------------------------------------------------------------------- #
def insert_bet(conn: sqlite3.Connection, bet: dict) -> int:
    cols = ("placed_ts", "kind", "sport", "description", "legs_json", "stake",
            "entry_price", "fair_prob", "edge", "status", "payout", "pnl",
            "closing_price", "result_ts")
    cur = conn.execute(
        f"INSERT INTO bets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        tuple(bet.get(c) for c in cols),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_bets(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM bets ORDER BY placed_ts DESC")]


def update_bet(conn: sqlite3.Connection, bet_id: int, **fields) -> None:
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE bets SET {sets} WHERE id=?", (*fields.values(), bet_id))
    conn.commit()


def result_for_market(conn: sqlite3.Connection, market_id: str) -> str | None:
    """Settled result ('yes'/'no') for a market, from snapshots."""
    row = conn.execute(
        "SELECT result FROM snapshots WHERE market_id=? AND status='settled' "
        "AND result IN ('yes','no') ORDER BY ts DESC LIMIT 1",
        (market_id,),
    ).fetchone()
    return row[0] if row else None


def closing_mid(conn: sqlite3.Connection, market_id: str) -> float | None:
    """Last liquid YES mid-price seen for a market (its closing line)."""
    row = conn.execute(
        "SELECT yes_bid, yes_ask FROM snapshots WHERE market_id=? AND status='active' "
        "AND yes_bid IS NOT NULL AND yes_ask IS NOT NULL ORDER BY ts DESC LIMIT 1",
        (market_id,),
    ).fetchone()
    return (row[0] + row[1]) / 2.0 if row else None
