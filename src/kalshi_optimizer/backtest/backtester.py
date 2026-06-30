"""Backtester — the validation gate (phase 3).

No real or automated trades until the model clears this bar:
  - good calibration (low Brier score / log-loss), and
  - positive **closing-line value (CLV)** — our entry consistently beats the
    market's closing price, the single best predictor of long-run edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def brier_score(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error of probabilistic predictions (lower is better)."""
    if not probs:
        return float("nan")
    return sum((p - o) ** 2 for p, o in zip(probs, outcomes)) / len(probs)


def log_loss(probs: list[float], outcomes: list[int], eps: float = 1e-9) -> float:
    """Negative log-likelihood (lower is better)."""
    if not probs:
        return float("nan")
    total = 0.0
    for p, o in zip(probs, outcomes):
        p = min(1 - eps, max(eps, p))
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / len(probs)


def closing_line_value(entry_prices: list[float], closing_prices: list[float]) -> float:
    """Average edge of our entry vs the closing price (higher is better).

    Positive mean CLV is the go/no-go signal for trading real money.
    """
    if not entry_prices:
        return float("nan")
    return sum(c - e for e, c in zip(entry_prices, closing_prices)) / len(entry_prices)


@dataclass
class BacktestResult:
    n: int
    brier: float
    log_loss: float
    mean_clv: float

    @property
    def passes_gate(self) -> bool:
        """Conservative gate: positive CLV and a calibrated Brier score."""
        return self.mean_clv > 0 and self.brier < 0.25

    def passes_bucket_gate(self, min_n: int = 100) -> bool:
        """Per-bucket gate: the full gate plus enough settled samples to trust it.

        A big CLV on a tiny, correlated sample is noise, so a bucket isn't
        considered armed until it has cleared ``min_n``."""
        return self.n >= min_n and self.passes_gate


def run_backtest(
    probs: list[float],
    outcomes: list[int],
    entry_prices: list[float],
    closing_prices: list[float],
) -> BacktestResult:
    """Compute the full validation report over historical predictions."""
    return BacktestResult(
        n=len(probs),
        brier=brier_score(probs, outcomes),
        log_loss=log_loss(probs, outcomes),
        mean_clv=closing_line_value(entry_prices, closing_prices),
    )


def _scored_rows(db_path: str | None = None, min_edge: float = 0.03) -> list[dict]:
    """Per-market scored rows: each has sport, market_type, prob, outcome,
    entry/closing price (bought-side). Shared by overall + grouped scoring.

    For each market it derives an entry (first liquid snapshot with a model
    fair value), a closing price (last liquid snapshot), and the settled
    result. Only markets where the model had an edge >= ``min_edge`` at entry
    are kept. Prices are in terms of the side we'd have bought, so positive
    mean CLV means our entry beat the closing line.
    """
    from collections import defaultdict

    from .. import storage
    from ..data.kalshi import SERIES_TYPE

    conn = storage.connect(db_path or storage.DEFAULT_DB)
    cur = conn.execute(
        "SELECT market_id, sport, ts, yes_bid, yes_ask, model_fair, status, result "
        "FROM snapshots ORDER BY ts"
    )
    markets: dict[str, dict] = defaultdict(lambda: {"entries": [], "result": None, "sport": None})
    for market_id, sport, ts, yes_bid, yes_ask, fair, status, result in cur:
        m = markets[market_id]
        m["sport"] = sport or m["sport"]
        if status == "settled" and result in ("yes", "no"):
            m["result"] = result
        elif status == "active" and yes_bid is not None and yes_ask is not None and fair is not None:
            m["entries"].append((ts, (yes_bid + yes_ask) / 2.0, fair))

    rows: list[dict] = []
    for market_id, m in markets.items():
        if m["result"] is None or not m["entries"]:
            continue
        m["entries"].sort()
        _, entry_yes, fair = m["entries"][0]
        _, closing_yes, _ = m["entries"][-1]
        if abs(fair - entry_yes) < min_edge:
            continue  # we wouldn't have bet this market
        bet_yes = fair >= entry_yes
        rows.append({
            "sport": m["sport"] or "?",
            "market_type": SERIES_TYPE.get(market_id.split("-", 1)[0], "winner"),
            "prob": fair,
            "outcome": 1 if m["result"] == "yes" else 0,
            "entry": entry_yes if bet_yes else 1.0 - entry_yes,
            "close": closing_yes if bet_yes else 1.0 - closing_yes,
        })
    return rows


def _result_from_rows(rows: list[dict]) -> BacktestResult:
    return run_backtest([r["prob"] for r in rows], [r["outcome"] for r in rows],
                        [r["entry"] for r in rows], [r["close"] for r in rows])


def score_from_db(db_path: str | None = None, min_edge: float = 0.03) -> BacktestResult:
    """Overall validation report from the snapshot DB (the gate)."""
    return _result_from_rows(_scored_rows(db_path, min_edge))


def armed_score_from_db(tradeable_types, db_path: str | None = None,
                        min_edge: float = 0.03) -> BacktestResult:
    """Gate scored over ONLY the armed market types — so props/winners don't
    dilute the signal for the markets you'd actually trade. Empty/None = all."""
    rows = _scored_rows(db_path, min_edge)
    if tradeable_types:
        allowed = set(tradeable_types)
        rows = [r for r in rows if r["market_type"] in allowed]
    return _result_from_rows(rows)


def grouped_score_from_db(db_path: str | None = None,
                          min_edge: float = 0.03) -> list[tuple[str, BacktestResult]]:
    """(label, result) per sport and per sport/market-type, for the breakdown.

    Sorted by sample size so the most-supported groups read first.
    """
    from collections import defaultdict

    rows = _scored_rows(db_path, min_edge)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["sport"]].append(r)
        groups[f"{r['sport']}/{r['market_type']}"].append(r)
    out = [(label, _result_from_rows(rs)) for label, rs in groups.items()]
    out.sort(key=lambda x: x[1].n, reverse=True)
    return out
