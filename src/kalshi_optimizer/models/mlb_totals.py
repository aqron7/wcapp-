"""Empirical MLB run-total distribution, fit from settled over/under markets.

The flat Poisson(8.6) under-predicted overs (too thin-tailed). Instead, learn the
real exceedance curve directly: for each line L, P(total > L) = the historical
fraction of games that went over L. That's calibrated by construction and fatter-
tailed than Poisson. Persisted to data/ and loaded by the totals model; falls
back to Poisson(8.6) when no fit exists.

This is a population-average curve (ignores matchup) — a per-team run model is the
next upgrade — but it fixes the calibration that made the flat model unusable.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

DIST_PATH = "data/mlb_totals_dist.json"


def load_distribution(path: str = DIST_PATH) -> dict[float, float]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return {float(k): float(v) for k, v in json.load(fh).items()}
    except (OSError, ValueError):
        return {}


def save_distribution(dist: dict[float, float], path: str = DIST_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({str(k): round(v, 4) for k, v in dist.items()}, fh, indent=0)


def fit_distribution(settled: list[dict], min_n: int = 20) -> dict[float, float]:
    """{line: empirical P(over)} from settled totals markets (lines with >=min_n)."""
    agg: dict[float, list[int]] = defaultdict(lambda: [0, 0])  # line -> [overs, total]
    for m in settled:
        if m.get("result") not in ("yes", "no") or m.get("floor_strike") is None:
            continue
        a = agg[float(m["floor_strike"])]
        a[1] += 1
        if m["result"] == "yes":
            a[0] += 1
    return {line: overs / n for line, (overs, n) in agg.items() if n >= min_n}


def prob_over(line: float, dist: dict[float, float]) -> float:
    """P(total > line), linearly interpolated from the empirical curve.

    Falls back to Poisson(8.6) if no fitted distribution is available.
    """
    if not dist:
        from .poisson import prob_over as poisson_over
        return poisson_over(8.6, line)
    lines = sorted(dist)
    if line <= lines[0]:
        return dist[lines[0]]
    if line >= lines[-1]:
        return dist[lines[-1]]
    for i in range(1, len(lines)):
        if line <= lines[i]:
            lo, hi = lines[i - 1], lines[i]
            t = (line - lo) / (hi - lo)
            return dist[lo] + t * (dist[hi] - dist[lo])
    return dist[lines[-1]]
