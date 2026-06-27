"""Recent-form / trends engine.

Derives each team's recent win rate from settled results already captured in the
snapshot DB. Each game's per-team market settles 'yes' (that team won) or 'no'
(it lost), so we can reconstruct a rolling record per team without any external
feed. Win rate feeds the models as a small form adjustment.

Sparse early on (needs settled games to accumulate) — it strengthens over time
as the logger runs.
"""

from __future__ import annotations

from collections import defaultdict

from . import storage


def recent_form(conn, sport: str, lookback: int = 10) -> dict[str, float]:
    """Return {team_outcome_code: recent_win_rate} over the last ``lookback`` games."""
    rows = conn.execute(
        "SELECT market_id, outcome, result FROM snapshots "
        "WHERE sport=? AND status='settled' AND result IN ('yes','no') "
        "ORDER BY ts DESC",
        (sport,),
    )
    history: dict[str, list[int]] = defaultdict(list)
    seen: set[str] = set()
    for market_id, outcome, result in rows:
        if market_id in seen:
            continue
        seen.add(market_id)
        if not outcome or outcome == "TIE":
            continue
        if len(history[outcome]) < lookback:
            history[outcome].append(1 if result == "yes" else 0)
    return {team: sum(v) / len(v) for team, v in history.items() if v}


def form_table(sport: str, lookback: int = 10) -> dict[str, float]:
    """Convenience wrapper that opens the default DB."""
    return recent_form(storage.connect(), sport, lookback)
