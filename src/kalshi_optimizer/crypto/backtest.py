"""Historical backtest of the catalyst agent — a same-day read instead of a
week of passive collection.

Pulls recent news going back a few days, has Fable flag catalysts, then scores
each against the price move that *already* happened (hourly history). No waiting.

IMPORTANT — read the numbers with suspicion: Fable was trained up to a cutoff,
so on older/famous news it may already know the outcome and flatter itself
(hindsight bias, the crypto twin of the props-leakage problem). Keep the window
recent, and confirm anything promising with a short forward run before trusting.
"""

from __future__ import annotations

import time

from .catalysts import extract_catalysts
from .news import fetch_hourly, fetch_news, price_at
from .score import _is_correct, _signed_move


def _collect_articles(api_key: str, days_back: int, per_page: int = 50) -> list:
    """Page the news feed backwards to cover roughly ``days_back`` days."""
    cutoff = int(time.time()) - days_back * 86400
    before, seen, out = None, set(), []
    for _ in range(days_back * 2 + 2):     # bounded paging
        batch = fetch_news(api_key, limit=per_page, before_ts=before)
        if not batch:
            break
        for a in batch:
            if a.id not in seen and a.published >= cutoff:
                seen.add(a.id)
                out.append(a)
        oldest = min(a.published for a in batch)
        if oldest <= cutoff:
            break
        before = oldest - 1
    return out


def run_backtest(secrets, days_back: int = 14, batch_size: int = 20) -> dict:
    """Extract catalysts from recent history and score them on realized moves."""
    api_key = getattr(secrets, "coindesk_api_key", "")
    articles = _collect_articles(api_key, days_back)
    if not articles:
        return {"leaderboard": [], "n": 0, "note": "no articles fetched"}

    # Extract in batches (one Fable call each) to keep prompts sane.
    signals = []
    for i in range(0, len(articles), batch_size):
        signals += extract_catalysts(articles[i:i + batch_size], secrets)
        time.sleep(1.0)

    price_series: dict[str, list] = {}
    scored = []
    for s in signals:
        asset = s["asset"]
        if asset not in price_series:
            try:
                price_series[asset] = fetch_hourly(asset, api_key)
            except Exception:  # noqa: BLE001 - unknown ticker
                price_series[asset] = []
        series = price_series[asset]
        entry = price_at(series, s["published"])
        after = price_at(series, s["published"] + (s["horizon_hours"] or 0) * 3600)
        if not entry or after is None:
            continue
        move = (after - entry) / entry
        s["move_pct"] = round(move, 5)
        s["correct"] = _is_correct(s["direction"], move)
        scored.append(s)

    return {"leaderboard": _leaderboard(scored), "n": len(scored),
            "flagged": len(signals), "articles": len(articles)}


def _leaderboard(scored: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for s in scored:
        groups.setdefault(s["catalyst_type"], []).append(s)
        groups.setdefault("ALL", []).append(s)

    def summarize(name, items):
        n = len(items)
        hits = sum(1 for s in items if s["correct"])
        edge = sum(_signed_move(s["direction"], s["move_pct"]) for s in items) / n
        return {"group": name, "n": n, "hit_rate": round(hits / n, 3),
                "mean_signed_move": round(edge, 4)}

    board = [summarize(k, v) for k, v in groups.items()]
    board.sort(key=lambda x: (x["group"] != "ALL", -x["n"]))
    return board
