"""Fit Elo ratings from finished games (the biggest accuracy lever).

Replays settled winner markets chronologically: each game nudges the two teams'
ratings toward the result. Fitted ratings are persisted per sport and loaded by
the models on startup (falling back to the seeded defaults when absent).

Soccer ratings are keyed by lowercased country name (the model looks teams up by
name); MLB by Kalshi team code. History is only as deep as Kalshi's settled
markets, so this strengthens as more games finish — and it's still subject to
the same CLV gate before being trusted.
"""

from __future__ import annotations

import json
import os

RATINGS_DIR = "data"
BASE_RATING = 1500.0
FIT_K = {"soccer": 20.0, "mlb": 6.0}


def ratings_path(sport: str) -> str:
    return os.path.join(RATINGS_DIR, f"ratings_{sport}.json")


def load_ratings(sport: str) -> dict[str, float]:
    """Fitted ratings for a sport, or {} if none saved yet."""
    path = ratings_path(sport)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {(k.lower() if sport == "soccer" else k): float(v) for k, v in data.items()}


def save_ratings(sport: str, ratings: dict[str, float]) -> None:
    os.makedirs(RATINGS_DIR, exist_ok=True)
    with open(ratings_path(sport), "w") as fh:
        json.dump({k: round(v, 1) for k, v in ratings.items()}, fh, indent=0)


def games_from_settled(raw_markets: list[dict], sport: str) -> list[tuple[str, str, float]]:
    """Chronological (team_a, team_b, score_a) from settled winner markets.

    score_a is 1 (a won), 0 (a lost) or 0.5 (draw). Team keys are lowercased
    names for soccer, codes for MLB — matching each model's rating keys.
    """
    events: dict[str, dict] = {}
    for m in raw_markets:
        ticker = m.get("ticker", "")
        code = ticker.rsplit("-", 1)[-1] if "-" in ticker else None
        if not code:
            continue
        ev = events.setdefault(m.get("event_ticker", ticker),
                               {"ts": m.get("close_time") or m.get("expiration_time") or "",
                                "teams": {}, "winner": None})
        label = (m.get("yes_sub_title") or "").split(":", 1)[-1].strip()
        if code != "TIE":
            ev["teams"][code] = label.lower() if sport == "soccer" else code
        if m.get("result") == "yes":
            ev["winner"] = code

    games: list[tuple[str, str, str, float]] = []
    for ev in events.values():
        teams = list(ev["teams"].items())
        if len(teams) != 2 or ev["winner"] is None:
            continue
        (code_a, key_a), (code_b, key_b) = teams
        if ev["winner"] == "TIE":
            score_a = 0.5
        elif ev["winner"] == code_a:
            score_a = 1.0
        elif ev["winner"] == code_b:
            score_a = 0.0
        else:
            continue
        games.append((ev["ts"], key_a, key_b, score_a))
    games.sort(key=lambda g: g[0])
    return [(a, b, s) for _ts, a, b, s in games]


def fit_ratings(games: list[tuple[str, str, float]], init: dict[str, float],
                k: float = 20.0) -> dict[str, float]:
    """Replay games through Elo starting from ``init`` priors."""
    r = dict(init)

    def rt(t: str) -> float:
        return r.get(t, BASE_RATING)

    for a, b, score_a in games:
        pa = 1.0 / (1.0 + 10 ** (-(rt(a) - rt(b)) / 400.0))
        r[a] = rt(a) + k * (score_a - pa)
        r[b] = rt(b) + k * ((1.0 - score_a) - (1.0 - pa))
    return r
