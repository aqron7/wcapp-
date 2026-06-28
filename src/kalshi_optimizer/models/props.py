"""Player-prop models. First up: pitcher strikeouts (Poisson).

A starter's strikeouts over a game ~ Poisson(lambda), with lambda from season
K/9 scaled to expected innings:

    lambda = K9 * expected_innings / 9

P(over line) follows as the Poisson survival above floor(line). HR/hit/goal
props will follow the same shape once their batter/scorer rates are wired.
"""

from __future__ import annotations

from math import exp, factorial, floor

DEFAULT_START_INNINGS = 5.5


def strikeout_lambda(k_per9: float, expected_innings: float = DEFAULT_START_INNINGS) -> float:
    return max(0.1, k_per9 * expected_innings / 9.0)


def prob_over(lmbda: float, line: float) -> float:
    """P(count > line) for a Poisson(lmbda)."""
    k = int(floor(line))
    cdf = sum(exp(-lmbda) * lmbda ** i / factorial(i) for i in range(0, k + 1))
    return max(0.0, min(1.0, 1.0 - cdf))


def _prop_predictions(quotes, prefix: str, lambdas: dict[str, float], model_name: str):
    """Generic player-prop predictions: market label "Name: N+" -> P(over line)."""
    from ..types import Prediction

    preds = []
    for q in quotes:
        if not (q.market_id or "").startswith(prefix) or q.strike is None or not q.outcome:
            continue
        name = (q.outcome_label or "").split(":", 1)[0].strip().lower()
        lam = lambdas.get(name)
        if lam is None:
            continue
        preds.append(Prediction(q.event_key, q.outcome, prob_over(lam, q.strike), "mlb", model_name))
    return preds


def strikeout_predictions(quotes, pitcher_lambdas: dict[str, float]):
    """Predictions for KXMLBKS markets given {pitcher_name_lower: K lambda}."""
    return _prop_predictions(quotes, "KXMLBKS", pitcher_lambdas, "k_prop_v1")


def _names_for_prefix(quotes, prefix: str) -> set[str]:
    return {(q.outcome_label or "").split(":", 1)[0].strip().lower()
            for q in quotes if (q.market_id or "").startswith(prefix) and q.outcome_label}


_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _dates_in_quotes(quotes) -> set[str]:
    import re
    dates = set()
    for q in quotes:
        if (q.market_id or "").startswith("KXMLBKS") and q.event_key:
            m = re.search(r"(\d{2})([A-Z]{3})(\d{2})", q.event_key)
            if m and m.group(2) in _MONTHS:
                dates.add(f"20{m.group(1)}-{_MONTHS[m.group(2)]:02d}-{m.group(3)}")
    return dates


def build_pitcher_lambdas(client, dates) -> dict[str, float]:
    """Fetch probable pitchers + season K rates -> {name_lower: K lambda}."""
    out: dict[str, float] = {}
    seen: dict[int, float] = {}
    for date in dates:
        for p in client.probable_pitchers(date):
            pid = p.get("pitcher_id")
            if pid is None:
                continue
            if pid not in seen:
                season = client.pitcher_season(pid)
                seen[pid] = strikeout_lambda(season["k9"], expected_innings(season)) \
                    if season and season["k9"] > 0 else 0.0
            if seen[pid] > 0 and p.get("pitcher"):
                out[p["pitcher"].lower()] = seen[pid]
    return out


def build_batter_lambdas(client, names: set[str], stat: str, season: int = 2026) -> dict[str, float]:
    """{name_lower: per-game rate} for the needed batters' season ``stat``."""
    if not names:
        return {}
    ids = client.all_players(season)
    out: dict[str, float] = {}
    for name in names:
        pid = ids.get(name)
        if not pid:
            continue
        s = client.batter_season(pid)
        if s and s.get("games", 0) > 0 and s.get(stat, 0) > 0:
            out[name] = max(0.01, s[stat] / s["games"])
    return out


def prop_predictions_live(quotes):
    """Wire all MLB player props (K / HR / hits) to live StatsAPI projections.

    Best-effort and fully guarded — a feed hiccup never breaks edge calc.
    """
    from ..data.mlb_stats import MlbStatsClient

    preds = []
    try:
        client = MlbStatsClient()
        dates = _dates_in_quotes(quotes)
        if dates:
            preds += strikeout_predictions(quotes, build_pitcher_lambdas(client, dates))
        hr_names = _names_for_prefix(quotes, "KXMLBHR")
        if hr_names:
            preds += _prop_predictions(quotes, "KXMLBHR",
                                       build_batter_lambdas(client, hr_names, "hr"), "hr_prop_v1")
        hit_names = _names_for_prefix(quotes, "KXMLBHIT")
        if hit_names:
            preds += _prop_predictions(quotes, "KXMLBHIT",
                                       build_batter_lambdas(client, hit_names, "hits"), "hits_prop_v1")
    except Exception:  # noqa: BLE001
        pass
    return preds


def expected_innings(season: dict, default: float = DEFAULT_START_INNINGS) -> float:
    """Average innings per start from season stats, clamped to a sane range."""
    gs = season.get("gs") or 0
    ip = season.get("ip") or 0.0
    if gs <= 0 or ip <= 0:
        return default
    return max(3.0, min(7.0, ip / gs))
