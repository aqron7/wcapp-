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


def strikeout_predictions(quotes, pitcher_lambdas: dict[str, float]):
    """Predictions for KXMLBKS markets given {pitcher_name_lower: K lambda}.

    Pure: the pitcher is read from the market label ("Drew Rasmussen: 10+").
    """
    from ..types import Prediction

    preds = []
    for q in quotes:
        if not (q.market_id or "").startswith("KXMLBKS") or q.strike is None or not q.outcome:
            continue
        name = (q.outcome_label or "").split(":", 1)[0].strip().lower()
        lam = pitcher_lambdas.get(name)
        if lam is None:
            continue
        preds.append(Prediction(q.event_key, q.outcome, prob_over(lam, q.strike), "mlb", "k_prop_v1"))
    return preds


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


def strikeout_predictions_live(quotes):
    """Wire KXMLBKS markets to live StatsAPI projections (best-effort)."""
    dates = _dates_in_quotes(quotes)
    if not dates:
        return []
    from ..data.mlb_stats import MlbStatsClient
    try:
        lambdas = build_pitcher_lambdas(MlbStatsClient(), dates)
    except Exception:  # noqa: BLE001 - feed optional; never break edge calc
        return []
    return strikeout_predictions(quotes, lambdas)


def expected_innings(season: dict, default: float = DEFAULT_START_INNINGS) -> float:
    """Average innings per start from season stats, clamped to a sane range."""
    gs = season.get("gs") or 0
    ip = season.get("ip") or 0.0
    if gs <= 0 or ip <= 0:
        return default
    return max(3.0, min(7.0, ip / gs))
