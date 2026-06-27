"""Match context adjustments: altitude, host advantage, recent form.

Produces Elo-point deltas per team plus a transparent breakdown so the UI can
show *why* a fair value moved. Coefficients are deliberately modest and, like
the base ratings, must be validated through the backtester before being trusted
— a plausible-sounding adjustment that hurts CLV is worse than none.

Venue is not present in Kalshi market data, so altitude/host only apply when a
venue is supplied (e.g. from a schedule map in data/venues.json). Without it,
the breakdown says so and no adjustment is made.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 2026 FIFA World Cup host venues -> elevation (metres).
VENUE_ALTITUDE_M: dict[str, int] = {
    "mexico city": 2240, "guadalajara": 1566, "monterrey": 540,
    "denver": 1609, "atlanta": 320, "kansas city": 270, "dallas": 160,
    "arlington": 160, "boston": 30, "foxborough": 30, "houston": 30,
    "los angeles": 40, "inglewood": 40, "miami": 2, "new york": 5,
    "east rutherford": 5, "philadelphia": 12, "san francisco": 9,
    "santa clara": 9, "seattle": 45, "toronto": 76, "vancouver": 4,
}

# Typical home-pitch elevation a national team is acclimatised to (metres).
NATION_HOME_ALTITUDE_M: dict[str, int] = {
    "mexico": 2240, "bolivia": 3640, "ecuador": 2850, "colombia": 2640,
    "peru": 3250, "iran": 1200, "afghanistan": 1790, "guatemala": 1500,
    "costa rica": 1170, "honduras": 990, "south africa": 1750,
}

HOST_NATIONS_2026 = {"usa": "usa", "united states": "usa",
                     "canada": "canada", "mexico": "mexico"}
VENUE_COUNTRY = {  # which host country a venue sits in
    "mexico city": "mexico", "guadalajara": "mexico", "monterrey": "mexico",
    "toronto": "canada", "vancouver": "canada",
}  # everything else defaults to "usa"

# Tunable coefficients (Elo points). Validate before trusting.
ALT_TOLERANCE_M = 600       # difference below this is ignored
ALT_POINTS_PER_KM = 45      # penalty per km of unacclimatised altitude deficit
ALT_MAX_PENALTY = 55
HOST_BONUS = 55
FORM_POINTS_PER_WIN_RATE = 80   # form delta = (recent_win_rate - 0.5) * this


@dataclass
class MatchContext:
    venue_city: str | None = None
    home_form: float | None = None   # recent win-rate [0,1] for team_a
    away_form: float | None = None   # recent win-rate [0,1] for team_b
    extra: dict = field(default_factory=dict)


def _altitude_penalty(nation: str, venue_alt: int) -> float:
    home_alt = NATION_HOME_ALTITUDE_M.get(nation.lower(), 50)
    deficit = max(0, venue_alt - home_alt - ALT_TOLERANCE_M)
    return -min(deficit / 1000.0 * ALT_POINTS_PER_KM, ALT_MAX_PENALTY)


def adjust(team_a: str, team_b: str, ctx: MatchContext | None):
    """Return (adj_a, adj_b, breakdown). breakdown: list of (factor, team, pts)."""
    adj_a = adj_b = 0.0
    breakdown: list[tuple[str, str, float]] = []
    if ctx is None:
        return 0.0, 0.0, [("context", "—", 0.0)]

    city = (ctx.venue_city or "").lower()
    venue_alt = VENUE_ALTITUDE_M.get(city)

    if venue_alt is not None and venue_alt > 1000:
        pa, pb = _altitude_penalty(team_a, venue_alt), _altitude_penalty(team_b, venue_alt)
        adj_a += pa; adj_b += pb
        breakdown.append(("altitude", team_a, round(pa, 1)))
        breakdown.append(("altitude", team_b, round(pb, 1)))
    elif city and venue_alt is None:
        breakdown.append(("altitude", "venue unknown", 0.0))

    if city:
        host = VENUE_COUNTRY.get(city, "usa")
        if HOST_NATIONS_2026.get(team_a.lower()) == host:
            adj_a += HOST_BONUS; breakdown.append(("host", team_a, HOST_BONUS))
        if HOST_NATIONS_2026.get(team_b.lower()) == host:
            adj_b += HOST_BONUS; breakdown.append(("host", team_b, HOST_BONUS))

    if ctx.home_form is not None:
        d = round((ctx.home_form - 0.5) * FORM_POINTS_PER_WIN_RATE, 1)
        adj_a += d; breakdown.append(("form", team_a, d))
    if ctx.away_form is not None:
        d = round((ctx.away_form - 0.5) * FORM_POINTS_PER_WIN_RATE, 1)
        adj_b += d; breakdown.append(("form", team_b, d))

    return adj_a, adj_b, breakdown
