"""Event-key normalization.

The arbitrage scanner can only compare prices across platforms if the SAME
real-world game maps to the SAME ``event_key`` on every platform. Titles differ
wildly ("Will the Yankees beat the Red Sox?" vs "Yankees vs. Red Sox"), so we
canonicalize teams to known tokens and build an order-independent key:

    "<sport>|<YYYY-MM-DD>|<team_a>+<team_b>"   (teams sorted alphabetically)

Sorting the pair makes the key independent of home/away labeling, which is not
consistent across platforms.

This is intentionally extensible: add aliases as you encounter new naming. When
no two known teams are found, ``event_key`` returns None and the market is
simply skipped by the scanner (never mismatched).
"""

from __future__ import annotations

import re
from datetime import date

# Canonical MLB team token -> set of aliases/substrings that identify it.
MLB_TEAMS: dict[str, list[str]] = {
    "ARI": ["diamondbacks", "arizona", "d-backs", "dbacks"],
    "ATL": ["braves", "atlanta"],
    "BAL": ["orioles", "baltimore"],
    "BOS": ["red sox", "boston"],
    "CHC": ["cubs", "chicago cubs"],
    "CWS": ["white sox", "chicago white sox"],
    "CIN": ["reds", "cincinnati"],
    "CLE": ["guardians", "cleveland"],
    "COL": ["rockies", "colorado"],
    "DET": ["tigers", "detroit"],
    "HOU": ["astros", "houston"],
    "KC": ["royals", "kansas city"],
    "LAA": ["angels", "los angeles angels", "anaheim"],
    "LAD": ["dodgers", "los angeles dodgers"],
    "MIA": ["marlins", "miami"],
    "MIL": ["brewers", "milwaukee"],
    "MIN": ["twins", "minnesota"],
    "NYM": ["mets", "new york mets"],
    "NYY": ["yankees", "new york yankees"],
    "OAK": ["athletics", "oakland", "a's"],
    "PHI": ["phillies", "philadelphia"],
    "PIT": ["pirates", "pittsburgh"],
    "SD": ["padres", "san diego"],
    "SF": ["giants", "san francisco"],
    "SEA": ["mariners", "seattle"],
    "STL": ["cardinals", "st. louis", "st louis"],
    "TB": ["rays", "tampa bay"],
    "TEX": ["rangers", "texas"],
    "TOR": ["blue jays", "toronto"],
    "WSH": ["nationals", "washington"],
}

# A starter set of World Cup national teams. Extend as needed.
SOCCER_TEAMS: dict[str, list[str]] = {
    "USA": ["united states", "usa", "usmnt"],
    "ARG": ["argentina"],
    "BRA": ["brazil"],
    "FRA": ["france"],
    "ENG": ["england"],
    "ESP": ["spain"],
    "GER": ["germany"],
    "POR": ["portugal"],
    "NED": ["netherlands", "holland"],
    "BEL": ["belgium"],
    "ITA": ["italy"],
    "MEX": ["mexico"],
    "CAN": ["canada"],
    "CRO": ["croatia"],
    "URU": ["uruguay"],
    "COL": ["colombia"],
    "JPN": ["japan"],
    "KOR": ["south korea", "korea republic"],
    "MAR": ["morocco"],
    "SEN": ["senegal"],
}

_ALIASES = {"mlb": MLB_TEAMS, "soccer": SOCCER_TEAMS}


def canonical_teams(text: str, sport: str) -> list[str]:
    """Return canonical team tokens found in ``text``, ordered by appearance.

    Order matters: for a title like "Will the Yankees beat the Red Sox?" the
    first team is the subject of the "Yes" outcome.
    """
    table = _ALIASES.get(sport, {})
    low = text.lower()
    found: dict[str, int] = {}
    for token, aliases in table.items():
        for alias in aliases:
            idx = low.find(alias)
            if idx >= 0:
                found[token] = min(found.get(token, idx), idx)
                break
    return sorted(found, key=lambda t: found[t])


def subject_team(text: str, sport: str) -> str | None:
    """The team the 'Yes' outcome refers to (first team named), or None."""
    teams = canonical_teams(text, sport)
    return teams[0] if teams else None


def event_key(text: str, sport: str, game_date: date | None) -> str | None:
    """Build an order-independent event key, or None if < 2 teams identified."""
    teams = canonical_teams(text, sport)
    if len(teams) < 2:
        return None
    a, b = sorted(teams[:2])
    day = game_date.isoformat() if game_date else "?"
    return f"{sport}|{day}|{a}+{b}"


def parse_iso_date(value: str | None) -> date | None:
    """Best-effort date extraction from an ISO timestamp string."""
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value)
    if not m:
        return None
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
