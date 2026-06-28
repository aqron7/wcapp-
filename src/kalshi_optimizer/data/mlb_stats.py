"""MLB StatsAPI client (free, no key) for player-prop modeling.

statsapi.mlb.com gives probable pitchers and season stats, which is enough to
project pitcher strikeouts. Parsing is split from the HTTP calls so it is
unit-tested against fixtures.

Docs (community): https://github.com/toddrob99/MLB-StatsAPI/wiki
"""

from __future__ import annotations

import requests

BASE = "https://statsapi.mlb.com/api/v1"


def parse_probables(payload: dict) -> list[dict]:
    """Probable pitchers from a schedule response (hydrate=probablePitcher)."""
    out: list[dict] = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            teams = game.get("teams", {})
            for side in ("home", "away"):
                pp = teams.get(side, {}).get("probablePitcher")
                team = teams.get(side, {}).get("team", {})
                if pp:
                    out.append({
                        "game_pk": game.get("gamePk"),
                        "side": side,
                        "team": team.get("name"),
                        "team_id": team.get("id"),
                        "pitcher_id": pp.get("id"),
                        "pitcher": pp.get("fullName"),
                    })
    return out


def parse_pitcher_season(payload: dict) -> dict | None:
    """Season K/9, innings, starts from a people/{id}/stats response."""
    stats = payload.get("stats", [])
    if not stats or not stats[0].get("splits"):
        return None
    st = stats[0]["splits"][0].get("stat", {})

    def _f(key):
        try:
            return float(st.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {"k9": _f("strikeoutsPer9Inn"), "ip": _f("inningsPitched"),
            "gs": int(_f("gamesStarted"))}


class MlbStatsClient:
    def __init__(self) -> None:
        self._session = requests.Session()

    def probable_pitchers(self, date: str) -> list[dict]:
        """Probable pitchers for a date ("YYYY-MM-DD")."""
        r = self._session.get(f"{BASE}/schedule",
                              params={"sportId": 1, "date": date, "hydrate": "probablePitcher"},
                              timeout=15)
        r.raise_for_status()
        return parse_probables(r.json())

    def pitcher_season(self, pitcher_id: int) -> dict | None:
        r = self._session.get(f"{BASE}/people/{pitcher_id}/stats",
                              params={"stats": "season", "group": "pitching"}, timeout=15)
        r.raise_for_status()
        return parse_pitcher_season(r.json())
