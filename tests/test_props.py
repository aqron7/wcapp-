"""Tests for the MLB StatsAPI parsing and the strikeout prop model."""

from kalshi_optimizer.data.mlb_stats import parse_pitcher_season, parse_probables
from kalshi_optimizer.models.props import expected_innings, prob_over, strikeout_lambda

SCHEDULE = {"dates": [{"games": [{
    "gamePk": 777,
    "teams": {
        "home": {"team": {"id": 138, "name": "St. Louis Cardinals"},
                 "probablePitcher": {"id": 1, "fullName": "Ace Smith"}},
        "away": {"team": {"id": 146, "name": "Miami Marlins"},
                 "probablePitcher": {"id": 2, "fullName": "Bob Jones"}},
    }}]}]}

STATS = {"stats": [{"splits": [{"stat": {
    "strikeoutsPer9Inn": "10.8", "inningsPitched": "100.0", "gamesStarted": 16}}]}]}


def test_parse_probables():
    probs = parse_probables(SCHEDULE)
    assert len(probs) == 2
    ace = next(p for p in probs if p["pitcher"] == "Ace Smith")
    assert ace["pitcher_id"] == 1 and ace["team"] == "St. Louis Cardinals"


def test_parse_pitcher_season():
    s = parse_pitcher_season(STATS)
    assert s["k9"] == 10.8 and s["gs"] == 16


def test_parse_pitcher_season_empty():
    assert parse_pitcher_season({"stats": [{"splits": []}]}) is None


def test_strikeout_model():
    s = parse_pitcher_season(STATS)
    ip = expected_innings(s)            # 100/16 = 6.25
    assert 6.0 < ip < 6.5
    lmbda = strikeout_lambda(s["k9"], ip)   # 10.8 * 6.25 / 9 = 7.5
    assert 7.0 < lmbda < 8.0
    assert prob_over(lmbda, 5.5) > prob_over(lmbda, 7.5)
