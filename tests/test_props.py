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


def test_strikeout_predictions_pure():
    from kalshi_optimizer.models.props import strikeout_predictions
    from kalshi_optimizer.types import MarketQuote

    quotes = [
        MarketQuote("kalshi", "KXMLBKS-G-TBRAS-8", "Drew Rasmussen: 8+ strikeouts?",
                    0.18, 0.20, sport="mlb", event_key="KXMLBKS-G",
                    outcome="TBRAS-8", outcome_label="Drew Rasmussen: 8+",
                    market_type="prop", strike=7.5),
        MarketQuote("kalshi", "KXMLBKS-G-TBRAS-6", "Drew Rasmussen: 6+ strikeouts?",
                    0.45, 0.47, sport="mlb", event_key="KXMLBKS-G",
                    outcome="TBRAS-6", outcome_label="Drew Rasmussen: 6+",
                    market_type="prop", strike=5.5),
    ]
    preds = {p.outcome: p.fair_prob for p in
             strikeout_predictions(quotes, {"drew rasmussen": 7.5})}
    # Unique outcomes (no collision) and 6+ likelier than 8+.
    assert set(preds) == {"TBRAS-8", "TBRAS-6"}
    assert preds["TBRAS-6"] > preds["TBRAS-8"]


def test_outcome_strips_event_prefix():
    from kalshi_optimizer.data import kalshi
    payload = {"markets": [{
        "ticker": "KXMLBKS-26JUN281340AZTB-TBDRASMUSSEN57-10",
        "event_ticker": "KXMLBKS-26JUN281340AZTB", "title": "K?",
        "status": "active", "yes_bid_dollars": "0.05", "yes_ask_dollars": "0.06",
        "yes_sub_title": "Drew Rasmussen: 10+", "floor_strike": 9.5,
    }]}
    q = kalshi.parse_markets(payload, "mlb", "prop")[0]
    assert q.outcome == "TBDRASMUSSEN57-10"   # pitcher kept, not just "10"


def test_batter_season_parse():
    from kalshi_optimizer.data.mlb_stats import parse_batter_season, parse_players
    pl = parse_players({"people": [{"id": 9, "fullName": "Aaron Judge"}]})
    assert pl["aaron judge"] == 9
    s = parse_batter_season({"stats": [{"splits": [{"stat": {
        "homeRuns": 40, "hits": 150, "gamesPlayed": 150}}]}]})
    assert s["hr"] == 40 and s["games"] == 150


def test_hr_and_hits_prop_predictions():
    from kalshi_optimizer.models.props import _prop_predictions
    from kalshi_optimizer.types import MarketQuote

    hr = MarketQuote("kalshi", "KXMLBHR-G-NYJUDGE-1", "Aaron Judge: 1+ HR?",
                     0.2, 0.22, sport="mlb", event_key="KXMLBHR-G", outcome="NYJUDGE-1",
                     outcome_label="Aaron Judge: 1+", market_type="prop", strike=0.5)
    hits = MarketQuote("kalshi", "KXMLBHIT-G-NYJUDGE-2", "Aaron Judge: 2+ hits?",
                       0.3, 0.32, sport="mlb", event_key="KXMLBHIT-G", outcome="NYJUDGE-2",
                       outcome_label="Aaron Judge: 2+", market_type="prop", strike=1.5)
    hr_p = _prop_predictions([hr, hits], "KXMLBHR", {"aaron judge": 0.27}, "hr")
    hit_p = _prop_predictions([hr, hits], "KXMLBHIT", {"aaron judge": 1.0}, "hits")
    assert len(hr_p) == 1 and hr_p[0].outcome == "NYJUDGE-1"
    assert len(hit_p) == 1 and 0 < hit_p[0].fair_prob < 1


def test_strikeout_model():
    s = parse_pitcher_season(STATS)
    ip = expected_innings(s)            # 100/16 = 6.25
    assert 6.0 < ip < 6.5
    lmbda = strikeout_lambda(s["k9"], ip)   # 10.8 * 6.25 / 9 = 7.5
    assert 7.0 < lmbda < 8.0
    assert prob_over(lmbda, 5.5) > prob_over(lmbda, 7.5)
