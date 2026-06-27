"""Tests for the MLB Elo model and the value-edge engine (phase 2)."""

import json
from pathlib import Path

from kalshi_optimizer.config import Config
from kalshi_optimizer.data import kalshi
from kalshi_optimizer.engine.edge import evaluate_market
from kalshi_optimizer.engine.value import find_value_edges, matchups_from_quotes
from kalshi_optimizer.models.baseball import BaseballModel, GameResult
from kalshi_optimizer.types import Side

FIX = Path(__file__).parent / "fixtures"


def _quotes():
    return kalshi.parse_markets(json.loads((FIX / "kalshi_markets.json").read_text()), "mlb")


def test_evaluate_market_prefers_yes_when_underpriced():
    side, entry, edge = evaluate_market(fair_prob=0.73, yes_bid=0.54, yes_ask=0.56, fee=0.01)
    assert side is Side.YES
    assert entry == 0.56
    assert round(edge, 2) == 0.16


def test_evaluate_market_prefers_no_when_overpriced():
    side, entry, edge = evaluate_market(fair_prob=0.30, yes_bid=0.54, yes_ask=0.56, fee=0.0)
    assert side is Side.NO
    assert entry == 0.46  # 1 - yes_bid
    assert round(edge, 2) == 0.24  # 0.54 - 0.30


def test_elo_fit_moves_ratings_toward_winner():
    model = BaseballModel()
    games = [GameResult("NYY", "BOS", home_won=True)] * 10
    model.fit(games)
    assert model.elo.rating("NYY") > model.elo.rating("BOS")


def test_matchups_dedup_and_subject_is_home():
    matchups = matchups_from_quotes(_quotes(), "mlb")
    keys = [m[0] for m in matchups]
    assert len(keys) == len(set(keys))  # unique events
    nyy = next(m for m in matchups if "NYY" in m)
    assert nyy[1] == "NYY"  # subject (Yankees) treated as home


def test_find_value_edges_ranks_and_sizes():
    quotes = _quotes()
    model = BaseballModel()
    model.elo.ratings.update({"NYY": 1600, "BOS": 1450, "LAD": 1550, "SF": 1500})
    preds = []
    for ek, home, away in matchups_from_quotes(quotes, "mlb"):
        preds += model.predict_matchup(ek, home, away)

    config = Config()  # defaults: min_edge 0.03, fee 0.01, quarter-Kelly
    ideas = find_value_edges(quotes, preds, config)

    # Yankees are a strong +EV YES at 0.56; Dodgers game is roughly fair -> no bet.
    assert len(ideas) == 1
    idea = ideas[0]
    assert "Yankees" in idea.title
    assert idea.side is Side.YES
    assert idea.edge > 0.03
    assert idea.stake > 0
    assert idea.stake <= config.sizing.max_per_market
