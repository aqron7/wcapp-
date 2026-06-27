"""Tests for the soccer model and soccer value path (phase 4)."""

import json
from pathlib import Path

from kalshi_optimizer.config import Config
from kalshi_optimizer.data import kalshi
from kalshi_optimizer.engine.value import find_value_edges, soccer_predictions
from kalshi_optimizer.models.soccer import SoccerModel
from kalshi_optimizer.types import Side

FIX = Path(__file__).parent / "fixtures"


def _soccer_quotes():
    return kalshi.parse_markets(json.loads((FIX / "kalshi_soccer.json").read_text()), "soccer")


def test_match_probs_sum_to_one_and_favor_stronger():
    model = SoccerModel()
    p_ger, p_draw, p_cuw = model.match_probs("Germany", "Curacao")
    assert abs(p_ger + p_draw + p_cuw - 1.0) < 1e-9
    assert p_ger > p_cuw
    assert 0 < p_draw < 0.3


def test_draw_peaks_for_even_teams():
    model = SoccerModel()
    _, even_draw, _ = model.match_probs("Brazil", "Brazil")
    _, lop_draw, _ = model.match_probs("Brazil", "Curacao")
    assert even_draw > lop_draw


def test_soccer_predictions_cover_all_three_outcomes():
    preds = soccer_predictions(_soccer_quotes(), SoccerModel())
    outcomes = {p.outcome for p in preds}
    assert outcomes == {"GER", "CUW", "TIE"}
    assert abs(sum(p.fair_prob for p in preds) - 1.0) < 1e-9


def test_soccer_value_edge_found_and_deduped():
    quotes = _soccer_quotes()
    preds = soccer_predictions(quotes, SoccerModel())
    ideas = find_value_edges(quotes, preds, Config())
    # Model rates Curacao stronger than the market implies; Germany at 0.93/0.94
    # is richer than model fair (~0.85) -> one bet per game after dedup.
    assert len(ideas) == 1
    assert ideas[0].edge > 0.03
    assert ideas[0].side in (Side.YES, Side.NO)
