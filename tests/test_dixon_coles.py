"""Tests for the Dixon-Coles scoreline model."""

from kalshi_optimizer.models import dixon_coles as dc
from kalshi_optimizer.models.soccer import SoccerModel


def test_matrix_normalized_and_outcomes_sum_to_one():
    m = dc.score_matrix(1.6, 1.1)
    total = sum(sum(r) for r in m)
    assert abs(total - 1.0) < 1e-9
    h, d, a = dc.outcome_probs(m)
    assert abs(h + d + a - 1.0) < 1e-9
    assert h > a  # higher lambda favored


def test_totals_and_btts_and_margin_consistent():
    m = dc.score_matrix(1.6, 1.1)
    assert dc.prob_over(m, 1.5) > dc.prob_over(m, 3.5)
    assert 0 < dc.prob_btts(m) < 1
    assert dc.prob_margin_over(m, 0.5) > dc.prob_margin_over(m, 2.5)


def test_lambdas_favor_stronger_side():
    la, lb = dc.lambdas_from_elo(1900, 1500)
    assert la > lb


def test_soccer_model_uses_dc_and_draw_peaks_for_even_teams():
    model = SoccerModel()
    _, even_draw, _ = model.match_probs("Brazil", "Brazil")
    _, lop_draw, _ = model.match_probs("Brazil", "Curacao")
    assert even_draw > lop_draw
    h, d, a = model.match_probs("Germany", "Curacao")
    assert abs(h + d + a - 1.0) < 1e-9 and h > a
