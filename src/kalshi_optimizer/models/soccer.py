"""Soccer model: Elo ratings -> win/draw/loss probabilities (phase 4).

World Cup is a neutral-site tournament, so by default there is no home
advantage. The three-way (W/D/L) split is derived from the Elo rating gap: the
draw probability peaks when teams are evenly matched and shrinks as the gap
widens.

Ratings are seeded with approximate World-Football-Elo-style values for the 2026
field. These are a starting point, NOT ground truth — the backtester gate
(phase 3) must confirm the model beats the closing line before betting real
money. Replace/extend ``DEFAULT_RATINGS`` or load from a feed to improve.
"""

from __future__ import annotations

# Approximate national-team strength (Elo points). Keys are lowercased to match
# Kalshi's yes_sub_title country names. Unknown teams fall back to ``base``.
DEFAULT_RATINGS: dict[str, float] = {
    "argentina": 2100, "france": 2050, "spain": 2050, "england": 2000,
    "brazil": 2000, "portugal": 1980, "netherlands": 1950, "germany": 1930,
    "belgium": 1900, "italy": 1880, "uruguay": 1880, "croatia": 1870,
    "colombia": 1850, "morocco": 1840, "switzerland": 1820, "denmark": 1820,
    "japan": 1810, "usa": 1790, "austria": 1790, "mexico": 1800,
    "senegal": 1800, "czechia": 1780, "ecuador": 1780, "norway": 1780,
    "ukraine": 1770, "nigeria": 1760, "iran": 1750, "south korea": 1750,
    "sweden": 1740, "canada": 1730, "peru": 1720, "ivory coast": 1720,
    "algeria": 1720, "australia": 1720, "paraguay": 1720, "egypt": 1700,
    "ghana": 1700, "bosnia and herzegovina": 1700, "congo dr": 1680,
    "south africa": 1660, "costa rica": 1650, "saudi arabia": 1650,
    "uzbekistan": 1650, "jamaica": 1620, "cape verde": 1620, "jordan": 1600,
    "iraq": 1600, "panama": 1650, "honduras": 1600, "qatar": 1600,
    "curacao": 1550, "new zealand": 1500,
}


class SoccerModel:
    def __init__(self, ratings: dict[str, float] | None = None, max_draw: float = 0.28):
        self.ratings = {k.lower(): v for k, v in (ratings or DEFAULT_RATINGS).items()}
        self.base = 1500.0
        self.max_draw = max_draw

    def rating_for(self, name: str) -> float:
        return self.ratings.get(name.strip().lower(), self.base)

    def score_matrix(self, team_a: str, team_b: str, a_advantage: float = 0.0):
        """Dixon-Coles score-probability matrix for the match."""
        from . import dixon_coles as dc

        la, lb = dc.lambdas_from_elo(self.rating_for(team_a), self.rating_for(team_b),
                                     a_advantage=a_advantage)
        return dc.score_matrix(la, lb)

    def match_probs(self, team_a: str, team_b: str, a_advantage: float = 0.0) -> tuple[float, float, float]:
        """(P(team_a win), P(draw), P(team_b win)) from the Dixon-Coles matrix."""
        from . import dixon_coles as dc

        return dc.outcome_probs(self.score_matrix(team_a, team_b, a_advantage))

    def contextual_match_probs(self, team_a: str, team_b: str, ctx):
        """match_probs with altitude/host/form adjustments; returns (probs, why)."""
        from .context import adjust

        adj_a, adj_b, breakdown = adjust(team_a, team_b, ctx)
        probs = self.match_probs(team_a, team_b, a_advantage=adj_a - adj_b)
        return probs, breakdown
