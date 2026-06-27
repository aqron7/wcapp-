"""Soccer model: Elo for W/D/L + Monte Carlo bracket simulation.

Phase 4. Single-match probabilities come from Elo (extended to handle draws).
The Monte Carlo simulator rolls the whole tournament forward thousands of times
to price *futures* markets — "team to advance", "team to win the World Cup" —
which are the timely, high-value Kalshi markets during the tournament.
"""

from __future__ import annotations

from ..types import Prediction
from .elo import EloModel


class SoccerModel:
    def __init__(self) -> None:
        self.elo = EloModel(k=20.0, home_advantage=0.0)  # WC is neutral-site
        # TODO(phase4): load World Football Elo style ratings.

    def match_probs(self, home: str, away: str) -> tuple[float, float, float]:
        """Return (home_win, draw, away_win).

        TODO(phase4): convert the 2-outcome Elo into a 3-outcome distribution,
        e.g. estimate a draw probability from the rating gap (closer match =>
        higher draw prob) or move to a Poisson/Dixon-Coles goals model.
        """
        p_home = self.elo.win_prob(home, away, neutral=True)
        draw = 0.26  # placeholder; TODO derive from rating gap
        scale = 1.0 - draw
        return p_home * scale, draw, (1.0 - p_home) * scale

    def simulate_tournament(self, n: int = 50_000) -> list[Prediction]:
        """Monte Carlo the bracket to price advancement / outright markets.

        TODO(phase4):
          - Encode group stage + knockout bracket structure.
          - For each sim: play matches via ``match_probs`` (knockouts resolve
            draws via extra time / penalties ~ coin flip weighted by strength).
          - Tally how often each team advances / wins -> probabilities.
        """
        raise NotImplementedError("phase4: implement Monte Carlo bracket sim")
