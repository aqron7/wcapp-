"""MLB model: Elo with a starting-pitcher adjustment.

Phase 2 — the simplest model, with daily games for fast feedback. Output is a
moneyline win probability per game, fed into the value engine vs Kalshi.

Teams are referenced by their canonical tokens (see normalize.MLB_TEAMS), so
predictions line up with market ``event_key``s and subject teams.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..types import Prediction
from .elo import EloModel

MODEL_NAME = "baseball_elo_v1"


@dataclass
class GameResult:
    """One historical game for fitting (canonical team tokens)."""

    home: str
    away: str
    home_won: bool


class BaseballModel:
    def __init__(self) -> None:
        # MLB-calibrated-ish defaults: low K (long season), modest home edge.
        self.elo = EloModel(k=4.0, home_advantage=24.0)

    def fit(self, games: list[GameResult]) -> None:
        """Update ratings over a chronological list of historical games."""
        for g in games:
            self.elo.update(g.home, g.away, g.home_won)

    def _win_prob(self, home: str, away: str, home_adj: float, away_adj: float) -> float:
        """Home win probability including pitcher adjustments (Elo points)."""
        diff = (
            self.elo.rating(home) + self.elo.home_advantage + home_adj
            - (self.elo.rating(away) + away_adj)
        )
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def predict_matchup(
        self,
        event_key: str,
        home: str,
        away: str,
        home_pitcher_adj: float = 0.0,
        away_pitcher_adj: float = 0.0,
    ) -> list[Prediction]:
        """Predictions for both teams in a game.

        ``*_pitcher_adj`` are Elo-point deltas for the starting pitcher (e.g.
        derived from projected ERA/FIP vs league average). Default 0 = no info.

        TODO(phase2): wire a starter-quality source to populate the adjustments,
        and a schedule feed to supply correct home/away (titles don't state it).
        """
        p_home = self._win_prob(home, away, home_pitcher_adj, away_pitcher_adj)
        return [
            Prediction(event_key, home, p_home, "mlb", MODEL_NAME),
            Prediction(event_key, away, 1.0 - p_home, "mlb", MODEL_NAME),
        ]
