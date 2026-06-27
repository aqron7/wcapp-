"""MLB model: Elo with a starting-pitcher adjustment.

Phase 2 — the simplest model, with daily games for fast feedback. Output is a
moneyline win probability per game, fed into the edge engine vs Kalshi.
"""

from __future__ import annotations

from ..types import Prediction
from .elo import EloModel


class BaseballModel:
    def __init__(self) -> None:
        self.elo = EloModel(k=4.0, home_advantage=24.0)  # MLB-ish defaults
        # TODO(phase2): load fitted ratings from historical seasons.

    def predict_game(
        self,
        event_key: str,
        home: str,
        away: str,
        home_pitcher: str | None = None,
        away_pitcher: str | None = None,
    ) -> list[Prediction]:
        """Return win-probability predictions for both teams.

        TODO(phase2):
          - Adjust team Elo by starting-pitcher quality (e.g. projected ERA/FIP
            mapped to an Elo delta).
          - Optionally blend in bullpen / recent form.
        """
        p_home = self.elo.win_prob(home, away)
        # TODO: apply pitcher adjustment to p_home here.
        return [
            Prediction(event_key, home, p_home, "mlb", "baseball_elo_v1"),
            Prediction(event_key, away, 1.0 - p_home, "mlb", "baseball_elo_v1"),
        ]
