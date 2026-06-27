"""Base Elo rating engine, shared by sport-specific models.

Elo is the pragmatic v1: cheap, well-understood, only needs historical results.
Sport modules subclass / wrap this to add adjustments (home field, starting
pitcher, neutral-site, etc.) and to convert ratings into market probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EloModel:
    k: float = 20.0            # update speed
    home_advantage: float = 65.0   # Elo points added to the home side
    ratings: dict[str, float] = field(default_factory=dict)
    base_rating: float = 1500.0

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.base_rating)

    def win_prob(self, home: str, away: str, neutral: bool = False) -> float:
        """Probability the home team wins (two-outcome; no draws)."""
        ha = 0.0 if neutral else self.home_advantage
        diff = (self.rating(home) + ha) - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home: str, away: str, home_won: bool, neutral: bool = False) -> None:
        """Update ratings after a result (use for fitting on historical games)."""
        p_home = self.win_prob(home, away, neutral)
        outcome = 1.0 if home_won else 0.0
        delta = self.k * (outcome - p_home)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
