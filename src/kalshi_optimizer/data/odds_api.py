"""The Odds API client (free tier).

Used only to *calibrate / sanity-check* our own models, not as the primary
truth (we deliberately avoid paying for a sharp feed). Free tier ~500 req/mo,
so cache aggressively.

Docs: https://the-odds-api.com/
"""

from __future__ import annotations

import requests

BASE = "https://api.the-odds-api.com/v4"

# Map our sport keys to The Odds API sport keys.
SPORT_KEYS = {
    "mlb": "baseball_mlb",
    "soccer": "soccer_fifa_world_cup",
}


class OddsApiClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._session = requests.Session()

    def get_consensus_probs(self, sport: str) -> dict[str, dict[str, float]]:
        """Return de-vigged consensus probabilities per event/outcome.

        TODO(phase2/3):
          - GET /sports/{key}/odds with regions=us,eu&markets=h2h.
          - De-vig each book (multiplicative method), then average across books.
          - Return {event_key: {outcome: prob}} for calibration.
        Returns {} when no api_key is configured (model runs standalone).
        """
        if not self.api_key:
            return {}
        raise NotImplementedError("phase2: implement odds consensus + de-vig")


def devig_multiplicative(implied: dict[str, float]) -> dict[str, float]:
    """Remove bookmaker vig by normalizing implied probabilities to sum to 1.

    ``implied`` maps outcome -> raw implied prob (1/decimal_odds). The naive
    multiplicative method just divides by the overround.
    """
    overround = sum(implied.values())
    if overround <= 0:
        return implied
    return {k: v / overround for k, v in implied.items()}
