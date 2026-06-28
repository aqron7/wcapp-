"""Player-prop models. First up: pitcher strikeouts (Poisson).

A starter's strikeouts over a game ~ Poisson(lambda), with lambda from season
K/9 scaled to expected innings:

    lambda = K9 * expected_innings / 9

P(over line) follows as the Poisson survival above floor(line). HR/hit/goal
props will follow the same shape once their batter/scorer rates are wired.
"""

from __future__ import annotations

from math import exp, factorial, floor

DEFAULT_START_INNINGS = 5.5


def strikeout_lambda(k_per9: float, expected_innings: float = DEFAULT_START_INNINGS) -> float:
    return max(0.1, k_per9 * expected_innings / 9.0)


def prob_over(lmbda: float, line: float) -> float:
    """P(count > line) for a Poisson(lmbda)."""
    k = int(floor(line))
    cdf = sum(exp(-lmbda) * lmbda ** i / factorial(i) for i in range(0, k + 1))
    return max(0.0, min(1.0, 1.0 - cdf))


def expected_innings(season: dict, default: float = DEFAULT_START_INNINGS) -> float:
    """Average innings per start from season stats, clamped to a sane range."""
    gs = season.get("gs") or 0
    ip = season.get("ip") or 0.0
    if gs <= 0 or ip <= 0:
        return default
    return max(3.0, min(7.0, ip / gs))
