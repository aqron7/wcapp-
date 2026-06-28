"""Dixon-Coles bivariate-Poisson scoreline model.

One goals model yields every soccer market consistently: build the full
score-probability matrix P(i,j) from each side's expected goals, then read off
1X2 (winner), totals (i+j), BTTS, and spread (margin i-j) from it. The
Dixon-Coles low-score correction (rho) fixes the Poisson under-count of 0-0,
1-0, 0-1, 1-1 results.

Expected goals come from Elo: the rating gap sets the supremacy while the league
average sets the scoring level. Crude vs a feed of real attack/defence rates,
but a proper joint model — validate via the gate.
"""

from __future__ import annotations

from math import exp, factorial


def _pois(lmbda: float, k: int) -> float:
    return exp(-lmbda) * lmbda ** k / factorial(k)


def _tau(i: int, j: int, la: float, lb: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - la * lb * rho
    if i == 0 and j == 1:
        return 1.0 + la * rho
    if i == 1 and j == 0:
        return 1.0 + lb * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def lambdas_from_elo(elo_a: float, elo_b: float, avg_total: float = 2.75,
                     a_advantage: float = 0.0, exp_div: float = 3.0) -> tuple[float, float]:
    """Expected goals (a, b) from Elo: gap sets supremacy, avg sets the level."""
    d = (elo_a + a_advantage - elo_b) / 400.0
    m = 10 ** (d / exp_div)
    base = avg_total / 2.0
    return base * m, base / m


def score_matrix(la: float, lb: float, rho: float = -0.05, max_goals: int = 10) -> list[list[float]]:
    grid = [[_pois(la, i) * _pois(lb, j) * _tau(i, j, la, lb, rho)
             for j in range(max_goals + 1)] for i in range(max_goals + 1)]
    total = sum(sum(row) for row in grid)
    return [[v / total for v in row] for row in grid]


def outcome_probs(matrix: list[list[float]]) -> tuple[float, float, float]:
    """(P home win, P draw, P away win)."""
    home = draw = away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                home += p
            elif i == j:
                draw += p
            else:
                away += p
    return home, draw, away


def prob_over(matrix: list[list[float]], line: float) -> float:
    return sum(p for i, row in enumerate(matrix) for j, p in enumerate(row) if i + j > line)


def prob_btts(matrix: list[list[float]]) -> float:
    return sum(p for i, row in enumerate(matrix) for j, p in enumerate(row) if i >= 1 and j >= 1)


def prob_margin_over(matrix: list[list[float]], line: float, away: bool = False) -> float:
    """P(team wins by more than ``line``). Home (team A, rows) by default; set
    away=True for team B (cols)."""
    if away:
        return sum(p for i, row in enumerate(matrix) for j, p in enumerate(row) if (j - i) > line)
    return sum(p for i, row in enumerate(matrix) for j, p in enumerate(row) if (i - j) > line)
