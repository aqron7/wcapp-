"""Value-edge engine (phase 2): model probabilities vs Kalshi prices.

Pipeline per market:
  model prob  --(blend with optional free market prob)-->  fair prob
  fair prob + bid/ask  --(evaluate_market)-->  side, entry price, edge
  edge >= threshold  --(fractional Kelly + caps)-->  sized TradeIdea

Pure function over already-fetched quotes and predictions, so it is fully
unit-tested offline.
"""

from __future__ import annotations

from ..config import Config
from ..normalize import canonical_teams, subject_team
from ..types import MarketQuote, Prediction, Side, TradeIdea
from .edge import evaluate_market
from .fair_value import blend
from .sizing import stake


def matchups_from_quotes(
    quotes: list[MarketQuote], sport: str
) -> list[tuple[str, str, str]]:
    """Derive unique (event_key, home, away) matchups from market titles.

    NOTE: titles don't state home/away, so the subject team is treated as home.
    TODO(phase2): join a schedule feed for correct home/away assignment.
    """
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for q in quotes:
        if q.event_key is None or q.sport != sport or q.event_key in seen:
            continue
        teams = canonical_teams(q.title, sport)
        if len(teams) < 2:
            continue
        seen.add(q.event_key)
        out.append((q.event_key, teams[0], teams[1]))
    return out


def find_value_edges(
    quotes: list[MarketQuote],
    predictions: list[Prediction],
    config: Config,
    market_probs: dict[tuple[str, str], float] | None = None,
    model_weight: float = 0.6,
) -> list[TradeIdea]:
    """Return ranked, sized TradeIdeas where the model beats the Kalshi price.

    ``market_probs`` optionally maps (event_key, team_token) -> a free market
    probability (Polymarket / Odds API) to blend into fair value.
    """
    pred_index = {(p.event_key, p.outcome): p.fair_prob for p in predictions}
    market_probs = market_probs or {}

    ideas: list[TradeIdea] = []
    running_exposure = 0.0
    for q in quotes:
        if q.platform != "kalshi" or q.event_key is None:
            continue
        if q.yes_bid is None or q.yes_ask is None or not q.sport:
            continue
        subj = subject_team(q.title, q.sport)
        if subj is None:
            continue
        model_p = pred_index.get((q.event_key, subj))
        if model_p is None:
            continue

        fair = blend(model_p, market_probs.get((q.event_key, subj)), model_weight)
        side, entry, edge = evaluate_market(fair, q.yes_bid, q.yes_ask, config.edge.kalshi_fee)
        if edge < config.edge.min_edge:
            continue

        win_prob = fair if side is Side.YES else 1.0 - fair
        sized = stake(win_prob, entry, config.bankroll, config.sizing, running_exposure)
        if sized <= 0:
            continue
        running_exposure += sized

        ideas.append(
            TradeIdea(
                market_id=q.market_id,
                platform="kalshi",
                title=q.title,
                side=side,
                price=entry,
                fair_prob=round(fair, 4),
                edge=round(edge, 4),
                stake=sized,
                kind="value",
                rationale=(
                    f"model {model_p:.0%} -> fair {fair:.0%} vs "
                    f"{side.value} @{entry:.2f} (edge {edge * 100:.1f}%)"
                ),
            )
        )

    ideas.sort(key=lambda i: i.edge, reverse=True)
    return ideas
