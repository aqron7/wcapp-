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
from ..types import MarketQuote, Prediction, Side, TradeIdea
from .edge import evaluate_market
from .fair_value import blend
from .sizing import stake


def matchups_from_quotes(
    quotes: list[MarketQuote], sport: str
) -> list[tuple[str, str, str]]:
    """Derive unique (event_ticker, home, away) matchups from Kalshi markets.

    Teams are the non-draw outcome codes sharing an ``event_ticker``. Home/away
    is inferred from the title ("away vs home"): the team whose label appears
    later in the title is treated as home.

    TODO: confirm home/away against a schedule feed; title order is a heuristic.
    """
    events: dict[str, dict] = {}
    for q in quotes:
        if q.platform != "kalshi" or not q.event_key or q.sport != sport:
            continue
        info = events.setdefault(q.event_key, {"title": q.title or "", "teams": {}})
        if q.outcome and q.outcome != "TIE":
            info["teams"][q.outcome] = q.outcome_label or ""

    out: list[tuple[str, str, str]] = []
    for event_ticker, info in events.items():
        teams = list(info["teams"].items())  # [(code, label), ...]
        if len(teams) != 2:
            continue
        title = info["title"].lower()

        def title_pos(label: str) -> int:
            name = label.split(":")[-1].strip().lower()
            idx = title.find(name) if name else -1
            return idx if idx >= 0 else 9999

        teams.sort(key=lambda t: title_pos(t[1]))
        away, home = teams[0][0], teams[1][0]
        out.append((event_ticker, home, away))
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

    candidates: list[TradeIdea] = []
    for q in quotes:
        if q.platform != "kalshi" or not q.event_key or not q.outcome:
            continue
        if not q.yes_bid or not q.yes_ask:  # 0 / None => no liquidity
            continue
        model_p = pred_index.get((q.event_key, q.outcome))
        if model_p is None:
            continue

        fair = blend(model_p, market_probs.get((q.event_key, q.outcome)), model_weight)
        side, entry, edge = evaluate_market(fair, q.yes_bid, q.yes_ask, config.edge.kalshi_fee)
        if edge < config.edge.min_edge:
            continue

        candidates.append(
            TradeIdea(
                market_id=q.market_id,
                platform="kalshi",
                title=f"{q.title} [{q.outcome_label or q.outcome}]",
                side=side,
                price=entry,
                fair_prob=round(fair, 4),
                edge=round(edge, 4),
                stake=0.0,  # filled after dedup + sizing
                kind="value",
                rationale=(
                    f"model {model_p:.0%} -> fair {fair:.0%} vs "
                    f"{side.value} @{entry:.2f} (edge {edge * 100:.1f}%)"
                ),
            )
        )

    # One bet per game (buying YES on one outcome == NO on the other), best edge.
    candidates.sort(key=lambda i: i.edge, reverse=True)
    seen_events: set[str] = set()
    ideas: list[TradeIdea] = []
    running_exposure = 0.0
    for idea in candidates:
        event = idea.market_id.rsplit("-", 1)[0]
        if event in seen_events:
            continue
        seen_events.add(event)
        win_prob = idea.fair_prob if idea.side is Side.YES else 1.0 - idea.fair_prob
        sized = stake(win_prob, idea.price, config.bankroll, config.sizing, running_exposure)
        if sized <= 0:
            continue
        running_exposure += sized
        idea.stake = sized
        ideas.append(idea)
    return ideas
