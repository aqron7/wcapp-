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
from ..models.soccer import SoccerModel
from ..types import MarketQuote, Prediction, Side, TradeIdea
from .edge import evaluate_market
from .fair_value import blend
from .sizing import stake


def predictions_with_context(sport: str, quotes: list[MarketQuote]) -> list[Prediction]:
    """predictions_for_sport with recent form (from the DB) + venue data loaded."""
    from .. import storage
    from ..form import recent_form
    from ..models.context import load_venues

    conn = storage.connect()
    return predictions_for_sport(sport, quotes, recent_form(conn, sport), load_venues())


def _form_delta(form: dict | None, code: str) -> float:
    from ..models.context import FORM_POINTS_PER_WIN_RATE
    if not form or code not in form:
        return 0.0
    return (form[code] - 0.5) * FORM_POINTS_PER_WIN_RATE


def predictions_for_sport(sport: str, quotes: list[MarketQuote],
                          form: dict | None = None, venues: dict | None = None) -> list[Prediction]:
    """Build model predictions for a sport's quotes (used by find-edges + logger).

    ``form`` maps team code -> recent win rate; ``venues`` maps event_ticker ->
    venue city (for soccer altitude/host). Both optional.
    """
    if sport == "mlb":
        from ..models.baseball import BaseballModel

        model = BaseballModel()
        preds: list[Prediction] = []
        for event_ticker, home, away in matchups_from_quotes(quotes, "mlb"):
            # Reuse the Elo-point adjustment params to inject recent form.
            preds += model.predict_matchup(
                event_ticker, home, away,
                home_pitcher_adj=_form_delta(form, home),
                away_pitcher_adj=_form_delta(form, away),
            )
        return preds
    if sport == "soccer":
        return soccer_predictions(quotes, SoccerModel(), form, venues)
    return []


def _clean_label(label: str | None) -> str:
    """Strip Kalshi's 'Reg Time: ' prefix from soccer outcome labels."""
    if not label:
        return ""
    return label.split(":", 1)[-1].strip() if ":" in label else label.strip()


def soccer_predictions(quotes: list[MarketQuote], model: SoccerModel,
                       form: dict | None = None, venues: dict | None = None) -> list[Prediction]:
    """Build W/D/L predictions keyed by (event_ticker, outcome_code).

    Applies altitude/host (from ``venues``: event_ticker -> city) and recent
    form (from ``form``: team code -> win rate) via the context engine.
    """
    from ..models.context import MatchContext

    events: dict[str, dict] = {}
    for q in quotes:
        if q.platform != "kalshi" or q.sport != "soccer" or not q.event_key or not q.outcome:
            continue
        info = events.setdefault(q.event_key, {"teams": {}})
        if q.outcome != "TIE":
            info["teams"][q.outcome] = _clean_label(q.outcome_label)

    preds: list[Prediction] = []
    for event_ticker, info in events.items():
        teams = list(info["teams"].items())  # [(code, country), ...]
        if len(teams) != 2:
            continue
        (code_a, name_a), (code_b, name_b) = teams[0], teams[1]
        ctx = MatchContext(
            venue_city=(venues or {}).get(event_ticker),
            home_form=(form or {}).get(code_a),
            away_form=(form or {}).get(code_b),
        )
        (p_a, p_draw, p_b), _why = model.contextual_match_probs(name_a, name_b, ctx)
        preds.append(Prediction(event_ticker, code_a, p_a, "soccer", "soccer_elo_v1"))
        preds.append(Prediction(event_ticker, code_b, p_b, "soccer", "soccer_elo_v1"))
        preds.append(Prediction(event_ticker, "TIE", p_draw, "soccer", "soccer_elo_v1"))
    return preds


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
    model_weight: float | None = None,
) -> list[TradeIdea]:
    """Return ranked, sized TradeIdeas where the model beats the Kalshi price.

    Fair value regresses the model toward the **market price as a prior** (the
    market is sharp; an unvalidated model shouldn't be trusted outright). With
    ``model_weight`` w, fair = w*model + (1-w)*market_mid, so only strong
    disagreements survive — this prevents the model's systematic under-confidence
    on favorites from spamming "fade the favorite" picks. ``model_weight``
    defaults to config.edge.model_weight. ``market_probs`` can override the
    market prior per outcome (e.g. a sharper external source).
    """
    from .calibration import devig, sharpen

    if model_weight is None:
        model_weight = getattr(config.edge, "model_weight", 0.5)
    gamma = getattr(config.edge, "model_sharpen", 1.0)
    do_devig = getattr(config.edge, "devig", True)
    pred_index = {(p.event_key, p.outcome): p.fair_prob for p in predictions}
    market_probs = market_probs or {}

    # Per-event de-vig: normalize each game's outcome YES mids to sum to 1 so the
    # overround doesn't make every outcome look overpriced (the favorite-fade bug).
    devigged: dict[tuple[str, str], float] = {}
    if do_devig:
        by_event: dict[str, dict[str, float]] = {}
        for q in quotes:
            if q.platform == "kalshi" and q.event_key and q.outcome and q.yes_mid:
                by_event.setdefault(q.event_key, {})[q.outcome] = q.yes_mid
        for ek, outs in by_event.items():
            if len(outs) < 2:
                continue  # need the full outcome set to remove the overround
            for outcome, prob in devig(outs).items():
                devigged[(ek, outcome)] = prob

    candidates: list[TradeIdea] = []
    for q in quotes:
        if q.platform != "kalshi" or not q.event_key or not q.outcome:
            continue
        if not q.yes_bid or not q.yes_ask:  # 0 / None => no liquidity
            continue
        model_p = pred_index.get((q.event_key, q.outcome))
        if model_p is None:
            continue
        model_p = sharpen(model_p, gamma)  # fix model under-confidence

        # Market prior: explicit source, else de-vigged market, else raw mid.
        market_prior = market_probs.get((q.event_key, q.outcome))
        if market_prior is None:
            market_prior = devigged.get((q.event_key, q.outcome), q.yes_mid)
        fair = blend(model_p, market_prior, model_weight)
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
