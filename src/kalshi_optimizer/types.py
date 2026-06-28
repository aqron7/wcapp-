"""Shared domain types used across data, model, and engine layers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    YES = "yes"
    NO = "no"


@dataclass
class MarketQuote:
    """A single binary market's current pricing on some platform.

    Prices are expressed as probabilities in [0, 1] (i.e. cents / 100).
    """

    platform: str            # "kalshi" | "polymarket" | ...
    market_id: str
    title: str
    yes_bid: float | None
    yes_ask: float | None
    sport: str | None = None
    event_key: str | None = None   # groups markets of the same game (Kalshi: event_ticker)
    outcome: str | None = None     # outcome code, e.g. "SEA", "AUS", "TIE"
    outcome_label: str | None = None   # human label, e.g. "Seattle", "Reg Time: Tie"
    market_type: str | None = None     # "winner" | "total" | "spread" | "btts" | "prop"
    close_time: datetime | None = None

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2


@dataclass
class Prediction:
    """Model output: our fair probability for an outcome."""

    event_key: str
    outcome: str
    fair_prob: float         # in [0, 1]
    sport: str
    model: str               # which model produced it


@dataclass
class TradeIdea:
    """A ranked, sized recommendation produced by the decision engine."""

    market_id: str
    platform: str
    title: str
    side: Side
    price: float             # entry price (prob)
    fair_prob: float
    edge: float              # EV gap after fees
    stake: float             # USD, post-Kelly + caps
    kind: str                # "value" | "arbitrage"
    rationale: str = ""
