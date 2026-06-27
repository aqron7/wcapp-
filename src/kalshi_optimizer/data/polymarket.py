"""Polymarket public-data client.

Polymarket exposes free public APIs (Gamma markets API + CLOB) with no key
required for read-only market data. Used as (1) an arbitrage counterparty to
Kalshi and (2) a second probability opinion for model calibration.

Docs: https://docs.polymarket.com/

The Gamma API returns several fields as JSON-encoded strings (``outcomes``,
``outcomePrices``), which this module decodes. As with the Kalshi client,
parsing is a pure function tested against fixtures.
"""

from __future__ import annotations

import json

import requests

from ..normalize import event_key, parse_iso_date
from ..types import MarketQuote

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Polymarket tag/category slugs vary; filter client-side by these keywords too.
SPORT_KEYWORDS: dict[str, list[str]] = {
    "mlb": ["mlb", "baseball"],
    "soccer": ["world cup", "soccer", "fifa"],
}


def _maybe_json_list(value) -> list:
    """Gamma encodes some list fields as JSON strings; decode defensively."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _yes_price(market: dict) -> float | None:
    """Extract the 'Yes' probability from a binary Gamma market.

    Prefers live best bid/ask; falls back to the last ``outcomePrices``.
    """
    bid = market.get("bestBid")
    ask = market.get("bestAsk")
    if bid is not None and ask is not None:
        return (float(bid) + float(ask)) / 2.0
    outcomes = [o.lower() for o in _maybe_json_list(market.get("outcomes"))]
    prices = _maybe_json_list(market.get("outcomePrices"))
    if "yes" in outcomes and len(prices) == len(outcomes):
        return float(prices[outcomes.index("yes")])
    if len(prices) == 2:  # assume [yes, no] ordering
        return float(prices[0])
    return None


def parse_markets(payload: list | dict, sport: str) -> list[MarketQuote]:
    """Parse a Gamma /markets response into MarketQuotes (pure function)."""
    rows = payload if isinstance(payload, list) else payload.get("data", [])
    keywords = SPORT_KEYWORDS.get(sport, [])
    quotes: list[MarketQuote] = []
    for m in rows:
        if m.get("closed") or m.get("active") is False:
            continue
        title = m.get("question") or m.get("title") or ""
        haystack = f"{title} {m.get('slug', '')}".lower()
        if keywords and not any(k in haystack for k in keywords):
            continue
        yes = _yes_price(m)
        bid = m.get("bestBid")
        ask = m.get("bestAsk")
        gday = parse_iso_date(m.get("endDate") or m.get("end_date_iso"))
        quotes.append(
            MarketQuote(
                platform="polymarket",
                market_id=str(m.get("id", m.get("conditionId", ""))),
                title=title,
                yes_bid=float(bid) if bid is not None else yes,
                yes_ask=float(ask) if ask is not None else yes,
                sport=sport,
                event_key=event_key(title, sport, gday),
                close_time=None,
            )
        )
    return quotes


class PolymarketClient:
    def __init__(self) -> None:
        self._session = requests.Session()

    def get_sports_markets(self, sport: str) -> list[MarketQuote]:
        """Return live binary market quotes for a sport from the Gamma API."""
        params = {"closed": "false", "active": "true", "limit": 500}
        resp = self._session.get(f"{GAMMA_BASE}/markets", params=params, timeout=15)
        resp.raise_for_status()
        return parse_markets(resp.json(), sport)
