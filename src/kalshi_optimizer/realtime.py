"""Real-time price feed via Kalshi's WebSocket.

A background task keeps a ``LiveBook`` of current prices fed by the ws ticker
channel; edges are recomputed from the book on demand and pushed to the browser.
Everything degrades to REST if the ws can't connect, so the app keeps working.

Kalshi ws notes (confirm against trading-api.readme.io when testing live):
  - subscribe: {"id":1,"cmd":"subscribe","params":{"channels":["ticker_v2"],
                "market_tickers":[...]}}
  - ticker messages arrive as {"type":"ticker_v2","msg":{"market_ticker":...,
    "yes_bid":<cents>,"yes_ask":<cents>,...}}  (prices in integer cents)
"""

from __future__ import annotations

import asyncio
import json
import time

from .config import Config
from .data.kalshi import KalshiClient
from .engine.value import find_value_edges, predictions_for_sport
from .types import MarketQuote


def _to_prob(cents) -> float | None:
    """Kalshi ws prices are integer cents; 0/None means no resting order."""
    if cents is None:
        return None
    try:
        v = float(cents)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v > 0 else None


class LiveBook:
    """In-memory state: latest quotes by market ticker + model predictions."""

    def __init__(self) -> None:
        self.markets: dict[str, MarketQuote] = {}
        self.preds: list = []
        self.connected = False
        self.last_update: float | None = None
        self.error: str | None = None

    def seed(self, quotes: list[MarketQuote], preds: list) -> None:
        """Replace the market universe + predictions (kept across price ticks)."""
        existing = self.markets
        new: dict[str, MarketQuote] = {}
        for q in quotes:
            prev = existing.get(q.market_id)
            if prev is not None:  # preserve any live price already received
                q.yes_bid, q.yes_ask = prev.yes_bid, prev.yes_ask
            new[q.market_id] = q
        self.markets = new
        self.preds = preds

    def update_price(self, ticker: str, yes_bid: float | None, yes_ask: float | None) -> None:
        q = self.markets.get(ticker)
        if q is None:
            return
        if yes_bid is not None:
            q.yes_bid = yes_bid
        if yes_ask is not None:
            q.yes_ask = yes_ask
        self.last_update = time.time()

    def quotes(self) -> list[MarketQuote]:
        return list(self.markets.values())

    def edges(self, config: Config) -> list:
        return find_value_edges(self.quotes(), self.preds, config)


def handle_message(msg: dict, book: LiveBook) -> None:
    if msg.get("type") in ("ticker", "ticker_v2"):
        m = msg.get("msg", {})
        ticker = m.get("market_ticker") or m.get("ticker")
        if ticker:
            book.update_price(ticker, _to_prob(m.get("yes_bid")), _to_prob(m.get("yes_ask")))


async def _refresh_universe(book: LiveBook, config: Config, client: KalshiClient) -> None:
    """Periodically rebuild the market list + predictions via REST."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            quotes, preds = [], []
            for sport in config.sports:
                qs = await loop.run_in_executor(None, client.get_sports_markets, sport)
                quotes += qs
                preds += predictions_for_sport(sport, qs)
            book.seed(quotes, preds)
            book.error = None
        except Exception as exc:  # noqa: BLE001
            book.error = f"universe: {exc}"
        await asyncio.sleep(600)  # refresh every 10 min to pick up new games


async def _run_ws(book: LiveBook, client: KalshiClient) -> None:
    """Connect, subscribe to ticker updates, and keep the book live (with reconnect)."""
    import websockets

    backoff = 1
    while True:
        # Wait until we know which markets to subscribe to.
        if not book.markets:
            await asyncio.sleep(1)
            continue
        try:
            url, headers = client.ws_connect_args()
            async with websockets.connect(url, extra_headers=headers, ping_interval=10) as ws:
                book.connected = True
                book.error = None
                backoff = 1
                await ws.send(json.dumps({
                    "id": 1, "cmd": "subscribe",
                    "params": {"channels": ["ticker_v2"],
                               "market_tickers": list(book.markets.keys())},
                }))
                async for raw in ws:
                    handle_message(json.loads(raw), book)
        except Exception as exc:  # noqa: BLE001
            book.connected = False
            book.error = f"ws: {exc}"
            await asyncio.sleep(min(backoff, 30))
            backoff *= 2


async def maintain(book: LiveBook, config: Config) -> None:
    """Entry point: keep the universe fresh and the ws feed running."""
    client = KalshiClient(config.secrets)
    asyncio.create_task(_refresh_universe(book, config, client))
    await _run_ws(book, client)
