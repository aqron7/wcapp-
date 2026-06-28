"""Kalshi API client.

Kalshi v2 uses RSA key-pair authentication: each request is signed with your
private key and the signature, key ID, and timestamp are sent as headers.

Docs: https://trading-api.readme.io/   (REST + WebSocket)

Design: the HTTP layer (``_get``) is kept separate from the pure parsing layer
(``parse_markets``) so parsing is unit-tested against fixtures without network.

Phase 0 deliverable: ``get_sports_markets`` returns live MarketQuotes.
Phase 6 deliverable: order placement lives in execution/trader.py, not here.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from ..config import Secrets
from ..types import MarketQuote

# Kalshi sports markets live under series tickers. These are sensible defaults;
# confirm/extend against the live GET /series listing for each sport.
# Per sport, the series we pull. Winner markets are modeled; totals/BTTS get a
# Poisson model; spreads/props are ingested for display + tracking. Discover more
# with: python -m kalshi_optimizer discover "<keyword>"
SPORT_SERIES: dict[str, list[str]] = {
    "mlb": ["KXMLBGAME", "KXMLBTOTAL", "KXMLBSPREAD", "KXMLBKS", "KXMLBHR", "KXMLBHIT"],
    "soccer": ["KXWCGAME", "KXWCTOTAL", "KXWCSPREAD", "KXWCBTTS"],
}

# Market type per series ticker (drives which model, if any, applies).
SERIES_TYPE: dict[str, str] = {
    "KXMLBGAME": "winner", "KXMLBTOTAL": "total", "KXMLBSPREAD": "spread",
    "KXMLBKS": "prop", "KXMLBHR": "prop", "KXMLBHIT": "prop",
    "KXWCGAME": "winner", "KXWCTOTAL": "total", "KXWCSPREAD": "spread",
    "KXWCBTTS": "btts",
}


def _price(raw) -> float | None:
    """Kalshi returns prices as decimal-dollar strings ("0.5600"); 0/empty = no
    resting order, so treat those as None (no liquidity)."""
    if raw in (None, ""):
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def parse_markets(payload: dict, sport: str, market_type: str = "winner") -> list[MarketQuote]:
    """Parse a GET /markets response body into MarketQuotes.

    Pure function — no network. Each game has several binary markets (one per
    outcome) sharing an ``event_ticker``; the per-outcome code is the suffix of
    the market ``ticker`` (e.g. ``...AUSEGY-AUS`` -> "AUS", ``-TIE`` = draw).
    """
    quotes: list[MarketQuote] = []
    for m in payload.get("markets", []):
        if m.get("status") not in (None, "active", "open"):
            continue
        ticker = m.get("ticker", "")
        event_ticker = m.get("event_ticker")
        outcome = ticker.rsplit("-", 1)[-1] if "-" in ticker else None
        strike = m.get("floor_strike")
        if strike is None:
            strike = m.get("cap_strike")
        quotes.append(
            MarketQuote(
                platform="kalshi",
                market_id=ticker,
                title=m.get("title", ""),
                yes_bid=_price(m.get("yes_bid_dollars")),
                yes_ask=_price(m.get("yes_ask_dollars")),
                sport=sport,
                event_key=event_ticker,
                outcome=outcome,
                outcome_label=m.get("yes_sub_title"),
                market_type=market_type,
                strike=float(strike) if strike is not None else None,
            )
        )
    return quotes


class KalshiClient:
    def __init__(self, secrets: Secrets):
        from urllib.parse import urlparse

        self.base = secrets.kalshi_api_base.rstrip("/")
        self.base_path = urlparse(self.base).path.rstrip("/")  # e.g. /trade-api/v2
        self.key_id = secrets.kalshi_api_key_id
        self._private_key = self._load_key(secrets.kalshi_private_key_path)
        self._session = requests.Session()

    @staticmethod
    def _load_key(path: str):
        if not path or not Path(path).exists():
            return None
        return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)

    def _sign(self, method: str, full_path: str) -> dict[str, str]:
        """Sign ``timestamp_ms + METHOD + full_path`` (full_path includes
        /trade-api/v2). Kalshi validates against the full request path, so this
        must be the absolute path — not just the route."""
        if self._private_key is None:
            raise RuntimeError("Kalshi private key not loaded; set KALSHI_PRIVATE_KEY_PATH")
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{full_path}".encode()
        signature = self._private_key.sign(
            msg,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": ts,
        }

    def _headers(self, method: str, route: str) -> dict[str, str]:
        return self._sign(method, self.base_path + route)

    def _get(self, route: str, params: dict | None = None) -> dict:
        url = f"{self.base}{route}"
        resp = self._session.get(url, headers=self._headers("GET", route), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def auth_check(self) -> dict:
        """Hit an authenticated endpoint to verify RSA signing actually works."""
        return self._get("/portfolio/balance")

    def _post(self, route: str, body: dict) -> dict:
        url = f"{self.base}{route}"
        headers = self._headers("POST", route)
        headers["Content-Type"] = "application/json"
        resp = self._session.post(url, headers=headers, json=body, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def place_order(self, ticker: str, side: str, count: int, price_cents: int,
                    client_order_id: str) -> dict:
        """Place a limit buy order. side 'yes'/'no'; price in cents (1..99)."""
        body = {
            "ticker": ticker, "action": "buy", "side": side,
            "count": int(count), "type": "limit",
            "client_order_id": client_order_id,
            ("yes_price" if side == "yes" else "no_price"): int(price_cents),
        }
        return self._post("/portfolio/orders", body)

    # ------------------------------------------------------------------ #
    # Phase 0: market data
    # ------------------------------------------------------------------ #
    def get_sports_markets(self, sport: str) -> list[MarketQuote]:
        """Return live open binary market quotes for a sport.

        Pages through GET /markets for each configured series ticker and parses
        with the pure ``parse_markets`` helper.
        """
        out: list[MarketQuote] = []
        for series in SPORT_SERIES.get(sport, []):
            mtype = SERIES_TYPE.get(series, "winner")
            cursor: str | None = None
            while True:
                params = {"series_ticker": series, "status": "open", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                payload = self._get("/markets", params=params)
                out.extend(parse_markets(payload, sport, mtype))
                cursor = payload.get("cursor")
                if not cursor:
                    break
        return out

    # ------------------------------------------------------------------ #
    # WebSocket auth (real-time prices)
    # ------------------------------------------------------------------ #
    WS_PATH = "/trade-api/ws/v2"

    def ws_connect_args(self) -> tuple[str, dict[str, str]]:
        """Return (wss_url, signed_headers) for the real-time feed.

        Derives the ws host from the REST base and signs the ws path with the
        same RSA scheme. NOTE: if live auth is rejected, the signing path
        (WS_PATH) is the most likely thing to adjust per Kalshi's docs.
        """
        from urllib.parse import urlparse

        host = urlparse(self.base).netloc
        url = f"wss://{host}{self.WS_PATH}"
        return url, self._sign("GET", self.WS_PATH)

    def iter_raw_markets(self, sport: str, status: str | None = None):
        """Yield raw market dicts for a sport, paging all series (for logging
        settlements). ``status`` filters server-side, e.g. "settled"."""
        for series in SPORT_SERIES.get(sport, []):
            cursor: str | None = None
            while True:
                params: dict = {"series_ticker": series, "limit": 200}
                if status:
                    params["status"] = status
                if cursor:
                    params["cursor"] = cursor
                payload = self._get("/markets", params=params)
                yield from payload.get("markets", [])
                cursor = payload.get("cursor")
                if not cursor:
                    break

    def get_orderbook(self, ticker: str) -> dict:
        """Raw orderbook for a single market (for arb depth + execution sizing)."""
        return self._get(f"/markets/{ticker}/orderbook")
