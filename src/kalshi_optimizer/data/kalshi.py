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
from ..normalize import event_key, parse_iso_date
from ..types import MarketQuote

# Kalshi sports markets live under series tickers. These are sensible defaults;
# confirm/extend against the live GET /series listing for each sport.
SPORT_SERIES: dict[str, list[str]] = {
    "mlb": ["KXMLBGAME"],
    # World Cup: per-match results (KXWCGAME) drive the model/arb; outright
    # winner (KXMENWORLDCUP) and advancement (KXWCADVANCE) are single-team
    # futures. Discover more with: python -m kalshi_optimizer discover "world cup"
    "soccer": ["KXWCGAME", "KXMENWORLDCUP", "KXWCADVANCE"],
}


def _cents_to_prob(cents: int | float | None) -> float | None:
    """Kalshi prices are integer cents (1..99). Convert to probability [0,1]."""
    if cents is None:
        return None
    return float(cents) / 100.0


def parse_markets(payload: dict, sport: str) -> list[MarketQuote]:
    """Parse a GET /markets response body into MarketQuotes.

    Pure function — no network. ``payload`` is the decoded JSON dict with a
    ``markets`` list. Closed/settled markets and those without an identifiable
    event are skipped.
    """
    quotes: list[MarketQuote] = []
    for m in payload.get("markets", []):
        if m.get("status") not in (None, "active", "open"):
            continue
        title = m.get("title") or m.get("subtitle") or m.get("ticker", "")
        close_time = m.get("close_time")
        gday = parse_iso_date(close_time)
        quotes.append(
            MarketQuote(
                platform="kalshi",
                market_id=m.get("ticker", ""),
                title=title,
                yes_bid=_cents_to_prob(m.get("yes_bid")),
                yes_ask=_cents_to_prob(m.get("yes_ask")),
                sport=sport,
                event_key=event_key(title, sport, gday),
                close_time=None,
            )
        )
    return quotes


class KalshiClient:
    def __init__(self, secrets: Secrets):
        self.base = secrets.kalshi_api_base.rstrip("/")
        self.key_id = secrets.kalshi_api_key_id
        self._private_key = self._load_key(secrets.kalshi_private_key_path)
        self._session = requests.Session()

    @staticmethod
    def _load_key(path: str):
        if not path or not Path(path).exists():
            return None
        return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)

    def _headers(self, method: str, route: str) -> dict[str, str]:
        """Build signed auth headers.

        Kalshi signs the string: ``timestamp_ms + METHOD + route_path``.
        """
        if self._private_key is None:
            raise RuntimeError("Kalshi private key not loaded; set KALSHI_PRIVATE_KEY_PATH")
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{route}".encode()
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

    def _get(self, route: str, params: dict | None = None) -> dict:
        url = f"{self.base}{route}"
        resp = self._session.get(url, headers=self._headers("GET", route), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

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
            cursor: str | None = None
            while True:
                params = {"series_ticker": series, "status": "open", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                payload = self._get("/markets", params=params)
                out.extend(parse_markets(payload, sport))
                cursor = payload.get("cursor")
                if not cursor:
                    break
        return out

    def get_orderbook(self, ticker: str) -> dict:
        """Raw orderbook for a single market (for arb depth + execution sizing)."""
        return self._get(f"/markets/{ticker}/orderbook")
