"""CoinDesk Data API: crypto news + spot prices.

News is the catalyst source; prices are used to record an entry level when a
signal is logged and to score it once its horizon elapses. Free tier needs an
API key (COINDESK_API_KEY); the news feed also works keyless but rate-limited.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

# CoinDesk Data (formerly CryptoCompare) public data API.
NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/"
PRICE_URL = "https://min-api.cryptocompare.com/data/price"
HISTOHOUR_URL = "https://min-api.cryptocompare.com/data/v2/histohour"


@dataclass
class Article:
    id: str
    published: int          # unix seconds
    title: str
    body: str
    source: str
    url: str
    categories: str         # CoinDesk's pipe-delimited tag string, e.g. "BTC|ETH"


def _auth(api_key: str) -> dict:
    return {"authorization": f"Apikey {api_key}"} if api_key else {}


def _check(r: requests.Response) -> None:
    """Raise a clear error for the common missing/bad-key case."""
    if r.status_code in (401, 403):
        raise RuntimeError("CoinDesk API rejected the request — set a free COINDESK_API_KEY "
                           "in .env (get one at min-api.cryptocompare.com).")
    r.raise_for_status()


def fetch_news(api_key: str = "", lang: str = "EN", limit: int = 30,
               before_ts: int | None = None) -> list[Article]:
    """Crypto news articles, newest first. ``before_ts`` pages back in time."""
    params: dict = {"lang": lang}
    if before_ts:
        params["lTs"] = int(before_ts)   # articles published before this unix ts
    r = requests.get(NEWS_URL, params=params, headers=_auth(api_key), timeout=30)
    _check(r)
    items = r.json().get("Data", [])[:limit]
    out: list[Article] = []
    for a in items:
        out.append(Article(
            id=str(a.get("id", "")),
            published=int(a.get("published_on", 0)),
            title=a.get("title", ""),
            body=(a.get("body", "") or "")[:1200],   # cap; headlines+lede carry the catalyst
            source=a.get("source_info", {}).get("name") or a.get("source", ""),
            url=a.get("url", ""),
            categories=a.get("categories", ""),
        ))
    return out


def spot_price(symbol: str, api_key: str = "", quote: str = "USD") -> float | None:
    """Current price of ``symbol`` (e.g. BTC) in ``quote``. None if unavailable."""
    r = requests.get(PRICE_URL, params={"fsym": symbol.upper(), "tsyms": quote.upper()},
                     headers=_auth(api_key), timeout=30)
    _check(r)
    return r.json().get(quote.upper())


def fetch_hourly(symbol: str, api_key: str = "", quote: str = "USD",
                 hours: int = 2000) -> list[tuple[int, float]]:
    """Hourly (unix_ts, close) history for ``symbol``, oldest first. ~2000h max."""
    r = requests.get(HISTOHOUR_URL, headers=_auth(api_key), timeout=30,
                     params={"fsym": symbol.upper(), "tsym": quote.upper(),
                             "limit": min(2000, hours)})
    _check(r)
    data = r.json().get("Data", {}).get("Data", [])
    return [(int(c["time"]), float(c["close"])) for c in data if c.get("close")]


def price_at(series: list[tuple[int, float]], ts: int) -> float | None:
    """Close of the last hourly candle at or before ``ts`` (series is time-sorted)."""
    picked = None
    for t, close in series:
        if t <= ts:
            picked = close
        else:
            break
    return picked
