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


def fetch_news(api_key: str = "", lang: str = "EN", limit: int = 30) -> list[Article]:
    """Most recent crypto news articles, newest first."""
    r = requests.get(NEWS_URL, params={"lang": lang}, headers=_auth(api_key), timeout=30)
    r.raise_for_status()
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
    r.raise_for_status()
    return r.json().get(quote.upper())
