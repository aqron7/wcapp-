"""Keyless free data for the crypto agent.

News comes from public crypto RSS feeds; prices from CoinGecko's keyless API.
No account or API key is required. (Kept the ``api_key`` params so callers don't
change, but they're ignored.)
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import requests

RSS_FEEDS = [
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Cointelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
]
COINGECKO = "https://api.coingecko.com/api/v3"

# Ticker -> CoinGecko id for the coins catalysts usually concern. Unknown
# tickers just get skipped (no price -> signal not scored).
COIN_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
    "BNB": "binancecoin", "ADA": "cardano", "DOGE": "dogecoin", "AVAX": "avalanche-2",
    "DOT": "polkadot", "MATIC": "matic-network", "POL": "polygon-ecosystem-token",
    "LINK": "chainlink", "TRX": "tron", "LTC": "litecoin", "BCH": "bitcoin-cash",
    "UNI": "uniswap", "ATOM": "cosmos", "XLM": "stellar", "NEAR": "near",
    "APT": "aptos", "ARB": "arbitrum", "OP": "optimism", "SUI": "sui",
    "SHIB": "shiba-inu", "PEPE": "pepe", "TON": "the-open-network", "ICP": "internet-computer",
    "FIL": "filecoin", "HBAR": "hedera-hashgraph", "INJ": "injective-protocol",
    "AAVE": "aave", "MKR": "maker", "RNDR": "render-token", "TIA": "celestia",
    "SEI": "sei-network", "IMX": "immutable-x", "GRT": "the-graph", "ALGO": "algorand",
}


@dataclass
class Article:
    id: str
    published: int          # unix seconds
    title: str
    body: str
    source: str
    url: str
    categories: str


def _text(item: ET.Element, tag: str) -> str:
    el = item.find(tag)
    return (el.text or "").strip() if el is not None and el.text else ""


def _pub_ts(item: ET.Element) -> int:
    raw = _text(item, "pubDate") or _text(item, "{http://purl.org/dc/elements/1.1/}date")
    if not raw:
        return 0
    try:
        return int(parsedate_to_datetime(raw).timestamp())
    except (TypeError, ValueError):
        return 0


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s).strip()


def fetch_news(api_key: str = "", lang: str = "EN", limit: int = 30,
               before_ts: int | None = None) -> list[Article]:
    """Recent crypto news merged from public RSS feeds, newest first."""
    out: list[Article] = []
    for source, url in RSS_FEEDS:
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "kalshi-edge/1.0"})
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except (requests.RequestException, ET.ParseError):
            continue
        for item in root.iter("item"):
            cats = "|".join(c.text for c in item.findall("category") if c.text)
            out.append(Article(
                id=_text(item, "guid") or _text(item, "link"),
                published=_pub_ts(item),
                title=_text(item, "title"),
                body=_strip_html(_text(item, "description"))[:1200],
                source=source,
                url=_text(item, "link"),
                categories=cats,
            ))
    if before_ts:
        out = [a for a in out if a.published and a.published < before_ts]
    out.sort(key=lambda a: a.published, reverse=True)
    return out[:limit]


def _coin_id(symbol: str) -> str | None:
    return COIN_IDS.get(symbol.upper())


def spot_price(symbol: str, api_key: str = "", quote: str = "USD") -> float | None:
    """Current price of ``symbol`` in ``quote`` via CoinGecko. None if unknown."""
    cid = _coin_id(symbol)
    if not cid:
        return None
    r = requests.get(f"{COINGECKO}/simple/price", timeout=30,
                     params={"ids": cid, "vs_currencies": quote.lower()})
    r.raise_for_status()
    return r.json().get(cid, {}).get(quote.lower())


def fetch_hourly(symbol: str, api_key: str = "", quote: str = "USD",
                 hours: int = 336) -> list[tuple[int, float]]:
    """Hourly (unix_ts, price) history via CoinGecko, oldest first.

    CoinGecko returns ~hourly granularity for 2-90 day ranges."""
    cid = _coin_id(symbol)
    if not cid:
        return []
    days = max(1, min(90, round(hours / 24)))
    r = requests.get(f"{COINGECKO}/coins/{cid}/market_chart", timeout=30,
                     params={"vs_currency": quote.lower(), "days": days})
    r.raise_for_status()
    prices = r.json().get("prices", [])
    return [(int(ms / 1000), float(p)) for ms, p in prices if p]


def price_at(series: list[tuple[int, float]], ts: int) -> float | None:
    """Close of the last hourly candle at or before ``ts`` (series is time-sorted)."""
    picked = None
    for t, close in series:
        if t <= ts:
            picked = close
        else:
            break
    return picked
