"""Orchestration: fetch news -> Fable catalysts -> log with entry price."""

from __future__ import annotations

from .. import storage
from .catalysts import extract_catalysts
from .news import fetch_news, spot_price


def scan_and_log(secrets, conn, limit: int = 30) -> list[dict]:
    """Pull recent news, extract catalysts with Fable, stamp each with the
    asset's current price, and log it as a pending signal. Returns logged rows."""
    api_key = getattr(secrets, "coindesk_api_key", "")
    articles = fetch_news(api_key, limit=limit)
    signals = extract_catalysts(articles, secrets)
    price_cache: dict[str, float | None] = {}
    logged = []
    for s in signals:
        asset = s["asset"]
        if asset not in price_cache:
            try:
                price_cache[asset] = spot_price(asset, api_key)
            except Exception:  # noqa: BLE001 - unknown ticker / API hiccup
                price_cache[asset] = None
        s["entry_price"] = price_cache[asset]
        s["status"] = "pending" if s["entry_price"] else "skipped"
        storage.insert_crypto_signal(conn, s)
        logged.append(s)
    return logged
