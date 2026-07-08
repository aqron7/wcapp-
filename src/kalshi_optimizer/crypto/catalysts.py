"""Fable-driven catalyst extraction.

Given a batch of news articles, Fable decides which are *medium-horizon*
catalysts (hours-to-days, where reasoning beats the millisecond headline bots),
and for each returns the affected asset, direction, magnitude, horizon, a
confidence, and a one-line thesis. Pure prompt/parse/dedup functions here are
unit-tested; the network call is a thin wrapper.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from ..providers import fable_complete
from .news import Article

DIRECTIONS = {"up", "down", "neutral"}
MAGNITUDES = {"small", "medium", "large"}
CATALYST_TYPES = {"unlock", "listing", "delisting", "governance", "regulatory",
                  "partnership", "upgrade", "hack", "macro", "other"}


def build_prompt(articles: list[Article]) -> str:
    blocks = []
    for a in articles:
        cats = f" [{a.categories}]" if a.categories else ""
        blocks.append(f"ID {a.id}{cats} ({a.source}): {a.title}\n{a.body[:400]}")
    return (
        "You are a crypto markets analyst. Below are recent news articles. Identify "
        "only the ones that are genuine MEDIUM-HORIZON catalysts — events likely to "
        "move a specific asset's price over the next few hours to a few days through "
        "reasoning the market under-reacts to (token unlocks, exchange listings, "
        "governance outcomes, protocol upgrades, regulatory actions, major "
        "partnerships, hacks). SKIP price recaps, generic commentary, and instant "
        "headlines already fully priced in.\n\n"
        + "\n\n".join(blocks) + "\n\n"
        "For each real catalyst, output an object: "
        '{"article_id":"<id>","asset":"<ticker e.g. BTC>","direction":"up|down|neutral",'
        '"magnitude":"small|medium|large","horizon_hours":<int>,"confidence":0.0-1.0,'
        '"catalyst_type":"unlock|listing|delisting|governance|regulatory|partnership|'
        'upgrade|hack|macro|other","thesis":"one sentence on the mechanism"}.\n'
        "Reply ONLY with a JSON array of these objects. Use article_id values exactly "
        "as given. If nothing qualifies, reply []."
    )


def parse_signals(text: str, valid_ids: set[str]) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except ValueError:
        return []
    out: list[dict] = []
    for d in arr if isinstance(arr, list) else []:
        if not isinstance(d, dict):
            continue
        aid = str(d.get("article_id", ""))
        asset = str(d.get("asset", "")).upper().strip()
        if aid not in valid_ids or not asset:
            continue
        direction = str(d.get("direction", "neutral")).lower()
        if direction not in DIRECTIONS:
            direction = "neutral"
        magnitude = str(d.get("magnitude", "small")).lower()
        if magnitude not in MAGNITUDES:
            magnitude = "small"
        ctype = str(d.get("catalyst_type", "other")).lower()
        if ctype not in CATALYST_TYPES:
            ctype = "other"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        try:
            horizon = max(1, int(d.get("horizon_hours", 24)))
        except (TypeError, ValueError):
            horizon = 24
        out.append({"article_id": aid, "asset": asset, "direction": direction,
                    "magnitude": magnitude, "horizon_hours": horizon,
                    "confidence": conf, "catalyst_type": ctype,
                    "thesis": str(d.get("thesis", ""))[:300]})
    return out


def dedup(signals: list[dict]) -> list[dict]:
    """Collapse duplicate coverage: one signal per (asset, catalyst_type),
    keeping the highest-confidence version."""
    best: dict[tuple[str, str], dict] = {}
    for s in signals:
        key = (s["asset"], s["catalyst_type"])
        if key not in best or s["confidence"] > best[key]["confidence"]:
            best[key] = s
    return sorted(best.values(), key=lambda s: s["confidence"], reverse=True)


def extract_catalysts(articles: list[Article], secrets) -> list[dict]:
    """One Fable call over the batch -> deduped catalyst signals."""
    if not articles:
        return []
    valid = {a.id for a in articles}
    by_id = {a.id: a for a in articles}
    text = fable_complete(build_prompt(articles),
                          system="You are a sharp, skeptical crypto markets analyst.",
                          secrets=secrets)
    ts = datetime.now(timezone.utc).isoformat()
    out = []
    for s in dedup(parse_signals(text, valid)):
        art = by_id[s["article_id"]]
        s.update({"ts": ts, "title": art.title, "url": art.url,
                  "source": art.source, "published": art.published})
        out.append(s)
    return out
