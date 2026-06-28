"""Configuration & secrets loading.

Settings come from ``config.yaml``; secrets come from environment / ``.env``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.25
    max_per_market: float = 50.0
    max_total_exposure: float = 500.0


@dataclass
class EdgeConfig:
    min_edge: float = 0.03
    kalshi_fee: float = 0.01
    model_weight: float = 0.5   # fair = w*model + (1-w)*market_prior; lower = trust market more
    devig: bool = True          # remove the market's overround before comparing
    model_sharpen: float = 1.25  # >1 fixes model under-confidence; fit from results later
    min_price: float = 0.05     # skip near-decided markets (<=5c or >=95c) — no 0-chance picks


@dataclass
class ExecutionConfig:
    mode: str = "dry_run"          # dry_run | paper | live
    kill_switch: bool = True


@dataclass
class Secrets:
    kalshi_api_key_id: str = ""
    kalshi_private_key_path: str = ""
    kalshi_api_base: str = "https://demo-api.kalshi.co/trade-api/v2"
    odds_api_key: str = ""
    gemini_api_key: str = ""        # free tier: aistudio.google.com
    gemini_model: str = ""          # optional override, e.g. gemini-2.5-flash
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    @classmethod
    def from_env(cls) -> "Secrets":
        return cls(
            kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID", ""),
            kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            kalshi_api_base=os.getenv(
                "KALSHI_API_BASE", "https://demo-api.kalshi.co/trade-api/v2"
            ),
            odds_api_key=os.getenv("ODDS_API_KEY", ""),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )


@dataclass
class Config:
    sports: list[str] = field(default_factory=lambda: ["mlb", "soccer"])
    bankroll: float = 1000.0
    sizing: SizingConfig = field(default_factory=SizingConfig)
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    arbitrage_min_profit: float = 0.01
    polling_interval_seconds: int = 60
    secrets: Secrets = field(default_factory=Secrets.from_env)

    @classmethod
    def load(cls, path: str | Path = "config.yaml") -> "Config":
        """Load config.yaml, falling back to defaults if absent."""
        data: dict[str, Any] = {}
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}

        return cls(
            sports=data.get("sports", ["mlb", "soccer"]),
            bankroll=data.get("bankroll", 1000.0),
            sizing=SizingConfig(**(data.get("sizing") or {})),
            edge=EdgeConfig(**(data.get("edge") or {})),
            execution=ExecutionConfig(**(data.get("execution") or {})),
            arbitrage_min_profit=(data.get("arbitrage") or {}).get("min_profit", 0.01),
            polling_interval_seconds=(data.get("polling") or {}).get("interval_seconds", 60),
            secrets=Secrets.from_env(),
        )
