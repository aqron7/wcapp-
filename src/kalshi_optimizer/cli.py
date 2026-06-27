"""Command-line entry point.

Subcommands map to the phased build order in PLAN.md:
  scan-arb   (phase 1)  cross-platform arbitrage scanner
  find-edges (phase 2+) model vs Kalshi value bets, ranked
  backtest   (phase 3)  validation gate report
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from .config import Config
from .data.kalshi import KalshiClient
from .data.polymarket import PolymarketClient
from .engine.arbitrage import scan_cross_platform
from .types import MarketQuote

console = Console()


def cmd_scan_arb(config: Config) -> None:
    """Pull Kalshi + Polymarket quotes and report cross-platform arbs."""
    kalshi = KalshiClient(config.secrets)
    poly = PolymarketClient()

    quotes: list[MarketQuote] = []
    for sport in config.sports:
        for name, client in (("kalshi", kalshi), ("polymarket", poly)):
            try:
                fetched = client.get_sports_markets(sport)
                quotes.extend(fetched)
                console.print(f"  {name}/{sport}: {len(fetched)} markets")
            except Exception as exc:  # noqa: BLE001 - report and keep going
                console.print(f"  [yellow]{name}/{sport} fetch failed:[/yellow] {exc}")

    arbs = scan_cross_platform(
        quotes, min_profit=config.arbitrage_min_profit, fee=config.edge.kalshi_fee
    )
    if not arbs:
        console.print("[dim]No arbitrage opportunities found.[/dim]")
        return

    table = Table(title="Cross-platform arbitrage")
    table.add_column("Event")
    table.add_column("Profit %", justify="right")
    table.add_column("Trade")
    for arb in arbs:
        table.add_row(arb.event_key, f"{arb.guaranteed_profit * 100:.1f}", arb.description)
    console.print(table)


def cmd_demo_arb(config: Config) -> None:
    """Run the arb scanner against bundled fixtures (no network/keys needed)."""
    import json
    from pathlib import Path

    from .data import kalshi as kalshi_data
    from .data import polymarket as poly_data

    fix = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    quotes = kalshi_data.parse_markets(json.loads((fix / "kalshi_markets.json").read_text()), "mlb")
    quotes += poly_data.parse_markets(
        json.loads((fix / "polymarket_markets.json").read_text()), "mlb"
    )
    arbs = scan_cross_platform(quotes, min_profit=config.arbitrage_min_profit, fee=0.0)

    table = Table(title="Cross-platform arbitrage (demo fixtures)")
    table.add_column("Event")
    table.add_column("Profit %", justify="right")
    table.add_column("Trade")
    for arb in arbs:
        table.add_row(arb.event_key, f"{arb.guaranteed_profit * 100:.1f}", arb.description)
    console.print(table)


def cmd_find_edges(config: Config) -> None:
    console.print("[bold]Find value edges[/bold] (phase 2+) — not yet implemented.")
    # TODO(phase2): run models -> fair_value.blend -> edge.best_side -> sizing.stake, rank.


def cmd_backtest(config: Config) -> None:
    console.print("[bold]Backtest[/bold] (phase 3) — not yet implemented.")
    # TODO(phase3): load history, run backtest.run_backtest, print pass/fail gate.


def main() -> None:
    parser = argparse.ArgumentParser(prog="kalshi-optimizer")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan-arb", help="cross-platform arbitrage scan (phase 1)")
    sub.add_parser("demo-arb", help="run arb scanner on bundled fixtures (no network)")
    sub.add_parser("find-edges", help="ranked model-vs-Kalshi value bets (phase 2+)")
    sub.add_parser("backtest", help="validation gate report (phase 3)")

    args = parser.parse_args()
    config = Config.load(args.config)

    {
        "scan-arb": cmd_scan_arb,
        "demo-arb": cmd_demo_arb,
        "find-edges": cmd_find_edges,
        "backtest": cmd_backtest,
    }[args.command](config)


if __name__ == "__main__":
    main()
