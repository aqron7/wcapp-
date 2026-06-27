"""Command-line entry point.

Subcommands map to the phased build order in PLAN.md:
  scan-arb   (phase 1)  cross-platform arbitrage scanner
  find-edges (phase 2+) model vs Kalshi value bets, ranked
  backtest   (phase 3)  validation gate report
"""

from __future__ import annotations

import argparse

from rich.console import Console

from .config import Config

console = Console()


def cmd_scan_arb(config: Config) -> None:
    console.print("[bold]Arbitrage scan[/bold] (phase 1) — not yet implemented.")
    # TODO(phase1): pull Kalshi + Polymarket quotes, run engine.arbitrage.scan_cross_platform.


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
    sub.add_parser("find-edges", help="ranked model-vs-Kalshi value bets (phase 2+)")
    sub.add_parser("backtest", help="validation gate report (phase 3)")

    args = parser.parse_args()
    config = Config.load(args.config)

    {
        "scan-arb": cmd_scan_arb,
        "find-edges": cmd_find_edges,
        "backtest": cmd_backtest,
    }[args.command](config)


if __name__ == "__main__":
    main()
