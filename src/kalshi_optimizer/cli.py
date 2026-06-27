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
from .engine.value import find_value_edges, matchups_from_quotes
from .models.baseball import BaseballModel
from .types import MarketQuote, Prediction, TradeIdea

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


def cmd_raw(config: Config, sport: str) -> None:
    """Dump raw JSON for a few Kalshi markets so we can see real field names."""
    import json

    from .data.kalshi import SPORT_SERIES

    kalshi = KalshiClient(config.secrets)
    series = SPORT_SERIES.get(sport, [None])[0]
    payload = kalshi._get("/markets", params={"series_ticker": series, "limit": 4})
    console.print_json(json.dumps(payload.get("markets", [])))


def cmd_sample(config: Config, sport: str) -> None:
    """Print the first ~30 live Kalshi markets for a sport (title, prices, key).

    Lets us inspect real market structure before writing model/parse logic.
    """
    kalshi = KalshiClient(config.secrets)
    quotes = kalshi.get_sports_markets(sport)
    table = Table(title=f"Kalshi {sport} markets (showing up to 30 of {len(quotes)})")
    table.add_column("title")
    table.add_column("yes_bid", justify="right")
    table.add_column("yes_ask", justify="right")
    table.add_column("event_key")
    for q in quotes[:30]:
        table.add_row(
            q.title[:55],
            "-" if q.yes_bid is None else f"{q.yes_bid:.2f}",
            "-" if q.yes_ask is None else f"{q.yes_ask:.2f}",
            q.event_key or "[dim]none[/dim]",
        )
    console.print(table)


def cmd_discover(config: Config, term: str) -> None:
    """List Kalshi series whose ticker/title matches ``term`` (case-insensitive).

    Use this to find the real series_ticker for a sport, e.g.:
        python -m kalshi_optimizer discover "world cup"
        python -m kalshi_optimizer discover soccer
    """
    kalshi = KalshiClient(config.secrets)
    term_low = term.lower()
    table = Table(title=f"Kalshi series matching '{term}'")
    table.add_column("series_ticker")
    table.add_column("title")
    table.add_column("category")

    cursor: str | None = None
    matched = 0
    while True:
        params: dict = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        try:
            payload = kalshi._get("/series", params=params)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]/series fetch failed:[/yellow] {exc}")
            return
        series = payload.get("series", [])
        for s in series:
            hay = f"{s.get('ticker', '')} {s.get('title', '')} {s.get('category', '')}".lower()
            if term_low in hay:
                table.add_row(s.get("ticker", ""), s.get("title", ""), s.get("category", ""))
                matched += 1
        cursor = payload.get("cursor")
        if not cursor or not series:
            break

    if matched:
        console.print(table)
    else:
        console.print(f"[dim]No series matched '{term}'.[/dim]")


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


def _render_ideas(ideas: list[TradeIdea], title: str) -> None:
    if not ideas:
        console.print("[dim]No value edges above threshold.[/dim]")
        return
    table = Table(title=title)
    for col in ("Market", "Side", "Price", "Fair", "Edge %", "Stake $"):
        table.add_column(col, justify="right" if col not in ("Market", "Side") else "left")
    for i in ideas:
        table.add_row(
            i.title[:48],
            i.side.value,
            f"{i.price:.2f}",
            f"{i.fair_prob:.2f}",
            f"{i.edge * 100:.1f}",
            f"{i.stake:.2f}",
        )
    console.print(table)


def _mlb_predictions(model: BaseballModel, quotes: list[MarketQuote]) -> list[Prediction]:
    preds: list[Prediction] = []
    for event_key, home, away in matchups_from_quotes(quotes, "mlb"):
        preds.extend(model.predict_matchup(event_key, home, away))
    return preds


def cmd_find_edges(config: Config) -> None:
    """Live: model probabilities vs Kalshi prices -> ranked value bets."""
    kalshi = KalshiClient(config.secrets)
    model = BaseballModel()
    # TODO(phase2): model.fit(load_historical_games()) — flat ratings until then.

    quotes: list[MarketQuote] = []
    for sport in config.sports:
        if sport != "mlb":
            continue  # only the MLB model exists so far (soccer is phase 4)
        try:
            quotes.extend(kalshi.get_sports_markets("mlb"))
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]kalshi/mlb fetch failed:[/yellow] {exc}")

    ideas = find_value_edges(quotes, _mlb_predictions(model, quotes), config)
    _render_ideas(ideas, "MLB value edges (model vs Kalshi)")


def cmd_demo_edges(config: Config) -> None:
    """Offline: run the value engine on bundled fixtures with seeded ratings."""
    import json
    from pathlib import Path

    from .data import kalshi as kalshi_data

    fix = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
    quotes = kalshi_data.parse_markets(json.loads((fix / "kalshi_markets.json").read_text()), "mlb")

    model = BaseballModel()
    model.elo.ratings.update({"NYY": 1600, "BOS": 1450, "LAD": 1550, "SF": 1500})

    ideas = find_value_edges(quotes, _mlb_predictions(model, quotes), config)
    _render_ideas(ideas, "MLB value edges (demo fixtures)")


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
    sub.add_parser("demo-edges", help="run value engine on bundled fixtures (no network)")
    sub.add_parser("backtest", help="validation gate report (phase 3)")
    p_discover = sub.add_parser("discover", help="find Kalshi series tickers by keyword")
    p_discover.add_argument("term", help="search term, e.g. 'world cup' or soccer")
    p_sample = sub.add_parser("sample", help="print sample live Kalshi markets for a sport")
    p_sample.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_raw = sub.add_parser("raw", help="dump raw JSON of a few Kalshi markets")
    p_raw.add_argument("sport", help="sport key, e.g. mlb or soccer")

    args = parser.parse_args()
    config = Config.load(args.config)

    if args.command == "discover":
        cmd_discover(config, args.term)
        return
    if args.command == "sample":
        cmd_sample(config, args.sport)
        return
    if args.command == "raw":
        cmd_raw(config, args.sport)
        return

    {
        "scan-arb": cmd_scan_arb,
        "demo-arb": cmd_demo_arb,
        "find-edges": cmd_find_edges,
        "demo-edges": cmd_demo_edges,
        "backtest": cmd_backtest,
    }[args.command](config)


if __name__ == "__main__":
    main()
