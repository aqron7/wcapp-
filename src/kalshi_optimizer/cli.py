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


def cmd_raw(config: Config, sport: str, series: str | None = None) -> None:
    """Dump raw JSON for a few Kalshi markets so we can see real field names.

    Optionally pass a specific series ticker, e.g.:
        python -m kalshi_optimizer raw soccer KXWCTOTAL
    """
    import json

    from .data.kalshi import SPORT_SERIES

    kalshi = KalshiClient(config.secrets)
    series = series or SPORT_SERIES.get(sport, [None])[0]
    payload = kalshi._get("/markets", params={"series_ticker": series, "limit": 4})
    console.print_json(json.dumps(payload.get("markets", [])))


def cmd_sample(config: Config, sport: str) -> None:
    """Print the first ~30 live Kalshi markets for a sport (title, prices, key).

    Lets us inspect real market structure before writing model/parse logic.
    """
    kalshi = KalshiClient(config.secrets)
    quotes = kalshi.get_sports_markets(sport)
    liquid = [q for q in quotes if q.yes_bid or q.yes_ask]
    # Show liquid (tradeable) markets first.
    quotes.sort(key=lambda q: 0 if (q.yes_bid or q.yes_ask) else 1)

    console.print(f"[bold]{len(liquid)} of {len(quotes)} {sport} markets have live prices.[/bold]")
    table = Table(title=f"Kalshi {sport} markets (first 30, liquid shown first)")
    table.add_column("title")
    table.add_column("outcome")
    table.add_column("yes_bid", justify="right")
    table.add_column("yes_ask", justify="right")
    for q in quotes[:30]:
        table.add_row(
            q.title[:42],
            q.outcome_label or q.outcome or "?",
            "-" if q.yes_bid is None else f"{q.yes_bid:.2f}",
            "-" if q.yes_ask is None else f"{q.yes_ask:.2f}",
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
    from .engine.value import predictions_with_context

    kalshi = KalshiClient(config.secrets)
    all_ideas: list[TradeIdea] = []

    for sport in config.sports:
        try:
            quotes = kalshi.get_sports_markets(sport)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [yellow]kalshi/{sport} fetch failed:[/yellow] {exc}")
            continue

        if sport not in ("mlb", "soccer"):
            continue
        preds = predictions_with_context(sport, quotes)

        ideas = find_value_edges(quotes, preds, config)
        console.print(f"  {sport}: {len(ideas)} edge(s) from {len(quotes)} markets")
        all_ideas.extend(ideas)

    all_ideas.sort(key=lambda i: i.edge, reverse=True)
    _render_ideas(all_ideas, "Value edges (model vs Kalshi)")


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


def cmd_fit(config: Config, sport: str) -> None:
    """Fit Elo ratings from Kalshi's settled winner markets and save them."""
    from .data.kalshi import KalshiClient
    from .models.baseball import DEFAULT_MLB_RATINGS
    from .models.fit import FIT_K, fit_ratings, games_from_settled, ratings_path, save_ratings
    from .models.soccer import DEFAULT_RATINGS

    winner_series = {"soccer": "KXWCGAME", "mlb": "KXMLBGAME"}.get(sport)
    if not winner_series:
        console.print(f"[yellow]no winner series for {sport}[/yellow]")
        return

    kalshi = KalshiClient(config.secrets)
    raw = list(kalshi.iter_raw_markets(sport, status="settled", series_list=[winner_series]))
    games = games_from_settled(raw, sport)
    if not games:
        console.print(f"[yellow]No settled {sport} games found yet to fit from.[/yellow]")
        return

    init = ({k.lower(): v for k, v in DEFAULT_RATINGS.items()} if sport == "soccer"
            else dict(DEFAULT_MLB_RATINGS))
    ratings = fit_ratings(games, init, k=FIT_K.get(sport, 20.0))
    save_ratings(sport, ratings)
    top = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)[:5]
    console.print(f"Fitted {len(games)} games -> {ratings_path(sport)}")
    console.print("Top: " + ", ".join(f"{k} {v:.0f}" for k, v in top))

    if sport == "mlb":
        from .models.mlb_totals import DIST_PATH, fit_distribution, save_distribution
        settled = list(kalshi.iter_raw_markets("mlb", status="settled", series_list=["KXMLBTOTAL"]))
        dist = fit_distribution(settled)
        if dist:
            save_distribution(dist)
            console.print(f"Fitted run-total distribution ({len(dist)} lines) -> {DIST_PATH}")
        else:
            console.print("[yellow]Not enough settled totals to fit a distribution yet.[/yellow]")


def cmd_analysts(config: Config, sport: str) -> None:
    """Generate + log AI analyst takes for a sport's live games."""
    from . import storage
    from .analysts import generate_takes
    from .data.kalshi import KalshiClient
    from .providers import provider_name

    if not provider_name(config.secrets):
        console.print("[yellow]Set GEMINI_API_KEY (free, aistudio.google.com) to use AI analysts.[/yellow]")
        return
    from .analysts import PERSONAS, build_packets, build_panel_prompt, generate_takes
    from .providers import llm_complete

    quotes = KalshiClient(config.secrets).get_sports_markets(sport)
    packets = build_packets(quotes, sport, horizon_hours=30.0)
    console.print(f"{len(quotes)} markets -> {len(packets)} games on today's slate "
                  f"(engine: {provider_name(config.secrets)})")
    if not packets:
        console.print("[yellow]No live games with liquid, non-decided markets right now.[/yellow]")
        return

    # One probe call so any LLM error is visible (not silently swallowed).
    sample = list(packets.values())[:1]
    try:
        txt = llm_complete(build_panel_prompt({"Quant": PERSONAS["Quant"]}, sample),
                           system="You are a panel of sharp sports betting analysts.",
                           secrets=config.secrets)
        console.print(f"[green]LLM OK[/green] — sample: {txt[:160]}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]LLM call failed:[/red] {exc}")
        return

    conn = storage.connect()
    picks = generate_takes(quotes, sport, config.secrets)
    for p in picks:
        storage.insert_analyst_pick(conn, p)
    plays = len({p["play_id"] for p in picks})
    console.print(f"Logged {plays} analyst plays ({len(picks)} legs) for {sport}.")


def cmd_calibrate(config: Config, sport: str) -> None:
    """Walk-forward calibration of the winner model from settled games."""
    from .backtest.model_backtest import report
    from .data.kalshi import KalshiClient
    from .models.baseball import DEFAULT_MLB_RATINGS
    from .models.fit import FIT_K, games_from_settled
    from .models.soccer import DEFAULT_RATINGS

    winner_series = {"soccer": "KXWCGAME", "mlb": "KXMLBGAME"}.get(sport)
    if not winner_series:
        console.print(f"[yellow]no winner series for {sport}[/yellow]")
        return
    kalshi = KalshiClient(config.secrets)
    raw = list(kalshi.iter_raw_markets(sport, status="settled", series_list=[winner_series]))
    games = games_from_settled(raw, sport)
    init = ({k.lower(): v for k, v in DEFAULT_RATINGS.items()} if sport == "soccer"
            else dict(DEFAULT_MLB_RATINGS))
    r = report(games, init, k=FIT_K.get(sport, 20.0))

    def show(label, m):
        if not m.get("n"):
            console.print(f"[yellow]{label}: no settled markets to calibrate from.[/yellow]")
            return
        verdict = "beats coin flip" if m["brier"] < m["baseline_brier"] else "[red]no better than coin flip[/red]"
        console.print(f"[bold]{label}[/bold]: {m['n']}  Brier {m['brier']} vs baseline "
                      f"{m['baseline_brier']} ({verdict}); log-loss {m['log_loss']}")
        for bk in m["reliability"]:
            console.print(f"  {bk['lo']:.1f}-{bk['hi']:.1f}: pred {bk['pred']:.2f} "
                          f"actual {bk['actual']:.2f}  (n={bk['n']})")

    show(f"{sport} winner", r)
    show(f"{sport} totals", _totals_calibration(kalshi, sport))


def _totals_calibration(kalshi, sport: str) -> dict:
    """Build totals calibration: flat Poisson for MLB, Dixon-Coles for soccer."""
    from .backtest.model_backtest import totals_calibration
    from .models.poisson import AVG_TOTAL, prob_over

    totals_series = {"mlb": "KXMLBTOTAL", "soccer": "KXWCTOTAL"}.get(sport)
    if not totals_series:
        return {"n": 0}
    settled = list(kalshi.iter_raw_markets(sport, status="settled", series_list=[totals_series]))

    if sport == "mlb":
        lam = AVG_TOTAL["mlb"]
        return totals_calibration(settled, lambda line, ev: prob_over(lam, line))

    # soccer: Dixon-Coles per game, teams from settled winner markets
    from .models import dixon_coles as dc
    from .models.soccer import SoccerModel

    winner = list(kalshi.iter_raw_markets("soccer", status="settled", series_list=["KXWCGAME"]))
    labels: dict[str, list[str]] = {}
    for m in winner:
        tk, ev = m.get("ticker", ""), m.get("event_ticker", "")
        code = tk[len(ev) + 1:] if ev and tk.startswith(ev + "-") else None
        if not code or code == "TIE":
            continue
        gk = ev.split("-", 1)[1] if "-" in ev else ev
        labels.setdefault(gk, []).append((m.get("yes_sub_title") or "").split(":", 1)[-1].strip())
    model = SoccerModel()
    matrices = {gk: model.score_matrix(l[0], l[1]) for gk, l in labels.items() if len(l) == 2}

    def fn(line, ev):
        gk = ev.split("-", 1)[1] if "-" in ev else ev
        m = matrices.get(gk)
        return dc.prob_over(m, line) if m else None

    return totals_calibration(settled, fn)


def cmd_auth_check(config: Config) -> None:
    """Verify RSA signing against an authenticated Kalshi endpoint."""
    kalshi = KalshiClient(config.secrets)
    console.print(f"base: [bold]{kalshi.base}[/bold]")
    console.print(f"key id set: {bool(kalshi.key_id)} · private key loaded: {kalshi._private_key is not None}")
    try:
        bal = kalshi.auth_check()
        console.print(f"[green]AUTH OK[/green] — balance endpoint returned: {bal}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]AUTH FAILED[/red] — {exc}")
        console.print("If 401: key/base environment mismatch (demo key vs prod base) "
                      "or wrong signing. Production base is "
                      "https://api.elections.kalshi.com/trade-api/v2")


def cmd_dashboard(config: Config) -> None:
    """Launch the web dashboard (FastAPI + custom frontend)."""
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[yellow]Dashboard needs extra deps:[/yellow] pip install -e \".[dashboard]\"")
        return
    console.print("[bold]Kalshi Edge[/bold] dashboard → http://127.0.0.1:8000  (Ctrl+C to stop)")
    import uvicorn
    uvicorn.run("kalshi_optimizer.web:app", host="127.0.0.1", port=8000, log_level="warning")


def cmd_snapshot(config: Config, loop_minutes: float = 0.0) -> None:
    """Record live prices + model fair values to the DB.

    With ``loop_minutes`` > 0, keep snapshotting on that interval (Ctrl+C to
    stop) so entry/closing prices and results accumulate for CLV.
    """
    import time
    from datetime import datetime
    from .logger import run_snapshot

    def once() -> None:
        try:
            n = run_snapshot(config)
            stamp = datetime.now().strftime("%H:%M:%S")
            console.print(f"[{stamp}] recorded [bold]{n}[/bold] market rows to data/snapshots.db")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]snapshot failed:[/yellow] {exc}")

    if loop_minutes <= 0:
        once()
        return

    console.print(f"Snapshotting every {loop_minutes:g} min. Leave this running "
                  f"(minimize it — it won't steal focus). Ctrl+C to stop.")
    try:
        while True:
            once()
            time.sleep(loop_minutes * 60)
    except KeyboardInterrupt:
        console.print("Stopped.")


def cmd_backtest(config: Config) -> None:
    """Score the model against accumulated snapshot history (the gate)."""
    from .backtest.backtester import (armed_score_from_db, grouped_score_from_db,
                                      realized_from_db, score_from_db)

    result = score_from_db(min_edge=config.edge.min_edge)
    if result.n == 0:
        console.print(
            "[yellow]No scorable markets yet.[/yellow] Run 'snapshot' regularly so "
            "entry prices, closing prices, and settled results can accumulate."
        )
        return
    armed = getattr(config.edge, "tradeable_types", None) or None
    min_clv = getattr(config.edge, "kalshi_fee", 0.01)  # CLV must beat round-trip cost
    table = Table(title="Backtest / validation gate")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("bets scored", str(result.n))
    table.add_row("Brier score", f"{result.brier:.4f}  (lower better, <0.25)")
    table.add_row("log loss", f"{result.log_loss:.4f}")
    table.add_row("mean CLV", f"{result.mean_clv * 100:+.2f}%  (want > 0)")
    table.add_row("PASSES GATE (all)", "✅ yes" if result.passes_gate else "❌ not yet")
    if armed:
        a = armed_score_from_db(armed, min_edge=config.edge.min_edge)
        table.add_row(f"armed types", ", ".join(armed))
        table.add_row("armed mean CLV", f"{a.mean_clv * 100:+.2f}%  (n={a.n})" if a.n else "— (no data)")
        table.add_row("ARMED GATE", "✅ yes" if a.passes_bucket_gate(min_clv=min_clv) else "❌ not yet")
    console.print(table)

    # Where the edge actually lives — split by sport and by market type.
    breakdown = grouped_score_from_db(min_edge=config.edge.min_edge)
    if breakdown:
        bt = Table(title="Breakdown by sport / market type")
        bt.add_column("group")
        bt.add_column("n", justify="right")
        bt.add_column("Brier", justify="right")
        bt.add_column("mean CLV", justify="right")
        bt.add_column("gate", justify="center")
        for label, r in breakdown:
            clv = f"{r.mean_clv * 100:+.2f}%" if r.n else "—"
            gate = "✅" if r.passes_bucket_gate(min_clv=min_clv) else ("·" if r.n < 100 else "❌")
            bt.add_row(label, str(r.n), f"{r.brier:.3f}", clv, gate)
        console.print(bt)
        console.print(f"[dim]gate: ✅ armed (n≥100, CLV>{min_clv*100:.0f}% cost, Brier<0.25)"
                      "  ·  · = too few samples yet[/dim]")

    # Realized return: what flat bets would actually have made at the ask, net of
    # fees — the number that predicts your bankroll, vs the optimistic mid CLV.
    rr = realized_from_db(armed, min_edge=config.edge.min_edge, fee=min_clv)
    if rr.get("n"):
        scope = ("armed (" + ", ".join(armed) + ")") if armed else "all scored markets"
        rt = Table(title=f"Realized return per $1 risked — {scope}")
        rt.add_column("fill")
        rt.add_column("ROI/bet", justify="right")
        rt.add_row("at mid (optimistic)", f"{rr['roi_mid'] * 100:+.2f}%")
        rt.add_row("at ask (real fill)", f"{rr['roi_ask'] * 100:+.2f}%")
        rt.add_row(f"at ask − {min_clv*100:.0f}% fee (net)", f"{rr['roi_net'] * 100:+.2f}%")
        rt.add_row("avg spread paid", f"{rr['avg_spread'] * 100:.2f}%")
        rt.add_row("win rate / n", f"{rr['win_rate']*100:.0f}%  /  {rr['n']}")
        console.print(rt)
        console.print("[dim]Net ROI/bet is the honest expectation; mid-fill flatters it by the spread.[/dim]")
    if not armed:
        console.print("[dim]Tip: set edge.tradeable_types in config.yaml (e.g. [total, spread]) "
                      "to restrict recommendations/execution to validated markets.[/dim]")


def cmd_crypto_scan(config: Config, limit: int) -> None:
    """Fetch crypto news, extract catalysts with Fable, log them as signals."""
    from . import storage
    from .crypto.run import scan_and_log

    if not config.secrets.anthropic_api_key:
        console.print("[yellow]Set ANTHROPIC_API_KEY (Fable) in .env to run the crypto agent.[/yellow]")
        return
    conn = storage.connect()
    try:
        logged = scan_and_log(config.secrets, conn, limit=limit)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]crypto scan failed:[/red] {exc}")
        return
    live = [s for s in logged if s["status"] == "pending"]
    console.print(f"Logged {len(live)} catalyst signals ({len(logged) - len(live)} skipped, no price).")
    if not live:
        console.print("[dim]No qualifying catalysts in the latest news.[/dim]")
        return
    table = Table(title="New crypto catalysts")
    for col in ("asset", "dir", "type", "horiz", "conf", "thesis"):
        table.add_column(col)
    for s in live:
        table.add_row(s["asset"], s["direction"], s["catalyst_type"],
                      f"{s['horizon_hours']}h", f"{s['confidence']:.0%}", s["thesis"][:70])
    console.print(table)


def cmd_crypto_score(config: Config) -> None:
    """Score matured signals and print the catalyst-agent leaderboard (the gate)."""
    from . import storage
    from .crypto.score import leaderboard, score_matured

    conn = storage.connect()
    n = score_matured(conn, config.secrets.coindesk_api_key)
    lb = leaderboard(conn)
    console.print(f"Scored {n} matured signals. {lb['pending']} still pending.")
    if not lb["leaderboard"]:
        console.print("[yellow]No scored signals yet.[/yellow] Run 'crypto-scan' regularly, "
                      "then 'crypto-score' after catalysts' horizons elapse.")
        return
    table = Table(title="Catalyst agent — accuracy & edge by type")
    table.add_column("type")
    table.add_column("n", justify="right")
    table.add_column("hit rate", justify="right")
    table.add_column("mean signed move", justify="right")
    for r in lb["leaderboard"]:
        table.add_row(r["group"], str(r["n"]), f"{r['hit_rate']*100:.0f}%",
                      f"{r['mean_signed_move']*100:+.2f}%")
    console.print(table)
    console.print("[dim]mean signed move > 0 across a real sample is the edge signal — "
                  "hit rate alone can be a coin flip. Validate before trading.[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(prog="kalshi-optimizer")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan-arb", help="cross-platform arbitrage scan (phase 1)")
    sub.add_parser("demo-arb", help="run arb scanner on bundled fixtures (no network)")
    sub.add_parser("find-edges", help="ranked model-vs-Kalshi value bets (phase 2+)")
    sub.add_parser("demo-edges", help="run value engine on bundled fixtures (no network)")
    p_fit = sub.add_parser("fit", help="fit Elo ratings from settled games")
    p_fit.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_cal = sub.add_parser("calibrate", help="walk-forward model calibration from settled games")
    p_cal.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_ana = sub.add_parser("analysts", help="generate AI analyst takes for a sport")
    p_ana.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_cscan = sub.add_parser("crypto-scan", help="Fable crypto catalyst scan -> logged signals")
    p_cscan.add_argument("--limit", type=int, default=30, help="news articles to consider")
    sub.add_parser("crypto-score", help="score matured catalyst signals (the crypto gate)")
    sub.add_parser("auth-check", help="verify Kalshi API auth on a private endpoint")
    sub.add_parser("dashboard", help="launch the web dashboard (phase 5)")
    p_snap = sub.add_parser("snapshot", help="record live prices + fair values to the DB (phase 3)")
    p_snap.add_argument("--loop", type=float, default=0.0, metavar="MIN",
                        help="keep snapshotting every MIN minutes (e.g. --loop 60)")
    sub.add_parser("backtest", help="validation gate report (phase 3)")
    p_discover = sub.add_parser("discover", help="find Kalshi series tickers by keyword")
    p_discover.add_argument("term", help="search term, e.g. 'world cup' or soccer")
    p_sample = sub.add_parser("sample", help="print sample live Kalshi markets for a sport")
    p_sample.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_raw = sub.add_parser("raw", help="dump raw JSON of a few Kalshi markets")
    p_raw.add_argument("sport", help="sport key, e.g. mlb or soccer")
    p_raw.add_argument("series", nargs="?", default=None, help="optional series ticker, e.g. KXWCTOTAL")

    args = parser.parse_args()
    config = Config.load(args.config)

    if args.command == "discover":
        cmd_discover(config, args.term)
        return
    if args.command == "sample":
        cmd_sample(config, args.sport)
        return
    if args.command == "raw":
        cmd_raw(config, args.sport, args.series)
        return
    if args.command == "fit":
        cmd_fit(config, args.sport)
        return
    if args.command == "calibrate":
        cmd_calibrate(config, args.sport)
        return
    if args.command == "analysts":
        cmd_analysts(config, args.sport)
        return

    if args.command == "snapshot":
        cmd_snapshot(config, loop_minutes=args.loop)
        return

    if args.command == "crypto-scan":
        cmd_crypto_scan(config, args.limit)
        return

    if args.command == "crypto-score":
        cmd_crypto_score(config)
        return

    {
        "scan-arb": cmd_scan_arb,
        "demo-arb": cmd_demo_arb,
        "find-edges": cmd_find_edges,
        "demo-edges": cmd_demo_edges,
        "auth-check": cmd_auth_check,
        "dashboard": cmd_dashboard,
        "backtest": cmd_backtest,
    }[args.command](config)


if __name__ == "__main__":
    main()
