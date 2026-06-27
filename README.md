# Kalshi Sports Odds Optimizer

Finds (and eventually auto-executes) the most profitable trades on Kalshi sports
markets — World Cup soccer, MLB, and more — by modeling true outcome
probabilities and exploiting mispricings vs Kalshi prices and free market
sources.

See [`PLAN.md`](./PLAN.md) for the full strategy, architecture, and build order.

## Status

Scaffold. Modules are stubs with documented interfaces and `TODO`s following the
phased build order in `PLAN.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env            # add your Kalshi API credentials
cp config.example.yaml config.yaml

python -m kalshi_optimizer --help

# See the pipeline work end-to-end on bundled fixtures (no keys/network):
python -m kalshi_optimizer demo-arb      # cross-platform arbitrage
python -m kalshi_optimizer demo-edges    # model-vs-Kalshi value bets

# Live (needs Kalshi key + open network):
python -m kalshi_optimizer --config config.yaml scan-arb     # arbitrage
python -m kalshi_optimizer --config config.yaml find-edges   # MLB value bets
```

> Note: `scan-arb` reaches Kalshi (`api.elections.kalshi.com`) and Polymarket
> (`gamma-api.polymarket.com`). It runs locally where you have your key and an
> open network; in a restricted/sandboxed network it will report each fetch
> failure and continue. Use `demo-arb` to exercise the full parse → match →
> scan pipeline offline.

## Project layout

```
src/kalshi_optimizer/
├── config.py           # settings & secrets loading
├── data/               # API clients
│   ├── kalshi.py       #   Kalshi REST/WS (RSA-key auth)
│   ├── polymarket.py   #   Polymarket free public API
│   └── odds_api.py     #   The Odds API (free tier, calibration)
├── models/             # do-it-yourself probability models
│   ├── elo.py          #   base Elo engine
│   ├── baseball.py     #   MLB Elo + starting-pitcher adjustment
│   └── soccer.py       #   soccer Elo + Monte Carlo bracket sim
├── engine/             # decision logic
│   ├── fair_value.py   #   blend model + market consensus
│   ├── edge.py         #   EV calculation
│   ├── sizing.py       #   fractional Kelly + exposure caps
│   └── arbitrage.py    #   cross-platform arb scanner
├── backtest/           # validation gate
│   └── backtester.py   #   Brier, log-loss, closing-line value
├── execution/          # order placement (last phase)
│   └── trader.py       #   Kalshi orders + kill-switch + dry-run
└── cli.py              # entry point
```

## Safety

No real or automated trades until the backtester shows **positive closing-line
value** and a paper-trading period confirms the edge. Auto-execution runs behind
a kill-switch, dry-run mode, and per-market / total exposure caps.

## Disclaimer

For research and educational use. Trading event contracts carries risk of loss.
You are responsible for compliance with Kalshi's terms and applicable law.
