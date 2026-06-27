# Kalshi Sports Odds Optimizer — Plan

A system to find and (eventually) auto-execute the most profitable trades on
Kalshi sports markets (World Cup soccer, MLB, and other sports) by modeling
true outcome probabilities ourselves and exploiting mispricings.

## How Kalshi works (the foundation)

Kalshi contracts are **binary**: they trade 1¢–99¢ and settle at $1 (Yes wins)
or $0 (No wins). A 60¢ price = the market's implied 60% probability. So
"finding the best trade" reduces to: estimate the *true* probability, compare to
Kalshi's price, and bet when there's a positive-expected-value gap.

```
EV(buy Yes) = true_prob * (1.00 - price) - (1 - true_prob) * price
            = true_prob - price          (per $1 contract, ignoring fees)
```

## Strategy

Two profit sources running together:

1. **Value betting (A-flavored, model-driven):** Our own models produce a fair
   probability for each outcome. Where our fair value beats Kalshi's price by
   more than fees + a safety margin, that's a +EV bet. Because we are **not
   paying for a sharp odds feed**, the model carries the weight and must be
   validated by backtesting before it is trusted with money.

2. **Arbitrage (C):** Compare the *same* event across Kalshi and free price
   sources (primarily **Polymarket**, plus Betfair/free odds where available).
   When prices are inconsistent, lock in risk-free profit. Needs no model —
   built first to validate the data pipeline.

### Fair value, without a paid feed

Fair value = our model, **calibrated and cross-checked against free market
prices**:
- **Polymarket** — free public API, lists sports markets; serves double duty as
  an arb counterparty and a second probability opinion.
- **The Odds API free tier** (~500 req/mo) / Betfair Exchange — free market
  prices to sanity-check and calibrate the model.

## Models (do-it-yourself core)

Start simple, upgrade later. v1 models use only free historical data.

| Sport | v1 model | v2 upgrade |
|---|---|---|
| Soccer / World Cup | Elo (W/D/L) + **Monte Carlo bracket sim** for advancement & outright-winner markets | Dixon-Coles / Poisson attack-defense |
| MLB | Elo with starting-pitcher adjustment → moneyline | Log5 / run-expectancy with park & bullpen factors |

The Monte Carlo bracket simulator is what unlocks World Cup futures
(advance/win) rather than just single matches.

## Validation gate (required before real money)

- **Backtester** measures Brier score, log-loss, and **closing-line value (CLV)**.
- **Gate:** no real or automated trades until the model shows **positive CLV**
  and a paper-trading period confirms the edge live.

## Position sizing & risk

- **Fractional Kelly** (e.g. half- or quarter-Kelly) for stake sizing.
- Bankroll cap, per-market exposure cap, total exposure cap.
- Kill-switch and dry-run mode mandatory before auto-execution.

## Architecture

```
Data layer                Model + fair value         Decision engine
──────────                ──────────────────         ───────────────
Kalshi API            →   Elo / Poisson models   →   EV edge (model vs Kalshi)
Polymarket API (free) →   Monte Carlo bracket sim     Kelly sizing (fractional)
The Odds API (free)   →   calibration vs market       Arb scanner (Kalshi⇄Polymarket)
historical results    →                               risk caps / kill-switch
        │                                                     │
        └──────────── Backtester (CLV, Brier, log-loss) ─────┘
                                  │
                  Output: ranked alerts → (later) auto-execute via Kalshi orders
```

## Build order

0. **Spike** — Kalshi auth + pull live MLB/World Cup markets; pull Polymarket
   equivalents; print side by side.
1. **Arb scanner (C)** — no model needed; price comparison across
   Kalshi/Polymarket. Ships a useful tool in days and proves the pipes.
2. **MLB Elo model** — simplest model, daily games = fast feedback. EV vs Kalshi.
3. **Backtester** — Brier + CLV. **Gate before real money.**
4. **Soccer Elo + Monte Carlo bracket** — unlock live World Cup futures.
5. **Sizing + alerting** — fractional Kelly, exposure caps, push notifications.
6. **Auto-execution** — Kalshi order API behind kill-switch + exposure limits;
   dry-run first.

## Tech stack

- **Python** backend (best ecosystem for the odds math).
- **pandas / numpy** for calculations; **APScheduler** for polling.
- **SQLite** (→ Postgres later) to log market snapshots for backtesting.
- **Streamlit** dashboard first; FastAPI + React later if real-time/mobile needed.

## Caveats

- Kalshi is CFTC-regulated, US-only, requires KYC.
- Profit is entirely determined by whether our model beats Kalshi's price —
  the software is the easy 80%, the edge is the hard 20%.
- Auto-execution risks real money on bugs; guardrails are not optional.
