# Kalshi Edge — project notes for Claude

Kalshi sports value-betting + tracking tool. See `PLAN.md` for the full plan and
`README.md` for usage. Python package under `src/kalshi_optimizer/`.

## Key facts
- Kalshi market-data endpoints are **public**; auth (RSA signing) is only
  validated on private endpoints (orders, portfolio) and the WebSocket. Sign the
  **full request path** including `/trade-api/v2`.
- Use the **production** base for real keys:
  `https://api.elections.kalshi.com/trade-api/v2`.
- Prices come from Kalshi as decimal-dollar strings (`yes_bid_dollars`); a
  game's outcome is the market ticker suffix, grouped by `event_ticker`.
- Fair value is regressed toward the market price (`edge.model_weight`) because
  the market is sharp and the models are unvalidated. The backtester/ledger
  **CLV gate** must be positive before trusting the model / arming execution.
- Tests run with `PYTHONPATH=src python -m pytest -q` (editable install is
  flaky with the system `cryptography`).

## Writing style (user preference)
- Do **not** overuse the word "genuinely" (and similar filler intensifiers like
  "truly", "really"). Keep prose plain and specific.
