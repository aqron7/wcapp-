"""Walk-forward calibration backtest from settled games.

Measures model accuracy NOW (no waiting): replay settled games chronologically,
predicting each game with ratings fit only on the games before it, then update.
That's genuinely out-of-sample. Reports Brier, log-loss, a coin-flip baseline,
and a reliability table (predicted vs actual by probability bucket).

This is calibration, not CLV — it says whether the model's probabilities are
accurate, which is the fastest signal you can get before live closing prices
accumulate.
"""

from __future__ import annotations

from .backtester import brier_score, log_loss


def walk_forward(games: list[tuple[str, str, float]], init: dict[str, float],
                 k: float = 20.0, base: float = 1500.0):
    """Predict each game before updating. Returns (preds, outcomes)."""
    r = dict(init)

    def rt(t: str) -> float:
        return r.get(t, base)

    preds, outs = [], []
    for a, b, score_a in games:
        pa = 1.0 / (1.0 + 10 ** (-(rt(a) - rt(b)) / 400.0))
        preds.append(pa)
        outs.append(score_a)
        r[a] = rt(a) + k * (score_a - pa)
        r[b] = rt(b) + k * ((1.0 - score_a) - (1.0 - pa))
    return preds, outs


def reliability(preds: list[float], outs: list[float], nbins: int = 10) -> list[dict]:
    bins = []
    for i in range(nbins):
        lo, hi = i / nbins, (i + 1) / nbins
        idx = [j for j, p in enumerate(preds)
               if (lo <= p < hi) or (i == nbins - 1 and p >= hi)]
        if idx:
            bins.append({
                "lo": round(lo, 2), "hi": round(hi, 2), "n": len(idx),
                "pred": round(sum(preds[j] for j in idx) / len(idx), 3),
                "actual": round(sum(outs[j] for j in idx) / len(idx), 3),
            })
    return bins


def report(games: list[tuple[str, str, float]], init: dict[str, float], k: float = 20.0) -> dict:
    preds, outs = walk_forward(games, init, k)
    if not preds:
        return {"n": 0}
    decisive = [(p, int(o)) for p, o in zip(preds, outs) if o in (0.0, 1.0)]
    dp = [p for p, _ in decisive]
    do = [o for _, o in decisive]
    return {
        "n": len(preds),
        "brier": round(brier_score(preds, outs), 4),
        "baseline_brier": round(brier_score([0.5] * len(outs), outs), 4),
        "log_loss": round(log_loss(dp, do), 4) if dp else None,
        "reliability": reliability(preds, outs),
    }
