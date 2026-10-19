"""Split-conformal thresholds: P(s >= tau_alpha) <= alpha under exchangeability."""

from __future__ import annotations

import math

import numpy as np


def conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(s)
    rank = math.ceil((n + 1) * (1.0 - alpha))
    return float(s[rank - 1]) if 0 < rank <= n else float("inf")


def min_alpha(n: int) -> float:
    return 1.0 / (n + 1)


def calibrate(cal_scores: np.ndarray, alphas, subsample: int = 1) -> dict[float, float]:
    s = np.asarray(cal_scores, dtype=np.float64)
    s = s[np.isfinite(s)][::max(1, subsample)]
    return {float(a): conformal_threshold(s, float(a)) for a in alphas}


def coverage_report(normal_scores: np.ndarray, thresholds: dict[float, float]) -> dict:
    """Realized false-alarm rate on held-out normal data (label-free)."""
    s = np.asarray(normal_scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = len(s)
    rows = {}
    for alpha, thr in sorted(thresholds.items()):
        realized = float((s >= thr).mean()) if n else float("nan")
        se = math.sqrt(alpha * (1 - alpha) / n) if n else float("nan")
        rows[float(alpha)] = {"nominal": alpha, "realized_fpr": realized, "gap": realized - alpha,
                              "covered": bool(realized <= alpha + 2 * se)}
    return rows


def repeated_coverage(scores: np.ndarray, alphas, n_repeats: int = 20, block: int = 500,
                      seed: int = 0, subsample: int = 1) -> dict:
    """Coverage averaged over random block-wise halves of one normal stream."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    rng = np.random.default_rng(seed)
    block_id = np.arange(len(s)) // max(1, block)
    n_blocks = int(block_id.max()) + 1

    per_alpha: dict[float, list[float]] = {float(a): [] for a in alphas}
    for _ in range(n_repeats):
        left = (rng.random(n_blocks) < 0.5)[block_id]
        if left.all() or (~left).all():
            continue
        for a, tau in calibrate(s[left], alphas, subsample=subsample).items():
            per_alpha[a].append(float((s[~left] >= tau).mean()))

    out = {}
    for a, vals in per_alpha.items():
        arr = np.asarray(vals)
        mean = float(arr.mean()) if len(arr) else float("nan")
        spread = float(arr.std()) if len(arr) else float("nan")
        out[a] = {"nominal": a, "realized_fpr": mean, "gap": mean - a, "spread": spread,
                  "covered": bool(mean <= a + (spread if len(arr) else 0.0))}
    return out


def calibration_error(report: dict) -> float:
    gaps = [abs(r["gap"]) for r in report.values() if np.isfinite(r["gap"])]
    return float(np.mean(gaps)) if gaps else float("nan")


def effective_n(scores: np.ndarray, max_lag: int = 2000) -> float:
    """n / integrated autocorrelation time."""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 8:
        return float(n)
    s = s - s.mean()
    var = float(s @ s) / n
    if var <= 0:
        return float(n)
    tau = 1.0
    for lag in range(1, min(max_lag, n - 1)):
        rho = float(s[lag:] @ s[:-lag]) / ((n - lag) * var)
        if rho <= 0:
            break
        tau += 2.0 * rho
    return float(n / max(tau, 1.0))
