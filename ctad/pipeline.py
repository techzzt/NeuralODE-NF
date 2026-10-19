"""Train -> calibrate (label-free) -> score -> evaluate, for one trace."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .config import ALPHAS
from .data import Trace, apply_volatility, inject_anomalies, make_split
from .evaluation import evaluate
from .models import CTAD, build_baseline
from .scoring import (calibrate, calibration_error, coverage_report, effective_n,
                      fit_channel_stats, min_alpha, repeated_coverage, score_timeline,
                      score_with_uncertainty)
from .training import train


@dataclass
class RunSpec:
    dataset: str = "PSM"
    model: str = "ctad"
    ablation: str = "full"
    missing: str = "mcar"
    rate: float = 0.0
    mean_gap: int = 20
    scenario: str = "native"   # native | deploy
    contamination: float = 0.0
    volatility: float = 0.0
    seed: int = 0
    n_thin: int = 5
    use_ode: bool = True
    impute: bool = False
    verbose: bool = True


def build_model(spec: RunSpec, cfg: dict, dim: int, device):
    if spec.model == "ctad":
        return CTAD(input_dim=dim, latent_dim=cfg["latent_dim"], hidden_dim=cfg["hidden_dim"],
                    time_emb_dim=cfg["time_emb_dim"], rk4_substeps=cfg["rk4_substeps"],
                    use_mask_channel=not spec.impute, use_ode=spec.use_ode,
                    logvar_min=cfg["logvar_min"], obs_df=cfg["obs_df"]).to(device)
    return build_baseline(spec.model, dim, cfg).to(device)


def run_trace(spec: RunSpec, cfg: dict, trace: Trace, device) -> dict:
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed + 1000)

    # stressors
    x_fit, x_cal, x_tst = trace.x_fit, trace.x_cal, trace.x_tst
    if spec.contamination > 0:
        x_fit, _ = inject_anomalies(x_fit, spec.contamination, rng)
        x_cal, _ = inject_anomalies(x_cal, spec.contamination, rng)
    if spec.volatility > 0:
        x_tst = apply_volatility(x_tst, spec.volatility, rng)

    # splits
    train_rate = spec.rate if spec.scenario == "native" else 0.0
    impute = spec.impute or spec.model != "ctad"
    common = dict(window=cfg["window"], device=device, missing_kind=spec.missing,
                  mean_gap=spec.mean_gap, impute=impute)
    fit = make_split(x_fit, stride=cfg["train_stride"], missing_rate=train_rate,
                     seed=spec.seed, **common)
    cal = make_split(x_cal, stride=cfg["test_stride"], missing_rate=train_rate,
                     seed=spec.seed + 1, **common)
    tst = make_split(x_tst, stride=cfg["test_stride"], labels=trace.y_tst,
                     missing_rate=spec.rate, seed=spec.seed + 3, **common)

    # train
    model = build_model(spec, cfg, trace.dim, device)
    print(f"[{trace.name}] model={spec.model} params={model.n_params()} "
          f"fit_windows={fit.n_windows} test_windows={tst.n_windows}", flush=True)
    history = train(model, fit, cal, cfg, device, verbose=spec.verbose)

    # calibrate on unlabeled normal stream
    chan_stats = fit_channel_stats(model, cal, cfg["batch_size"], device)
    cal_scores = score_timeline(model, cal, cfg["batch_size"], device, channel_stats=chan_stats)
    cal_scores = cal_scores[np.isfinite(cal_scores)]
    thresholds = calibrate(cal_scores, ALPHAS)
    half = len(cal_scores) // 2
    cov_exch = repeated_coverage(cal_scores, ALPHAS, seed=spec.seed)
    cov_drift = coverage_report(cal_scores[half:], calibrate(cal_scores[:half], ALPHAS))

    # score test stream
    t0 = time.time()
    scores, epistemic = score_with_uncertainty(
        model, tst, cfg["batch_size"], device, n_thin=spec.n_thin,
        keep_low=cfg["keep_low"], keep_high=cfg["keep_high"], channel_stats=chan_stats)
    infer_seconds = time.time() - t0
    keep = np.isfinite(scores)

    metrics = evaluate(tst.y[keep], scores[keep], conformal_thresholds=thresholds,
                       vus_window=cfg["vus_window"])

    row = {
        "dataset": spec.dataset, "trace": trace.name, "model": spec.model,
        "ablation": spec.ablation, "scenario": spec.scenario, "missing_kind": spec.missing,
        "missing_rate": spec.rate, "contamination": spec.contamination,
        "volatility": spec.volatility, "seed": spec.seed, "epochs": cfg["epochs"],
        "n_params": model.n_params(), "train_seconds": history["train_seconds"],
        "infer_seconds": infer_seconds, "best_epoch": history["best_epoch"],
        "cal_nll": history["best_cal_nll"], "n_cal": int(len(cal_scores)),
        "n_cal_effective": effective_n(cal_scores),
        "min_certifiable_alpha": min_alpha(len(cal_scores)),
        "calibration_error": calibration_error(cov_exch),
        "calibration_error_drift": calibration_error(cov_drift),
        "observed_frac": float((tst.obs > 0).float().mean()),
        "mean_epistemic": float(np.nanmean(epistemic[keep])) if spec.n_thin > 0 else float("nan"),
        **metrics,
    }
    if spec.verbose:
        print_report(row)
    return row


def print_report(row: dict) -> None:
    base = row["base_rate"]
    print("\n" + "=" * 64)
    print(f"  {row['dataset']} / {row['trace']}  ({row['model']}, {row['ablation']}, "
          f"{row['missing_rate']:.0%} missing)")
    print("=" * 64)
    print(f"  AUC-PR {row['auc_pr']:.4f} ({row['auc_pr'] / base:.2f}x base)   "
          f"VUS-PR {row['vus_pr']:.4f}")
    print(f"  best F1 [oracle] {row['best_f1']:.4f}   PA20-F1 {row['pa20_f1']:.4f}   "
          f"always-alarm F1 {row['trivial_f1']:.4f}")
    print(f"  {'alpha':>7s} {'prec':>7s} {'recall':>7s} {'F1':>7s} {'FPR':>8s}")
    for a in ALPHAS:
        if f"a{a:g}_f1" in row:
            print(f"  {a:>7g} {row[f'a{a:g}_precision']:>7.3f} {row[f'a{a:g}_recall']:>7.3f} "
                  f"{row[f'a{a:g}_f1']:>7.3f} {row[f'a{a:g}_fpr']:>8.4f}")
    print("=" * 64 + "\n", flush=True)
