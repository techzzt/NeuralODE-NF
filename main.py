"""CTAD entry point.

    python main.py --dataset PSM
    python main.py --dataset SWaT --rate 0.9
    python main.py --sweep --datasets PSM SWaT SMD --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from ctad.config import ABLATIONS, MISSING_RATES, get_config
from ctad.data import load_traces
from ctad.models import BASELINES
from ctad.pipeline import RunSpec, run_trace


def run(spec: RunSpec, args, device) -> list[dict]:
    overrides, flags = ABLATIONS[spec.ablation]
    spec = replace(spec, **flags)
    cfg = {**get_config(spec.dataset, epochs=args.epochs, window=args.window), **overrides}
    traces = load_traces(spec.dataset, root=args.root, limit=args.limit,
                         max_entities=cfg["max_entities"],
                         min_length=cfg["window"] + cfg["test_stride"])
    return [run_trace(spec, cfg, tr, device) for tr in traces]


def append(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="CTAD: continuous-time anomaly detection")
    p.add_argument("--dataset", default="PSM")
    p.add_argument("--model", default="ctad", choices=["ctad", *sorted(BASELINES)])
    p.add_argument("--ablation", default="full", choices=sorted(ABLATIONS))
    p.add_argument("--missing", default="mcar", choices=["mcar", "burst", "gap"])
    p.add_argument("--rate", type=float, default=0.0)
    p.add_argument("--mean-gap", type=int, default=20)
    p.add_argument("--scenario", default="native", choices=["native", "deploy"])
    p.add_argument("--contamination", type=float, default=0.0)
    p.add_argument("--volatility", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int)
    p.add_argument("--window", type=int)
    p.add_argument("--limit", type=int, help="truncate streams (smoke test)")
    p.add_argument("--root", default=".")
    p.add_argument("--out", default="results/summary.csv")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--datasets", nargs="+", default=["PSM", "SWaT", "SMD"])
    p.add_argument("--rates", nargs="+", type=float, default=list(MISSING_RATES))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    base = RunSpec(dataset=args.dataset, model=args.model, ablation=args.ablation,
                   missing=args.missing, rate=args.rate, mean_gap=args.mean_gap,
                   scenario=args.scenario, contamination=args.contamination,
                   volatility=args.volatility, seed=args.seed, verbose=not args.quiet)
    out = Path(args.out)

    if not args.sweep:
        append(run(base, args, device), out)
    else:
        jobs = list(itertools.product(args.datasets, args.rates, args.seeds))
        for i, (d, r, s) in enumerate(jobs, 1):
            print(f"\n--- [{i}/{len(jobs)}] {d} rate={r} seed={s} ---", flush=True)
            append(run(replace(base, dataset=d, rate=r, seed=s), args, device), out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
