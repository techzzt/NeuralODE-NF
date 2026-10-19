from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from ..config import DATASET_PATHS


@dataclass
class Trace:
    name: str
    x_fit: np.ndarray  # train, unlabeled
    x_cal: np.ndarray  # calibration, unlabeled
    x_tst: np.ndarray  # test
    y_tst: np.ndarray  # point labels
    channels: list[str] | None = None

    @property
    def dim(self) -> int:
        return self.x_fit.shape[1]


def _resolve(dataset: str, root: Path) -> Path:
    for cand in DATASET_PATHS.get(dataset, [f"{dataset}.pk"]):
        if (root / cand).exists():
            return root / cand
    raise FileNotFoundError(f"{dataset}: none of {DATASET_PATHS.get(dataset)} under {root}")


_CACHE: dict[tuple, list[Trace]] = {}


def load_traces(dataset: str, root: str | Path = ".", cal_frac: float = 0.15,
                limit: int | None = None, max_entities: int | None = None,
                min_length: int = 0) -> list[Trace]:
    key = (dataset, str(root), cal_frac, limit, max_entities, min_length)
    if key in _CACHE:
        return _CACHE[key]

    with open(_resolve(dataset, Path(root)), "rb") as f:
        raw = pickle.load(f)

    x_trn, x_tst, y_tst = raw["x_trn"], raw["x_tst"], raw["lab_tst"]
    names_raw = raw.get("channels", raw.get("cols"))
    channels = [str(c) for c in np.asarray(names_raw).ravel()] if names_raw is not None else None

    if isinstance(x_trn, list):
        entities = list(zip(x_trn, x_tst, y_tst))[:max_entities]
        names = [f"{dataset}-{i}" for i in range(len(entities))]
    else:
        entities, names = [(x_trn, x_tst, y_tst)], [dataset]

    traces = []
    for name, (xt, xs, ys) in zip(names, entities):
        xt = np.asarray(xt, dtype=np.float64)[:limit]
        xs = np.asarray(xs, dtype=np.float64)[:limit]
        ys = (np.asarray(ys).ravel() > 0).astype(np.int64)[:limit]

        n_cal = max(1, int(len(xt) * cal_frac))
        if min_length and min(len(xt) - n_cal, n_cal, len(xs)) < min_length:
            print(f"  skip {name}: too short")
            continue

        fit_raw, cal_raw = xt[: len(xt) - n_cal], xt[len(xt) - n_cal:]
        scaler = StandardScaler().fit(fit_raw)  # fit split only
        traces.append(Trace(
            name=name,
            x_fit=scaler.transform(fit_raw).astype(np.float32),
            x_cal=scaler.transform(cal_raw).astype(np.float32),
            x_tst=scaler.transform(xs).astype(np.float32),
            y_tst=ys[: len(xs)],
            channels=channels if channels and len(channels) == xt.shape[1] else None,
        ))

    if not traces:
        raise RuntimeError(f"{dataset}: every trace shorter than {min_length}")
    _CACHE[key] = traces
    return traces
