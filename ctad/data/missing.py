from __future__ import annotations

import numpy as np

ANOMALY_KINDS = ("spike", "level_shift", "stuck", "drift", "variance_burst")


def observation_mask(length: int, rate: float, kind: str = "mcar",
                     rng: np.random.Generator | None = None, mean_gap: int = 20) -> np.ndarray:
    """True = observed. mcar | burst (Markov) | gap (fixed blackouts)."""
    rng = rng or np.random.default_rng(0)
    if rate <= 0:
        return np.ones(length, dtype=bool)
    if rate >= 1:
        raise ValueError("rate must be < 1")

    if kind == "mcar":
        observed = rng.random(length) >= rate
    elif kind == "burst":
        b = 1.0 / max(1, mean_gap)
        a = b * rate / (1.0 - rate)
        observed = np.empty(length, dtype=bool)
        state, draws = True, rng.random(length)
        for i in range(length):
            observed[i] = state
            state = (draws[i] >= a) if state else (draws[i] < b)
    elif kind == "gap":
        observed = np.ones(length, dtype=bool)
        block, target, guard = max(1, int(mean_gap)), int(length * rate), 0
        while (~observed).sum() < target and guard < 20 * (target // block + 1) + 100:
            guard += 1
            start = int(rng.integers(0, max(1, length - block)))
            if observed[max(0, start - 1): start + block + 1].all():
                observed[start: start + block] = False
    else:
        raise ValueError(f"unknown missingness kind {kind!r}")

    observed[0] = True  # anchor
    return observed


def forward_fill(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    idx = np.where(observed, np.arange(len(observed)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = values[idx]
    if not observed[0]:
        filled[: np.argmax(observed)] = 0.0
    return filled


def inject_anomalies(values: np.ndarray, rate: float, rng: np.random.Generator,
                     kinds: tuple[str, ...] = ANOMALY_KINDS, min_len: int = 5,
                     max_len: int = 60, strength: float = 3.0):
    """Synthetic faults for the contamination stressor. Returns (values, injected mask)."""
    x = np.array(values, dtype=np.float64, copy=True)
    T, D = x.shape
    injected = np.zeros(T, dtype=bool)
    if rate <= 0:
        return x.astype(np.float32), injected

    scale = x.std(axis=0) + 1e-8
    budget, guard = int(T * rate), 0
    while injected.sum() < budget and guard < 10 * budget + 100:
        guard += 1
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(1, T - length)))
        if injected[start: start + length].any():
            continue
        n_ch = max(1, int(rng.integers(1, max(2, D // 4) + 1)))
        chans = rng.choice(D, size=n_ch, replace=False)
        kind = kinds[int(rng.integers(0, len(kinds)))]
        block = np.ix_(np.arange(start, start + length), chans)

        if kind == "spike":
            sign = rng.choice([-1.0, 1.0], size=n_ch)
            x[block] += strength * scale[chans] * sign * rng.uniform(1.0, 2.0, size=n_ch)
        elif kind == "level_shift":
            x[block] += strength * 0.7 * scale[chans] * rng.choice([-1.0, 1.0], size=n_ch)
        elif kind == "stuck":
            x[block] = x[start, chans]
        elif kind == "drift":
            ramp = np.linspace(0.0, 1.0, length)[:, None]
            x[block] = x[block] + ramp * 0.8 * scale[chans] * rng.choice([-1.0, 1.0], size=n_ch)
            lo = np.quantile(x[:, chans], 0.01, axis=0)
            hi = np.quantile(x[:, chans], 0.99, axis=0)
            x[block] = np.clip(x[block], lo, hi)
        elif kind == "variance_burst":
            mean = x[block].mean(axis=0, keepdims=True)
            x[block] = mean + (x[block] - mean) * strength + rng.normal(
                0.0, strength * 0.3 * scale[chans], size=(length, n_ch))
        injected[start: start + length] = True

    return x.astype(np.float32), injected


def apply_volatility(values: np.ndarray, level: float, rng: np.random.Generator,
                     n_regimes: int = 12) -> np.ndarray:
    """Non-fault variance regimes for the volatility stressor."""
    x = np.array(values, dtype=np.float64, copy=True)
    if level <= 0:
        return x.astype(np.float32)
    T, D = x.shape
    scale = x.std(axis=0) + 1e-8
    bounds = np.sort(rng.choice(np.arange(1, T), size=min(2 * n_regimes, T - 1), replace=False))
    for lo, hi in zip(bounds[::2], bounds[1::2]):
        seg = slice(int(lo), int(hi))
        factor = 1.0 + level * rng.uniform(0.5, 1.5)
        mean = x[seg].mean(axis=0, keepdims=True)
        x[seg] = mean + (x[seg] - mean) * factor
        x[seg] += rng.normal(0.0, level * 0.2 * scale, size=(int(hi - lo), D))
    return x.astype(np.float32)
