from __future__ import annotations

import torch


def rk4_solve(func, z0: torch.Tensor, t: torch.Tensor, substeps: int = 1) -> torch.Tensor:
    """Fixed-step RK4. z0 (B, Z), t (L,) -> (L, B, Z)."""
    if t.dim() != 1:
        raise ValueError(f"t must be 1-D, got shape {tuple(t.shape)}")
    substeps = max(1, int(substeps))

    out, z = [z0], z0
    for i in range(t.numel() - 1):
        t0, t1 = t[i], t[i + 1]
        h = (t1 - t0) / substeps
        for s in range(substeps):
            ts = t0 + s * h
            k1 = func(ts, z)
            k2 = func(ts + 0.5 * h, z + 0.5 * h * k1)
            k3 = func(ts + 0.5 * h, z + 0.5 * h * k2)
            k4 = func(ts + h, z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out.append(z)
    return torch.stack(out, dim=0)
