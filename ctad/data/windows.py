from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .missing import forward_fill, observation_mask


@dataclass
class Split:
    """Windowed view of one stream; windows are built lazily in `batch`."""

    x: torch.Tensor          # (T, D), unobserved rows zeroed
    obs: torch.Tensor        # (T,) observation mask
    starts: torch.Tensor     # (N,) window starts
    window: int
    y: np.ndarray | None = None
    x_model: torch.Tensor | None = None  # model input (imputed for baselines)
    m_model: torch.Tensor | None = None

    def __post_init__(self):
        if self.x_model is None:
            self.x_model = self.x
        if self.m_model is None:
            self.m_model = self.obs

    @property
    def n_windows(self) -> int:
        return int(self.starts.numel())

    @property
    def length(self) -> int:
        return int(self.x.shape[0])

    def batch(self, idx: torch.Tensor):
        """-> x (B, L, D), m (B, L), t (B, L) with t in [0, 1]."""
        offs = torch.arange(self.window, device=self.x.device)
        pos = self.starts[idx].unsqueeze(1) + offs.unsqueeze(0)
        t = (offs.float() / max(1, self.window - 1)).unsqueeze(0).expand(pos.shape[0], -1)
        return self.x_model[pos], self.m_model[pos], t


def make_split(values: np.ndarray, window: int, stride: int, device: torch.device,
               labels: np.ndarray | None = None, missing_rate: float = 0.0,
               missing_kind: str = "mcar", seed: int = 0, mean_gap: int = 20,
               impute: bool = False) -> Split:
    T = len(values)
    if T < window:
        raise ValueError(f"stream of length {T} is shorter than window {window}")

    observed = observation_mask(T, missing_rate, missing_kind, np.random.default_rng(seed), mean_gap)
    obs = torch.as_tensor(observed.astype(np.float32), device=device)
    x = torch.as_tensor(values, dtype=torch.float32, device=device) * obs.unsqueeze(1)

    x_model = m_model = None
    if impute:
        x_model = torch.as_tensor(forward_fill(np.asarray(values, dtype=np.float32), observed),
                                  device=device)
        m_model = torch.ones_like(obs)

    starts = torch.arange(0, T - window + 1, stride, device=device)
    return Split(x=x, obs=obs, starts=starts, window=window, y=labels,
                 x_model=x_model, m_model=m_model)
