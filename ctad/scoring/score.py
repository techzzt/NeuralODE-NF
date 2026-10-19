"""Per-timestep scores and window -> timeline aggregation."""

from __future__ import annotations

import numpy as np
import torch

from ..data import Split
from ..training.losses import thin_mask


def _slots(split: Split) -> int:
    if split.n_windows < 2:
        return 1
    stride = int(split.starts[1] - split.starts[0])
    return (split.window - 1) // max(1, stride) + 1


@torch.no_grad()
def fit_channel_stats(model, split: Split, batch_size: int, device, max_windows: int = 400):
    """Per-channel median / MAD of the score on unlabeled normal data."""
    model.eval()
    n = min(split.n_windows, max_windows)
    chunks = []
    for i in range(0, n, batch_size):
        x, m, t = split.batch(torch.arange(i, min(i + batch_size, n), device=device))
        chunks.append(model.per_channel_score(x, m, t)[m > 0])
    flat = torch.cat(chunks, dim=0)
    med = flat.median(dim=0).values
    mad = (flat - med).abs().median(dim=0).values
    return med, 1.4826 * mad + 1e-6


@torch.no_grad()
def per_step_score(model, x, m, t, encoder_mask=None, channel_stats=None) -> torch.Tensor:
    """(B, L), NaN at unobserved steps."""
    enc = m if encoder_mask is None else encoder_mask
    if channel_stats is None:
        scores = model.per_step_score(x, enc, t)
    else:
        med, scale = channel_stats
        scores = ((model.per_channel_score(x, enc, t) - med) / scale).mean(dim=-1)
    return torch.where(m > 0, scores, torch.full_like(scores, float("nan")))


def carry_forward(scores: np.ndarray) -> np.ndarray:
    """Last score carried over gaps."""
    idx = np.where(np.isfinite(scores), np.arange(len(scores)), -1)
    np.maximum.accumulate(idx, out=idx)
    out = np.full_like(scores, np.nan)
    seen = idx >= 0
    out[seen] = scores[idx[seen]]
    return out


@torch.no_grad()
def score_timeline(model, split: Split, batch_size: int, device,
                   encoder_thinning: tuple[float, float] | None = None,
                   channel_stats=None) -> np.ndarray:
    """(T,) median over overlapping windows, carried forward."""
    model.eval()
    K = _slots(split)
    buf = torch.full((split.length, K), float("nan"), device=device)
    offs = torch.arange(split.window, device=device)

    for i in range(0, split.n_windows, batch_size):
        idx = torch.arange(i, min(i + batch_size, split.n_windows), device=device)
        x, m, t = split.batch(idx)
        enc_m = m if encoder_thinning is None else thin_mask(m, *encoder_thinning)
        s = per_step_score(model, x, m, t, encoder_mask=enc_m, channel_stats=channel_stats)
        pos = split.starts[idx].unsqueeze(1) + offs.unsqueeze(0)
        slot = (idx % K).unsqueeze(1).expand(-1, split.window)
        buf[pos.reshape(-1), slot.reshape(-1)] = s.reshape(-1)

    return carry_forward(torch.nanmedian(buf, dim=1).values.cpu().numpy())


@torch.no_grad()
def score_with_uncertainty(model, split: Split, batch_size: int, device, n_thin: int = 5,
                           keep_low: float = 0.3, keep_high: float = 1.0, channel_stats=None):
    """(score, epistemic): epistemic = std over random encoder thinnings."""
    base = score_timeline(model, split, batch_size, device, channel_stats=channel_stats)
    if n_thin <= 0:
        return base, np.zeros_like(base)
    draws = [score_timeline(model, split, batch_size, device,
                            encoder_thinning=(keep_low, keep_high), channel_stats=channel_stats)
             for _ in range(n_thin)]
    return base, np.nanstd(np.stack(draws, axis=0), axis=0)
