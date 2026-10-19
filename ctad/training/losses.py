"""NLL + beta * KL + lambda * sampling-consistency."""

from __future__ import annotations

import torch


def thin_mask(m: torch.Tensor, keep_low: float, keep_high: float, generator=None) -> torch.Tensor:
    """Drop observed steps at a per-window keep rate ~ U[low, high]."""
    B, L = m.shape
    u = torch.rand(B, 1, device=m.device, generator=generator) * (keep_high - keep_low) + keep_low
    keep = (torch.rand(B, L, device=m.device, generator=generator) < u).float()
    thinned = m * keep
    empty = thinned.sum(dim=1, keepdim=True) < 1
    return torch.where(empty, m, thinned)


def entrywise_nll(x, mu, logvar, df: float | None = None) -> torch.Tensor:
    """Gaussian (df=None) or Student-t NLL per (t, channel)."""
    z2 = (x - mu) ** 2 * torch.exp(-logvar)
    if df is None:
        return 0.5 * (z2 + logvar)
    return 0.5 * logvar + 0.5 * (float(df) + 1.0) * torch.log1p(z2 / float(df))


def masked_nll(x, mu, logvar, m, df: float | None = None) -> torch.Tensor:
    w = m.unsqueeze(-1)
    denom = (w.sum() * x.shape[-1]).clamp(min=1.0)
    return (entrywise_nll(x, mu, logvar, df) * w).sum() / denom


def kl_standard_normal(mu0, logvar0) -> torch.Tensor:
    logvar0 = logvar0.clamp(-7.0, 7.0)
    return -0.5 * (1 + logvar0 - mu0.pow(2) - logvar0.exp()).sum(dim=1).mean()


def consistency(z_a, z_b) -> torch.Tensor:
    return ((z_a - z_b) ** 2).mean()


def total_loss(out_a, out_b, x, m, beta_kl: float, lambda_cons: float, df: float | None = None):
    # likelihood on the full mask, encoder saw the thinned one
    nll = masked_nll(x, out_a["mu"], out_a["logvar"], m, df)
    kl = kl_standard_normal(out_a["mu0"], out_a["logvar0"])
    cons = x.new_zeros(())

    if out_b is not None:
        nll = 0.5 * (nll + masked_nll(x, out_b["mu"], out_b["logvar"], m, df))
        kl = 0.5 * (kl + kl_standard_normal(out_b["mu0"], out_b["logvar0"]))
        cons = consistency(out_a["z"], out_b["z"])

    total = nll + beta_kl * kl + lambda_cons * cons
    return total, {"nll": nll.detach(), "kl": kl.detach(), "cons": cons.detach()}
