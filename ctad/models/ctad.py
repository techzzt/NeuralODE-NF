"""Masked GRU encoder -> latent Neural ODE -> heteroscedastic decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..training.losses import entrywise_nll, thin_mask, total_loss
from .odeint import rk4_solve


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(1, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.linear(t.reshape(-1, 1))


class ODEFunc(nn.Module):
    """dz/dt = f(z, t)."""

    def __init__(self, latent_dim: int, hidden_dim: int, time_emb_dim: int):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t.reshape(1)).expand(z.shape[0], -1)
        return self.net(torch.cat([z, t_emb], dim=-1))


class MaskedGRUEncoder(nn.Module):
    """q(z0 | observed steps). Reverse-time GRU; unobserved steps skip the update."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int,
                 use_mask_channel: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_mask_channel = use_mask_channel
        feat = input_dim + (input_dim + 1 if use_mask_channel else 0)  # x, mask, gap
        self.cell = nn.GRUCell(feat, hidden_dim)
        self.to_mu = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, m: torch.Tensor, t: torch.Tensor):
        B, L, D = x.shape
        h = x.new_zeros(B, self.hidden_dim)
        gap = x.new_zeros(B)
        prev_t = t[:, -1]

        for i in range(L - 1, -1, -1):
            mi = m[:, i]
            gap = torch.where(mi > 0, torch.zeros_like(gap), gap + (prev_t - t[:, i]).abs())
            xi = x[:, i] * mi.unsqueeze(1)
            feat = (torch.cat([xi, mi.unsqueeze(1).expand(-1, D), gap.unsqueeze(1)], dim=-1)
                    if self.use_mask_channel else xi)
            h = torch.where(mi.unsqueeze(1) > 0, self.cell(feat, h), h)
            prev_t = torch.where(mi > 0, t[:, i], prev_t)

        return self.to_mu(h), self.to_logvar(h)


class HeteroDecoder(nn.Module):
    """z(t) -> (mu, log sigma^2) per channel."""

    def __init__(self, latent_dim: int, input_dim: int, hidden_dim: int,
                 logvar_min: float = -4.0, logvar_max: float = 7.0):
        super().__init__()
        self.input_dim = input_dim
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, input_dim * 2)

    def forward(self, z: torch.Tensor):
        L, B, _ = z.shape
        out = self.head(self.net(z.reshape(L * B, -1)))
        mu, logvar = out[:, : self.input_dim], out[:, self.input_dim:]
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)  # variance floor
        return mu.reshape(L, B, -1), logvar.reshape(L, B, -1)


class CTAD(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 64,
                 time_emb_dim: int = 8, rk4_substeps: int = 1, use_mask_channel: bool = True,
                 use_ode: bool = True, logvar_min: float = -4.0, obs_df: float | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.rk4_substeps = rk4_substeps
        self.use_ode = use_ode
        self.obs_df = obs_df
        self.encoder = MaskedGRUEncoder(input_dim, hidden_dim, latent_dim, use_mask_channel)
        self.ode_func = ODEFunc(latent_dim, hidden_dim, time_emb_dim)
        self.decoder = HeteroDecoder(latent_dim, input_dim, hidden_dim, logvar_min=logvar_min)

    def forward(self, x, m, t, sample: bool = True) -> dict:
        mu0, logvar0 = self.encoder(x, m, t)
        z0 = mu0 + torch.randn_like(mu0) * torch.exp(0.5 * logvar0.clamp(-7.0, 7.0)) if sample else mu0

        grid = t[0]
        if self.use_ode:
            z = rk4_solve(self.ode_func, z0, grid, substeps=self.rk4_substeps)
        else:
            z = z0.unsqueeze(0).expand(grid.numel(), -1, -1)  # ablation

        mu, logvar = self.decoder(z)
        return {"mu": mu.transpose(0, 1), "logvar": logvar.transpose(0, 1),
                "z": z.transpose(0, 1), "mu0": mu0, "logvar0": logvar0}

    def per_channel_score(self, x, m, t) -> torch.Tensor:
        out = self(x, m, t, sample=False)
        return entrywise_nll(x, out["mu"], out["logvar"], self.obs_df)

    def per_step_score(self, x, m, t) -> torch.Tensor:
        return self.per_channel_score(x, m, t).mean(dim=-1)

    def training_loss(self, x, m, t, cfg: dict):
        lam = float(cfg.get("lambda_cons", 0.0))
        lo, hi = cfg.get("keep_low", 0.3), cfg.get("keep_high", 1.0)
        # two random thinnings of the same window
        out_a = self(x, thin_mask(m, lo, hi), t)
        out_b = self(x, thin_mask(m, lo, hi), t) if lam > 0 else None
        return total_loss(out_a, out_b, x, m, cfg.get("beta_kl", 0.0), lam, df=self.obs_df)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
