"""Fixed-grid baselines, fed a forward-filled stream. Same hooks as CTAD."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _ReconBaseline(nn.Module):
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def per_channel_score(self, x, m, t) -> torch.Tensor:
        return (x - self.reconstruct(x)) ** 2

    def per_step_score(self, x, m, t) -> torch.Tensor:
        return self.per_channel_score(x, m, t).mean(dim=-1)

    def training_loss(self, x, m, t, cfg: dict):
        loss = ((x - self.reconstruct(x)) ** 2).mean()
        return loss, {"nll": loss.detach(), "kl": x.new_zeros(()), "cons": x.new_zeros(())}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUAE(_ReconBaseline):
    """Discrete counterpart of CTAD."""

    def __init__(self, input_dim, latent_dim=32, hidden_dim=64, **_):
        super().__init__()
        self.enc = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_z = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def reconstruct(self, x):
        _, h = self.enc(x)
        z = self.to_z(h[-1]).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.out(self.dec(z)[0])


class LSTMAE(_ReconBaseline):
    def __init__(self, input_dim, latent_dim=32, hidden_dim=64, **_):
        super().__init__()
        self.enc = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.to_z = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.LSTM(latent_dim, hidden_dim, num_layers=2, batch_first=True)
        self.out = nn.Linear(hidden_dim, input_dim)

    def reconstruct(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(h[-1]).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.out(self.dec(z)[0])


class _PosEnc(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)[:, : pe[:, 1::2].shape[1]]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class TransformerAE(_ReconBaseline):
    def __init__(self, input_dim, hidden_dim=64, n_heads=4, **_):
        super().__init__()
        while hidden_dim % n_heads:
            n_heads -= 1
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos = _PosEnc(hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 2,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(hidden_dim, input_dim)

    def reconstruct(self, x):
        return self.out(self.encoder(self.pos(self.proj(x))))


class ExtremeValue(nn.Module):
    """Per-channel robust z-score, max over channels (TSB-AD trivial reference)."""

    def __init__(self, input_dim, **_):
        super().__init__()
        self.register_buffer("center", torch.zeros(input_dim))
        self.register_buffer("scale", torch.ones(input_dim))
        self._fitted = False
        self.unused = nn.Parameter(torch.zeros(1))

    def _fit(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        med = flat.median(dim=0).values
        self.center.copy_(med)
        self.scale.copy_(1.4826 * (flat - med).abs().median(dim=0).values + 1e-6)
        self._fitted = True

    def per_channel_score(self, x, m, t) -> torch.Tensor:
        if not self._fitted:
            self._fit(x)
        return ((x - self.center) / self.scale).abs()

    def per_step_score(self, x, m, t) -> torch.Tensor:
        return self.per_channel_score(x, m, t).max(dim=-1).values

    def training_loss(self, x, m, t, cfg: dict):
        if not self._fitted:
            self._fit(x)
        zero = (self.unused ** 2).sum()
        return zero, {"nll": zero.detach(), "kl": x.new_zeros(()), "cons": x.new_zeros(())}

    def n_params(self) -> int:
        return 0


REGISTRY = {"gruae": GRUAE, "lstmae": LSTMAE, "transformerae": TransformerAE,
            "extreme": ExtremeValue}


def build_baseline(name: str, input_dim: int, cfg: dict):
    if name not in REGISTRY:
        raise KeyError(f"unknown baseline {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](input_dim=input_dim, latent_dim=cfg["latent_dim"],
                          hidden_dim=cfg["hidden_dim"])
