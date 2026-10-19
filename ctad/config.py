from __future__ import annotations

DATASET_PATHS = {
    "PSM": ["datasets/PSM/PSM.pk"],
    "SWaT": ["datasets/SWaT/SWaT.pk"],
    "WADI": ["datasets/WADI/WADI.pk"],
    "SMD": ["datasets/SMD/SMD.pk"],
    "SMAP": ["datasets/SMAP/SMAP.pk"],
}

_BASE = dict(
    latent_dim=32,
    hidden_dim=64,
    time_emb_dim=8,
    window=100,
    train_stride=25,
    test_stride=25,
    batch_size=256,
    lr=1e-3,
    epochs=30,
    beta_kl=1e-3,
    lambda_cons=1.0,
    keep_low=0.3,
    keep_high=1.0,
    rk4_substeps=1,
    logvar_min=-4.0,
    obs_df=None,
    vus_window=100,
    max_entities=None,
)

DATASET_CONFIG = {
    "PSM": dict(_BASE, epochs=20),
    "SWaT": dict(_BASE, window=120, train_stride=30, test_stride=30, vus_window=120,
                 latent_dim=64, hidden_dim=128, lr=3e-4, epochs=30, logvar_min=-3.0,
                 batch_size=128),
    "WADI": dict(_BASE, window=120, train_stride=60, test_stride=30, vus_window=120,
                 latent_dim=64, hidden_dim=128, lr=3e-4, epochs=25, logvar_min=-3.0,
                 batch_size=128),
    "SMD": dict(_BASE, epochs=40, max_entities=8),
    "SMAP": dict(_BASE, window=50, train_stride=10, test_stride=10, vus_window=50,
                 epochs=40, batch_size=128, max_entities=8),
}

ALPHAS = (0.001, 0.005, 0.01, 0.05, 0.10)
MISSING_RATES = (0.0, 0.5, 0.8, 0.9, 0.95, 0.98)

# name -> (config overrides, structural flags)
ABLATIONS = {
    "full": ({}, {}),
    "no_cons": (dict(lambda_cons=0.0), {}),
    "no_ode": ({}, dict(use_ode=False)),
    "no_mask": ({}, dict(impute=True)),
    "thin_mild": (dict(keep_low=0.7), {}),
    "thin_harsh": (dict(keep_low=0.1), {}),
    "student_t": (dict(obs_df=4.0), {}),
}


def get_config(dataset: str, **overrides) -> dict:
    if dataset not in DATASET_CONFIG:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASET_CONFIG)}")
    cfg = dict(DATASET_CONFIG[dataset])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg
