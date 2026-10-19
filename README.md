# CTAD

### Continuous-Time Anomaly Detection for Irregular, Unlabeled Time Series with Latent Neural ODEs

PyTorch implementation of **CTAD**, an anomaly detector for multivariate sensor streams that are **irregularly sampled** and **unlabeled**.

The project started at **Georgia Institute of Technology** with campus building-meter data. That data has hours-long gaps and counter resets, and no fault labels. Detectors built for regularly sampled benchmarks need imputed inputs and test labels to tune thresholds, and neither is available there. CTAD instead models the observed points directly in continuous time, and sets its alarm threshold from normal data alone.

Given a window with values $x_t$, observation mask $m_t$ and timestamps $t$, CTAD encodes only the observed points into an initial latent state. It evolves that state with a Neural ODE and decodes a per-channel Gaussian at every timestamp:

$$
z_0 \sim q_\phi(z_0 \mid x, m), \qquad \frac{dz}{dt} = f_\theta(z, t), \qquad x_t \sim \mathcal{N}\left(\mu(z_t), ~ \sigma^2(z_t)\right)
$$

---

## Model

```
x(t), m(t)            irregular window: values + observation mask
    ↓
Masked GRU encoder    reverse time, skips unobserved steps
    ↓
z0                    initial latent state
    ↓
Latent Neural ODE     dz/dt = f(z, t), RK4 over the full window grid
    ↓
Decoder               mean and variance per channel at every timestamp
    ↓
Score s(t)            channel-normalized negative log-likelihood
    ↓
Alarm                 s(t) >= threshold, set by conformal calibration on normal data
```

- **Masked GRU encoder:** reads observed values, the mask and the time since the last observation, backwards in time. **Unobserved steps leave the hidden state unchanged**, so the model never sees an imputed value.
- **Latent Neural ODE:** integrated with fixed-step RK4 over the **whole window grid**, so the latent state is defined **even where nothing was measured**.
- **Heteroscedastic decoder:** outputs a mean and a variance per channel. A **variance floor** keeps near-constant channels from dominating the score.

**Anomaly score.** Per-channel negative log-likelihood, normalized by its median and MAD on normal data, then averaged over channels:

$$
s(t) = \frac{1}{D} \sum_{c=1}^{D} \frac{\ell_c(t) - \mathrm{med}_c}{\mathrm{MAD}_c}, \qquad
\ell_c(t) = \frac{1}{2} \left[ \frac{\left(x_{t,c} - \mu_c(t)\right)^2}{\sigma_c^2(t)} + \log \sigma_c^2(t) \right]
$$

Overlapping windows are combined by their median. **Unobserved timestamps carry the last score forward**, so every missing rate is evaluated on the same timeline.

**Alarm threshold.** Split-conformal calibration on a held-out, unlabeled normal stream of $n$ scores:

$$
\tau_\alpha = s_{(k)}, \quad k = \lceil (n+1)(1-\alpha) \rceil, \qquad
P\left(s_{\mathrm{test}} \ge \tau_\alpha\right) \le \alpha ~~ \text{(under exchangeability)}
$$

### Training objective

Each window is encoded under **two random thinnings** of its mask, $\tilde m^{(1)}$ and $\tilde m^{(2)}$. The likelihood is computed on **all observed steps**, so the ODE has to fill gaps the encoder was not shown:

$$
\mathcal{L} = \mathrm{NLL}(x \mid z, m) + \beta ~ \mathrm{KL}\left(q(z_0) ~ \Vert ~ \mathcal{N}(0, I)\right) + \lambda ~ \mathbb{E}_t \left[ \Vert z^{(1)}(t) - z^{(2)}(t) \Vert^2 \right]
$$

- The **consistency term** asks the latent dynamics not to depend on which timestamps happened to be sampled.
- **Label-free protocol:** the training stream is split into **fit (85%)** and a **calibration tail (15%)**. The checkpoint and the thresholds are both chosen on the calibration tail. **Test labels are used only for the reported metrics.**

---

## Main Results

CTAD with timestamps removed uniformly at random from both the training and test streams. Mean ± std over 3 seeds; SMD is averaged over 8 machines.

| Dataset | Base rate | AUC-PR, 0% missing | AUC-PR, 90% missing | VUS-PR, 0% missing | VUS-PR, 90% missing |
|---|---|---|---|---|---|
| PSM | 0.278 | 0.616 ± 0.012 | 0.635 ± 0.015 | 0.605 ± 0.012 | 0.552 ± 0.017 |
| SWaT | 0.121 | 0.608 ± 0.045 | 0.463 ± 0.185 | 0.432 ± 0.033 | 0.335 ± 0.093 |
| SMD | 0.043 | 0.424 ± 0.039 | 0.375 ± 0.021 | 0.377 ± 0.041 | 0.256 ± 0.009 |

Note: following recent benchmark critiques, **AUC-PR and VUS-PR are the main metrics**. Point-adjusted F1 and oracle-threshold F1 are kept in [`results/main.csv`](results/main.csv) only for comparison with prior work.

- **Robust to missing data:** PSM VUS-PR drops only **0.605 → 0.552** with 90% of timestamps removed.
- **Calibration does not carry over to the test period** (main limitation): on held-out normal data the false-alarm rate matches α to within 0.01. On normal points in the test period, α = 1% fires **3.6% (PSM), 25% (SMD), 44% (SWaT)** of the time, because normal behavior shifts between the two periods.
- Fixed-grid baselines and ablations are implemented, but no completed comparison is included, so **no advantage over them is claimed**.

---

## Code Structure

```
.
├── main.py                 # entry point: single run or sweep
├── ctad/
│   ├── config.py           # per-dataset hyperparameters, ablations
│   ├── pipeline.py         # train → calibrate → score → evaluate (one trace)
│   ├── data/
│   │   ├── loader.py       # load .pk, fit / calibration / test split, scaling
│   │   ├── missing.py      # missingness (mcar, burst, gap), contamination, volatility
│   │   └── windows.py      # sliding windows with observation masks
│   ├── models/
│   │   ├── ctad.py         # masked GRU encoder, ODE function, decoder
│   │   ├── odeint.py       # fixed-step RK4 solver
│   │   └── baselines.py    # GRU-AE, LSTM-AE, Transformer-AE, extreme-value
│   ├── training/
│   │   ├── losses.py       # NLL, KL, sampling consistency
│   │   └── trainer.py      # training loop, checkpoint by calibration NLL
│   ├── scoring/
│   │   ├── score.py        # per-timestep scores, timeline aggregation
│   │   └── conformal.py    # conformal thresholds, coverage checks
│   └── evaluation/
│       └── metrics.py      # AUC-PR, VUS-PR, range / event / PA%K F1
├── results/main.csv        # runs behind the table above
└── tests/test_core.py
```

---

## Setup

### Installation

```bash
conda create -n ctad python=3.11
conda activate ctad
pip install torch
pip install -r requirements.txt
```

### Training and Testing

Each call trains the model, calibrates thresholds on unlabeled normal data, scores the test stream and appends one row per trace to `results/summary.csv`.

```bash
python main.py --dataset PSM
python main.py --dataset SWaT --rate 0.9
python main.py --sweep --datasets PSM SWaT SMD --rates 0 0.5 0.9 --seeds 0 1 2
```

| Argument | Meaning |
|---|---|
| `--dataset` | `PSM`, `SWaT`, `WADI`, `SMD`, `SMAP` |
| `--model` | `ctad` (default) or a fixed-grid baseline: `gruae`, `lstmae`, `transformerae`, `extreme` |
| `--rate`, `--missing` | fraction of timestamps removed; pattern `mcar`, `burst` or `gap` |
| `--ablation` | `full`, `no_ode`, `no_cons`, `no_mask`, `thin_mild`, `thin_harsh`, `student_t` |
| `--contamination` | fraction of injected faults hidden in the "normal" training data |
| `--volatility` | non-fault variance shifts added to the test stream |
| `--epochs`, `--limit` | shorter runs for quick checks |

Hyperparameters per dataset are in `ctad/config.py`.

```bash
python -m pytest tests      # unit tests, no data needed
```

---

## Datasets

Put each dataset at `./datasets/DATASET/DATASET.pk`. The file is a pickled dictionary `dat` with three keys:

- `dat['x_trn']`: training data, shape `(time points, dims)`, assumed normal and unlabeled
- `dat['x_tst']`: test data, shape `(time points, dims)`
- `dat['lab_tst']`: test labels, shape `(time points,)`

For multi-entity datasets such as SMD, each key holds a list of arrays, one per entity. To add a new dataset, register its path in `DATASET_PATHS` and its hyperparameters in `DATASET_CONFIG` in `ctad/config.py`.

---

## Status

This repository is the **Neural ODE stage** of the project. The calibration gap above motivates the ongoing follow-up, which is not included here: a **masked autoencoder with prototype memory** of normal operating modes, with **missingness-conditioned decision radii**.
