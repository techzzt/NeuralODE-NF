import numpy as np
import torch

from ctad.data import make_split, observation_mask
from ctad.evaluation.metrics import apply_pa_k, segments, vus
from ctad.models import CTAD
from ctad.models.odeint import rk4_solve
from ctad.scoring import carry_forward, conformal_threshold, coverage_report, calibrate
from ctad.training import thin_mask


def test_rk4_exponential():
    t = torch.linspace(0, 1, 11)
    z = rk4_solve(lambda _, z: -z, torch.ones(1, 1), t)
    assert torch.allclose(z[-1], torch.exp(torch.tensor(-1.0)), atol=1e-5)


def test_ctad_shapes_and_loss():
    model = CTAD(input_dim=4, latent_dim=8, hidden_dim=16)
    x, m = torch.randn(3, 20, 4), torch.ones(3, 20)
    t = torch.linspace(0, 1, 20).expand(3, -1)
    out = model(x, m, t)
    assert out["mu"].shape == (3, 20, 4) and out["z"].shape == (3, 20, 8)
    loss, _ = model.training_loss(x, m, t, {"lambda_cons": 1.0, "beta_kl": 1e-3})
    loss.backward()


def test_encoder_ignores_unobserved_values():
    model = CTAD(input_dim=2, latent_dim=4, hidden_dim=8).eval()
    t = torch.linspace(0, 1, 10).expand(1, -1)
    m = torch.ones(1, 10)
    m[0, 3:6] = 0
    x1 = torch.randn(1, 10, 2)
    x2 = x1.clone()
    x2[0, 3:6] += 100.0
    assert torch.allclose(model.encoder(x1, m, t)[0], model.encoder(x2, m, t)[0])


def test_thin_mask_keeps_rows_nonempty():
    m = torch.zeros(4, 10)
    m[:, 0] = 1
    assert (thin_mask(m, 0.0, 0.0).sum(1) >= 1).all()


def test_observation_mask_rate_and_anchor():
    obs = observation_mask(10000, 0.9, "mcar", np.random.default_rng(0))
    assert obs[0] and abs((~obs).mean() - 0.9) < 0.02


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    thr = calibrate(rng.normal(size=5000), [0.05])
    rep = coverage_report(rng.normal(size=20000), thr)
    assert abs(rep[0.05]["realized_fpr"] - 0.05) < 0.01
    assert conformal_threshold(np.arange(5), 0.001) == float("inf")


def test_carry_forward_and_split():
    s = carry_forward(np.array([np.nan, 1.0, np.nan, 2.0]))
    assert np.isnan(s[0]) and s[2] == 1.0
    sp = make_split(np.random.randn(200, 3).astype(np.float32), 50, 25, torch.device("cpu"))
    x, m, t = sp.batch(torch.arange(sp.n_windows))
    assert x.shape == (7, 50, 3) and t[0, -1] == 1.0


def test_metrics():
    y = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    assert segments(y).tolist() == [[1, 3], [6, 6]]
    assert apply_pa_k(np.array([0, 1, 0, 0, 0, 0, 0, 0]), y, 0).tolist() == [0, 1, 1, 1, 0, 0, 0, 0]
    _, vpr = vus(y, y.astype(float), max_window=2, device=torch.device("cpu"))
    assert vpr > 0.9
