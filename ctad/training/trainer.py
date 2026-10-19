"""Training loop. Checkpoint chosen on calibration NLL (no test labels)."""

from __future__ import annotations

import time

import torch

from ..data import Split


def iter_batches(split: Split, batch_size: int, shuffle: bool, device):
    n = split.n_windows
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, batch_size):
        yield split.batch(order[i: i + batch_size])


@torch.no_grad()
def evaluate_score(model, split: Split, batch_size: int, device) -> float:
    model.eval()
    total, count = 0.0, 0.0
    for x, m, t in iter_batches(split, batch_size, shuffle=False, device=device):
        s = model.per_step_score(x, m, t)
        total += float((s * m).sum())
        count += float(m.sum())
    return total / max(1.0, count)


def train(model, fit_split: Split, cal_split: Split, cfg: dict, device,
          verbose: bool = True) -> dict:
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["epochs"]))

    history = {"train": [], "cal": []}
    best = {"cal": float("inf"), "epoch": -1, "state": None}
    t_start = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        running, parts, nb = 0.0, {"nll": 0.0, "kl": 0.0, "cons": 0.0}, 0
        for x, m, t in iter_batches(fit_split, cfg["batch_size"], shuffle=True, device=device):
            loss, log = model.training_loss(x, m, t, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss)
            for k in parts:
                parts[k] += float(log[k])
            nb += 1

        sched.step()
        nb = max(1, nb)
        cal_nll = evaluate_score(model, cal_split, cfg["batch_size"], device)
        history["train"].append(running / nb)
        history["cal"].append(cal_nll)

        if cal_nll < best["cal"] - 1e-5:
            best = {"cal": cal_nll, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

        if verbose:
            print(f"  epoch {epoch + 1:3d}/{cfg['epochs']}  train={running / nb:8.4f}  "
                  f"cal={cal_nll:8.4f}  nll={parts['nll'] / nb:7.4f}  "
                  f"kl={parts['kl'] / nb:7.2f}  cons={parts['cons'] / nb:6.4f}"
                  + ("  <-best" if epoch == best["epoch"] else ""), flush=True)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history.update(best_epoch=best["epoch"], best_cal_nll=best["cal"],
                   train_seconds=time.time() - t_start)
    return history
