"""VUS-PR/ROC, AUC-PR, range/event/point F1, PA%K, conformal operating points."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)


def segments(labels: np.ndarray) -> np.ndarray:
    """Inclusive (start, end) of every non-zero run."""
    nz = (np.asarray(labels) != 0).astype(np.int8)
    if nz.sum() == 0:
        return np.zeros((0, 2), dtype=np.int64)
    d = np.diff(np.concatenate([[0], nz, [0]]))
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1], axis=1)


# VUS (port of TSB-UAD)

def _extend_positive_range(labels: np.ndarray, window: int) -> np.ndarray:
    out = labels.astype(np.float64).copy()
    if window <= 0:
        return out
    T = len(out)
    for s, e in segments(labels):
        x1 = np.arange(e + 1, min(e + window // 2 + 1, T))
        if x1.size:
            out[x1] += np.sqrt(1 - (x1 - e) / window)
        x2 = np.arange(max(s - window // 2, 0), s)
        if x2.size:
            out[x2] += np.sqrt(1 - (s - x2) / window)
    return np.minimum(1.0, out)


def _range_curves(labels_ext, pred, pred_sum, P, seg, device, chunk: int = 32):
    T = labels_ext.numel()
    TP = pred @ labels_ext
    P_new = (P + float(labels_ext.sum())) / 2.0
    recall = torch.clamp(TP / max(P_new, 1e-12), max=1.0)
    if len(seg) == 0:
        return 0.0, 0.0

    s_idx = torch.as_tensor(seg[:, 0], device=device, dtype=torch.long)
    e_idx = torch.as_tensor(seg[:, 1] + 1, device=device, dtype=torch.long)
    hits = []
    for lo in range(0, pred.shape[0], chunk):
        block = pred[lo: lo + chunk] * labels_ext.unsqueeze(0)
        cs = torch.cat([torch.zeros(block.shape[0], 1, device=device), block.cumsum(dim=1)], dim=1)
        hits.append((cs[:, e_idx] - cs[:, s_idx]) > 0)
    existence = torch.cat(hits, dim=0).double().mean(dim=1)

    tpr = (recall.double() * existence).cpu().numpy()
    prec = (TP / pred_sum.clamp(min=1e-12)).double().cpu().numpy()
    fpr = ((pred_sum - TP) / max(T - P_new, 1e-12)).double().cpu().numpy()

    K = len(tpr)
    tf_t, tf_f = np.zeros(K + 2), np.zeros(K + 2)
    tf_t[1: K + 1], tf_t[K + 1] = tpr, 1.0
    tf_f[1: K + 1], tf_f[K + 1] = fpr, 1.0
    pr = np.ones(K + 1)
    pr[1: K + 1] = prec

    auc = float(np.dot(tf_f[1:] - tf_f[:-1], (tf_t[1:] + tf_t[:-1]) / 2.0))
    ap = float(np.dot(tf_t[1:-1] - tf_t[:-2], (pr[1:] + pr[:-1]) / 2.0))
    return auc, ap


def vus(labels, scores, max_window: int, n_thresholds: int = 250, device=None):
    """-> (VUS-ROC, VUS-PR)."""
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0 or len(labels) < 3:
        return float("nan"), float("nan")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T, P = len(scores), float(labels.sum())
    s = torch.as_tensor(scores, device=device, dtype=torch.float32)
    order = torch.sort(s, descending=True).values
    thr = order[torch.linspace(0, T - 1, n_thresholds, device=device).long()]
    pred = (s.unsqueeze(0) >= thr.unsqueeze(1)).float()
    pred_sum = pred.sum(dim=1)

    aucs, aps = [], []
    for w in range(max_window + 1):
        ext = _extend_positive_range(labels, w)
        auc, ap = _range_curves(torch.as_tensor(ext, device=device, dtype=torch.float32),
                                pred, pred_sum, P, segments(ext), device)
        aucs.append(auc)
        aps.append(ap)
    return float(np.mean(aucs)), float(np.mean(aps))


# point / range / event

def prf(labels, pred) -> dict:
    return {"precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0))}


def apply_pa_k(pred, labels, k: float) -> np.ndarray:
    adjusted = pred.copy()
    for s, e in segments(labels):
        seg_pred = pred[s: e + 1]
        if seg_pred.any() and seg_pred.mean() * 100.0 >= k:
            adjusted[s: e + 1] = 1
    return adjusted


def _threshold_grid(scores, n: int = 200) -> np.ndarray:
    return np.unique(np.percentile(scores, np.linspace(0.0, 100.0, n)))


def best_over_thresholds(labels, scores, fn, n: int = 200) -> dict:
    """Oracle: best F1 of fn(labels, pred) over a threshold sweep."""
    best = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "threshold": float("nan")}
    for thr in _threshold_grid(scores, n):
        r = fn(labels, (scores >= thr).astype(int))
        if r["f1"] > best["f1"]:
            best = {**r, "threshold": float(thr)}
    return best


def event_prf(labels, pred) -> dict:
    true_seg, pred_seg = segments(labels), segments(pred)
    if len(true_seg) == 0 or len(pred_seg) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    rec = sum(1 for s, e in true_seg if pred[s: e + 1].any()) / len(true_seg)
    prec = sum(1 for s, e in pred_seg if labels[s: e + 1].any()) / len(pred_seg)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def range_prf(labels, pred, alpha: float = 0.0) -> dict:
    """Tatbul et al. (NeurIPS 2018), flat bias."""
    true_seg, pred_seg = segments(labels), segments(pred)
    if len(true_seg) == 0 or len(pred_seg) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    def overlap(a, b) -> int:
        return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)

    rec = 0.0
    for r in true_seg:
        omega = sum(overlap(r, p) for p in pred_seg) / (r[1] - r[0] + 1)
        exists = 1.0 if any(overlap(r, p) > 0 for p in pred_seg) else 0.0
        rec += alpha * exists + (1 - alpha) * omega
    rec /= len(true_seg)
    prec = sum(sum(overlap(p, r) for r in true_seg) / (p[1] - p[0] + 1) for p in pred_seg)
    prec /= len(pred_seg)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def event_recall(labels, pred) -> float:
    seg = segments(labels)
    return sum(1 for s, e in seg if pred[s: e + 1].any()) / len(seg) if len(seg) else float("nan")


def false_alarm_rate(labels, pred) -> float:
    neg = labels == 0
    return float(pred[neg].mean()) if neg.any() else float("nan")


def evaluate(labels, scores, conformal_thresholds: dict | None = None,
             vus_window: int = 100, pa_k: float = 20.0) -> dict:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    has_pos = labels.sum() > 0

    vus_roc, vus_pr = vus(labels, scores, max_window=vus_window)
    out = {
        "n_points": int(len(labels)),
        "base_rate": float(labels.mean()),
        "n_events": int(len(segments(labels))),
        "auc_pr": float(average_precision_score(labels, scores)) if has_pos else float("nan"),
        "auc_roc": float(roc_auc_score(labels, scores)) if has_pos else float("nan"),
        "vus_pr": vus_pr,
        "vus_roc": vus_roc,
    }

    # oracle (test-label thresholds), reported for comparability only
    b = best_over_thresholds(labels, scores, prf)
    out.update(best_f1=b["f1"], best_f1_precision=b["precision"], best_f1_recall=b["recall"])
    pak = best_over_thresholds(labels, scores, lambda y, p: prf(y, apply_pa_k(p, y, pa_k)))
    out[f"pa{int(pa_k)}_f1"] = pak["f1"]
    out["pa_f1"] = best_over_thresholds(labels, scores, lambda y, p: prf(y, apply_pa_k(p, y, 0.0)))["f1"]
    out["event_f1"] = best_over_thresholds(labels, scores, event_prf)["f1"]
    out["range_f1"] = best_over_thresholds(labels, scores, range_prf)["f1"]
    out["trivial_f1"] = 2 * out["base_rate"] / (1 + out["base_rate"]) if out["base_rate"] else 0.0

    # deployable: thresholds from unlabeled normal data
    if conformal_thresholds:
        best_alpha, best_conf = float("nan"), -1.0
        for alpha, thr in sorted(conformal_thresholds.items()):
            pred = (scores >= thr).astype(int)
            m = prf(labels, pred)
            tag = f"a{alpha:g}"
            out.update({f"{tag}_precision": m["precision"], f"{tag}_recall": m["recall"],
                        f"{tag}_f1": m["f1"], f"{tag}_fpr": false_alarm_rate(labels, pred),
                        f"{tag}_event_recall": event_recall(labels, pred)})
            if m["f1"] > best_conf:
                best_alpha, best_conf = float(alpha), m["f1"]
        out["f1_conformal_best"] = best_conf
        out["f1_conformal_alpha"] = best_alpha
    return out
