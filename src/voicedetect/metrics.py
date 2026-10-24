"""Metric helpers (all verified against hand-computed examples in tests/test_metrics.py)."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as skm


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]:
    sum_b (n_b / N) * |mean(y in b) - mean(p in b)|."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """y: 1 = AI (positive class), p: predicted probability of AI."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    out = {
        "n": int(len(y)), "n_ai": int(y.sum()), "n_human": int((1 - y).sum()),
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred) if both else np.nan,
        "precision": skm.precision_score(y, pred, zero_division=0),
        "recall": skm.recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": skm.f1_score(y, pred, zero_division=0),
        "roc_auc": skm.roc_auc_score(y, p) if both else np.nan,
        "pr_auc": skm.average_precision_score(y, p) if both else np.nan,
        "brier": skm.brier_score_loss(y, p),
        "log_loss": skm.log_loss(y, p, labels=[0, 1]),
        "ece": ece(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def cluster_bootstrap_ci(y, p, groups, metric=skm.roc_auc_score, n_boot: int = 1000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole GROUPS with replacement (respects the
    dependence between windows of one file / clips of one speaker)."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, p)), float(lo), float(hi)


def paired_bootstrap_auc_diff(y, pa, pb, groups, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) on the same rows, with a 95 % CI from resampling whole groups."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) == 2:
            diffs.append(skm.roc_auc_score(y[idx], pa[idx]) - skm.roc_auc_score(y[idx], pb[idx]))
    point = skm.roc_auc_score(y, pa) - skm.roc_auc_score(y, pb)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def expected_cost(y, p, threshold, c_fp: float, c_fn: float) -> float:
    """Average cost per clip; c_fp = false alarm on a real person, c_fn = missed synthetic voice."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    return float((c_fp * fp + c_fn * fn) / len(y))


def choose_threshold(y, p, c_fp: float, c_fn: float, grid=None) -> tuple[float, float]:
    """Threshold minimising expected cost on VALIDATION predictions; ties -> closest to 0.5."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    costs = np.array([expected_cost(y, p, t, c_fp, c_fn) for t in grid])
    best = np.flatnonzero(np.isclose(costs, costs.min()))
    t = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return float(t), float(costs.min())
