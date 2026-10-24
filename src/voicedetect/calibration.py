"""Calibration diagnostics and decision-threshold analysis on VALIDATION (out-of-fold) predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import binary_metrics, choose_threshold, ece, expected_cost


def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({"bin": b, "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "count": int(m.sum()),
                     "mean_predicted": float(p[m].mean()) if m.any() else np.nan,
                     "fraction_ai": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def threshold_table(y, p, c_fp: float, c_fn: float, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.02, 0.981, 0.01), 2) if grid is None else grid
    rows = []
    for t in grid:
        m = binary_metrics(y, p, threshold=float(t))
        rows.append({"threshold": float(t), "precision": m["precision"], "recall": m["recall"],
                     "specificity": m["specificity"], "false_alarm_rate": 1 - m["specificity"],
                     "f1": m["f1"], "balanced_accuracy": m["balanced_accuracy"], "accuracy": m["accuracy"],
                     "expected_cost": expected_cost(y, p, t, c_fp, c_fn),
                     "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"]})
    return pd.DataFrame(rows)


def inconclusive_stats(y, p, threshold: float, halfwidth: float) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    band = np.abs(p - threshold) < halfwidth
    out = {"threshold": threshold, "halfwidth": halfwidth, "band_low": max(0.0, threshold - halfwidth),
           "band_high": min(1.0, threshold + halfwidth), "fraction_inconclusive": float(band.mean())}
    if (~band).any():
        dec = p[~band] >= threshold
        out["accuracy_outside_band"] = float(np.mean(dec == y[~band]))
        out["n_decided"] = int((~band).sum())
    if band.any():
        out["accuracy_inside_band_if_forced"] = float(np.mean((p[band] >= threshold) == y[band]))
    return out


def decide(y, p, c_fp: float, c_fn: float, halfwidth: float) -> dict:
    """Cost-minimising threshold on validation predictions plus the inconclusive band stats."""
    t, cost = choose_threshold(y, p, c_fp, c_fn)
    return {"threshold": t, "expected_cost_at_threshold": cost,
            "expected_cost_at_0.5": expected_cost(y, p, 0.5, c_fp, c_fn),
            "cost_false_alarm": c_fp, "cost_miss": c_fn,
            "metrics_at_threshold": binary_metrics(y, p, t), "metrics_at_0.5": binary_metrics(y, p, 0.5),
            "inconclusive": inconclusive_stats(y, p, t, halfwidth), "ece": ece(y, p)}
