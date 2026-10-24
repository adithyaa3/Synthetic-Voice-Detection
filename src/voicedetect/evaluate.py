"""Nested, group-aware cross-validation on the development partition, plus summaries.

Settings (which rows are evaluated):
  user               : user data only, audio as stored
  user_codec_matched : user data only; AI windows re-encoded to the exact Common Voice format
                       (32 kHz, 48 kbps MP3) so file format can no longer separate the classes
  combined           : user + supplementary (ASVspoof 2019 LA subset)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


from .config import N_INNER_SPLITS, N_OUTER_SPLITS, SEED
from .dataset import feature_columns
from .metrics import binary_metrics, cluster_bootstrap_ci, paired_bootstrap_auc_diff
from .splits import assert_no_leakage, group_kfold, with_augmented
from .tuning import fit_tuned

SETTINGS = ("user", "user_codec_matched", "combined")
SETTING_LABELS = {
    "user": "User data only",
    "user_codec_matched": "User data, codec-matched control",
    "combined": "User + supplementary (ASVspoof 2019 LA)",
}
METRIC_COLS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
               "roc_auc", "pr_auc", "brier", "log_loss", "ece"]


def eval_mask(table: pd.DataFrame, setting: str, partition: str) -> np.ndarray:
    part = table.partition.eq(partition)
    if setting == "user":
        m = table.origin.eq("user") & table.variant.eq("orig")
    elif setting == "user_codec_matched":
        m = table.origin.eq("user") & ((table.label.eq(0) & table.variant.eq("orig")) |
                                       (table.label.eq(1) & table.variant.eq("mp3cv")))
    elif setting == "combined":
        m = table.variant.eq("orig")
    else:
        raise ValueError(setting)
    return (part & m).values


def eval_variants(setting: str) -> tuple:
    return ("orig", "mp3cv") if setting == "user_codec_matched" else ("orig",)


def _check(meta, tr, va, setting):
    """Leakage guard; the codec-matched setting evaluates on mp3cv rows by design."""
    if setting == "user_codec_matched":
        m = meta.copy()
        m["variant"] = "orig"
        assert_no_leakage(m, tr, va)
    else:
        assert_no_leakage(meta, tr, va)


def counterpart_rows(meta: pd.DataFrame, rows: np.ndarray, variant: str) -> np.ndarray:
    """Row positions of ``variant`` copies of the given windows (same file, same window)."""
    v = meta[meta.variant == variant]
    lookup = dict(zip(zip(v.file_id.values, v.window_idx.values), np.flatnonzero(meta.variant.values == variant)))
    return np.array([lookup[(f, w)] for f, w in zip(meta.file_id.values[rows], meta.window_idx.values[rows])])


# validation-time perturbations: variant name in the table -> column suffix in the fold metrics
EVAL_PERTURBATIONS = {"mp3cv": "mp3", "ev_white20": "white20", "ev_room20": "room20", "ev_mic": "mic"}


def nested_cv(table: pd.DataFrame, setting: str, feature_set: str, model_names: list[str],
              aug_variants: tuple = (), n_outer: int = N_OUTER_SPLITS, n_inner: int = N_INNER_SPLITS,
              seed: int = SEED, n_jobs: int = -1, calibration: str | None = None,
              fold_params: dict | None = None, perturbations: tuple = ("mp3cv",), log=print) -> dict:
    """Outer StratifiedGroupKFold on the dev partition; each model tuned on the outer-training part
    only. Validation windows are also scored in perturbed versions (``perturbations``; by default the
    MP3 / Common Voice format), giving dev-only robustness measures ("*_mp3", "*_white20", ...)
    without touching the test set. ``fold_params`` ({fold: params}) reuses hyperparameters already
    selected by an identical inner search on the same fold."""
    meta = table.drop(columns=feature_columns("extended"))
    X = table[feature_columns(feature_set)].to_numpy()
    y = table.label.to_numpy()
    rows = np.flatnonzero(eval_mask(table, setting, "dev"))
    outer = group_kfold(meta, rows, n_outer, seed)
    aug_tag = "+".join(aug_variants) or "none"
    calib_tag = calibration or "default"
    perts = [] if setting == "user_codec_matched" else [v for v in perturbations if (meta.variant == v).any()]
    fold_rows, oof_rows, grids = [], [], []
    for k, (tr, va) in enumerate(outer):
        _check(meta, with_augmented(meta, tr, aug_variants), va, setting)
        va_pert = {v: counterpart_rows(meta, va, v) for v in perts}
        for name in model_names:
            fixed = fold_params.get(k) if fold_params else None
            res = fit_tuned(name, X, y, meta, tr, aug_variants, n_inner, seed, n_jobs, fixed_params=fixed,
                            calibration=calibration)
            p = res["model"].predict_proba(X[va])[:, 1]
            m = binary_metrics(y[va], p)
            extra, p_extra = {}, {}
            for v, rows_v in va_pert.items():
                pv = res["model"].predict_proba(X[rows_v])[:, 1]
                mv = binary_metrics(y[va], pv)
                sfx = EVAL_PERTURBATIONS.get(v, v)
                extra |= {f"{c}_{sfx}": mv[c] for c in ("roc_auc", "balanced_accuracy", "recall", "specificity")}
                p_extra[f"p_ai_{sfx}"] = pv
            if "p_ai_mp3" not in p_extra:
                p_extra["p_ai_mp3"] = np.full(len(va), np.nan)
            fold_rows.append({"setting": setting, "feature_set": feature_set, "model": name, "fold": k,
                              "augment": aug_tag, "calibration": calib_tag, **m, **extra,
                              "best_params": str(res["best_params"]), "fit_seconds": res["fit_seconds"],
                              "n_train_rows": res["n_train_rows"], "n_val_groups": int(meta.iloc[va].group.nunique())})
            oof_rows.append(pd.DataFrame({"setting": setting, "feature_set": feature_set, "model": name, "fold": k,
                                          "augment": aug_tag, "calibration": calib_tag, "row": va,
                                          "file_id": meta.file_id.values[va], "window_idx": meta.window_idx.values[va],
                                          "group": meta.group.values[va], "source": meta.source.values[va],
                                          "generator": meta.generator.values[va], "label": y[va], "p_ai": p,
                                          **p_extra}))
            if res["cv_results"] is not None and name == "rbf_svm":
                g = res["cv_results"][["param_clf__C", "param_clf__gamma", "mean_test_score", "std_test_score"]].copy()
                g.columns = ["C", "gamma", "mean_auc", "std_auc"]
                g.insert(0, "fold", k)
                g.insert(0, "feature_set", feature_set)
                g.insert(0, "setting", setting)
                grids.append(g)
            log(f"  [{setting}/{feature_set}] fold {k} {name:<16} AUC={m['roc_auc']:.3f} "
                f"bAcc={m['balanced_accuracy']:.3f} ({res['fit_seconds']:.1f}s) {res['best_params']}")
    return {"folds": pd.DataFrame(fold_rows), "oof": pd.concat(oof_rows, ignore_index=True),
            "grid": pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()}


def summarize(folds: pd.DataFrame) -> pd.DataFrame:
    keys = ["setting", "feature_set", "model", "augment", "calibration"]
    sfx = tuple(f"_{s}" for s in EVAL_PERTURBATIONS.values())
    cols = METRIC_COLS + [c for c in folds.columns if c.endswith(sfx)]
    agg = folds.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_folds"] = folds.groupby(keys).size()
    agg["fit_seconds_mean"] = folds.groupby(keys).fit_seconds.mean()
    return agg.reset_index()


def paired_comparison(folds: pd.DataFrame, oof: pd.DataFrame, setting: str, feature_set: str,
                      reference: str, metric: str = "roc_auc", n_boot: int = 1000, seed: int = SEED,
                      augment: str = "none", calibration: str = "default") -> pd.DataFrame:
    """Reference model vs every other model on the SAME folds: per-fold differences, Wilcoxon test
    (note: with 5 folds the smallest attainable two-sided p is 0.0625) and a paired group-bootstrap
    CI of the pooled out-of-fold AUC difference."""
    sel = lambda d: d[(d.setting == setting) & (d.feature_set == feature_set) &  # noqa: E731
                      (d.augment == augment) & (d.calibration == calibration)]
    f, o = sel(folds), sel(oof)
    ref_f = f[f.model == reference].set_index("fold")[metric]
    ref_o = o[o.model == reference].set_index("row")
    rows = []
    for name in sorted(set(f.model) - {reference}):
        oth_f = f[f.model == name].set_index("fold")[metric]
        d = (ref_f - oth_f).dropna()
        try:
            wp = float(stats.wilcoxon(d.values).pvalue) if len(d) >= 2 and np.any(d.values != 0) else np.nan
        except ValueError:
            wp = np.nan
        oth_o = o[o.model == name].set_index("row").loc[ref_o.index]
        point, lo, hi = paired_bootstrap_auc_diff(ref_o.label.values, ref_o.p_ai.values, oth_o.p_ai.values,
                                                  ref_o.group.values, n_boot=n_boot, seed=seed)
        rows.append({"setting": setting, "feature_set": feature_set, "reference": reference, "other": name,
                     f"mean_fold_diff_{metric}": float(d.mean()), f"std_fold_diff_{metric}": float(d.std()),
                     "folds_reference_better": int((d > 0).sum()), "n_folds": int(len(d)), "wilcoxon_p": wp,
                     "pooled_auc_diff": point, "pooled_auc_diff_ci_low": lo, "pooled_auc_diff_ci_high": hi})
    return pd.DataFrame(rows)


def pooled_ci(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["setting", "feature_set", "model", "augment", "calibration"]
    for kv, d in oof.groupby(keys):
        if d.label.nunique() < 2:
            continue
        est, lo, hi = cluster_bootstrap_ci(d.label.values, d.p_ai.values, d.group.values, n_boot=1000, seed=SEED)
        rows.append({**dict(zip(keys, kv)), "pooled_oof_auc": est, "ci_low": lo, "ci_high": hi,
                     "n_groups": int(d.group.nunique())})
    return pd.DataFrame(rows)


def by_source(oof: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Per-source / per-generator detection rates from out-of-fold predictions."""
    rows = []
    keys = ["setting", "feature_set", "model", "augment", "calibration", "source", "generator"]
    for kv, d in oof.groupby(keys):
        pred = d.p_ai.values >= threshold
        rows.append(dict(zip(keys, kv), n=int(len(d)), label=int(d.label.iloc[0]),
                         rate_flagged_ai=float(pred.mean()), mean_p_ai=float(d.p_ai.mean())))
    return pd.DataFrame(rows)
