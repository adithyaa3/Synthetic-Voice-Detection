"""Explainability on the development partition only (validation folds never used for fitting)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import N_OUTER_SPLITS, SEED
from .dataset import feature_columns
from .evaluate import eval_mask
from .features import feature_group
from .splits import group_kfold
from .tuning import fit_tuned


def _folds(table, setting):
    meta = table.drop(columns=feature_columns("extended"))
    rows = np.flatnonzero(eval_mask(table, setting, "dev"))
    return meta, group_kfold(meta, rows, N_OUTER_SPLITS, SEED)


def permutation_analysis(table: pd.DataFrame, setting: str, feature_set: str, model: str, params: dict,
                         aug: tuple, calibration: str | None, n_repeats: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-feature permutation importance (sklearn, ROC-AUC) and joint permutation of each feature
    GROUP (more meaningful when features are correlated), averaged over outer validation folds."""
    cols = feature_columns(feature_set)
    X, y = table[cols].to_numpy(), table.label.to_numpy()
    meta, folds = _folds(table, setting)
    groups = pd.Series([feature_group(c) for c in cols])
    per_feat, per_group = [], []
    rng = np.random.default_rng(SEED)
    for k, (tr, va) in enumerate(folds):
        m = fit_tuned(model, X, y, meta, tr, aug, seed=SEED, fixed_params=params, calibration=calibration)["model"]
        # n_jobs=1: prediction is cheap here, and avoiding a process pool removes a Windows failure mode
        # (BrokenProcessPool / "handle is invalid") seen when a long-running parent spawns workers late
        pi = permutation_importance(m, X[va], y[va], scoring="roc_auc", n_repeats=n_repeats, random_state=SEED, n_jobs=1)
        per_feat.append(pd.DataFrame({"feature": cols, "group": groups, "fold": k,
                                      "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}))
        base = roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])
        for g in groups.unique():
            idx = np.flatnonzero(groups.values == g)
            drops = []
            for _ in range(n_repeats):
                Xp = X[va].copy()
                perm = rng.permutation(len(va))
                Xp[:, idx] = Xp[perm][:, idx]
                drops.append(base - roc_auc_score(y[va], m.predict_proba(Xp)[:, 1]))
            per_group.append({"group": g, "fold": k, "n_features": int(len(idx)), "auc_drop": float(np.mean(drops)),
                              "baseline_auc": float(base)})
    pf = pd.concat(per_feat).groupby(["feature", "group"], as_index=False).agg(
        importance=("importance_mean", "mean"), importance_fold_std=("importance_mean", "std"))
    pg = pd.DataFrame(per_group).groupby(["group", "n_features"], as_index=False).agg(
        auc_drop=("auc_drop", "mean"), auc_drop_fold_std=("auc_drop", "std"))
    return pf.sort_values("importance", ascending=False), pg.sort_values("auc_drop", ascending=False)


def group_ablation(table: pd.DataFrame, setting: str, model: str, params: dict, aug: tuple,
                   calibration: str | None) -> pd.DataFrame:
    """Retrain with each extended feature group removed ('drop') or alone ('only'); same folds.
    Hyperparameters are fixed to the selected values (no re-tuning) to isolate the feature effect."""
    all_cols = feature_columns("extended")
    groups = pd.Series([feature_group(c) for c in all_cols])
    y = table.label.to_numpy()
    meta, folds = _folds(table, setting)
    configs = [("all extended features", list(all_cols)), ("baseline (original 80)", feature_columns("baseline"))]
    for g in groups.unique():
        configs.append((f"drop {g}", [c for c, gg in zip(all_cols, groups) if gg != g]))
        configs.append((f"only {g}", [c for c, gg in zip(all_cols, groups) if gg == g]))
    rows = []
    for name, cols in configs:
        X = table[cols].to_numpy()
        aucs = []
        for tr, va in folds:
            m = fit_tuned(model, X, y, meta, tr, aug, seed=SEED, fixed_params=params, calibration=calibration)["model"]
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        rows.append({"config": name, "n_features": len(cols), "roc_auc_mean": float(np.mean(aucs)),
                     "roc_auc_std": float(np.std(aucs, ddof=1))})
    return pd.DataFrame(rows)


def univariate_auc(table: pd.DataFrame, scopes: dict[str, np.ndarray]) -> pd.DataFrame:
    """Shortcut probe: ROC-AUC of every single feature, within each scope (folded so 0.5 = useless,
    1.0 = one feature alone separates the classes). Very high single-feature AUCs inside one data
    source point to simple acoustic differences (channel, codec, bandwidth) rather than subtle
    synthesis artefacts."""
    cols = feature_columns("extended")
    rows = []
    for scope, mask in scopes.items():
        d = table[mask]
        y = d.label.values
        for c in cols:
            a = roc_auc_score(y, d[c].values)
            rows.append({"scope": scope, "feature": c, "group": feature_group(c), "auc_folded": max(a, 1 - a),
                         "direction": "higher in AI" if a > 0.5 else "lower in AI",
                         "median_human": float(np.median(d.loc[d.label == 0, c])),
                         "median_ai": float(np.median(d.loc[d.label == 1, c])), "n": int(len(d))})
    return pd.DataFrame(rows).sort_values(["scope", "auc_folded"], ascending=[True, False])


def projections(table: pd.DataFrame, setting: str, feature_set: str, max_points: int = 2500) -> pd.DataFrame:
    rows = np.flatnonzero(eval_mask(table, setting, "dev"))
    if len(rows) > max_points:
        rows = np.sort(np.random.default_rng(SEED).choice(rows, max_points, replace=False))
    d = table.iloc[rows]
    X = StandardScaler().fit_transform(d[feature_columns(feature_set)].to_numpy())
    pca = PCA(n_components=2, random_state=SEED).fit(X)
    P = pca.transform(X)
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED).fit_transform(X)
    out = d[["file_id", "label", "class_name", "source", "generator", "group"]].copy()
    out["pca1"], out["pca2"] = P[:, 0], P[:, 1]
    out["tsne1"], out["tsne2"] = Z[:, 0], Z[:, 1]
    out.attrs["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
    return out
