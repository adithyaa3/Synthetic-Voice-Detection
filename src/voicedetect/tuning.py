"""Hyperparameter tuning on TRAINING folds only (inner group-aware CV), then refit."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from .config import N_INNER_SPLITS, SEED
from .models import NEEDS_PLATT, PARAM_GRIDS, make_estimator
from .splits import group_kfold, with_augmented


def inner_splits(meta: pd.DataFrame, train_eval_rows: np.ndarray, train_all: np.ndarray,
                 aug_variants: tuple, n_inner: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-aware inner folds expressed as positions into ``train_all``."""
    pos = {r: i for i, r in enumerate(train_all)}
    out = []
    for itr, iva in group_kfold(meta, train_eval_rows, n_inner, seed):
        itr_all = with_augmented(meta, itr, aug_variants)
        out.append((np.array([pos[r] for r in itr_all]), np.array([pos[r] for r in iva])))
    return out


def fit_tuned(name: str, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, train_eval_rows: np.ndarray,
              aug_variants: tuple = (), n_inner: int = N_INNER_SPLITS, seed: int = SEED,
              n_jobs: int = -1, fixed_params: dict | None = None, calibration: str | None = None) -> dict:
    """Tune ``name`` by inner CV (ROC-AUC) on the training rows and refit on all training rows.
    calibration=None   : margin models get Platt scaling, probabilistic models are left as they are.
    calibration="none" : same as None for margin models (they need a probability mapping).
    calibration="sigmoid"/"isotonic": any model is wrapped in CalibratedClassifierCV fit on the same
    group-aware inner folds (each fold's calibrator only sees that fold's held-out training groups)."""
    t0 = time.time()
    train_all = with_augmented(meta, train_eval_rows, aug_variants)
    Xtr, ytr = X[train_all], y[train_all]
    splits = inner_splits(meta, train_eval_rows, train_all, aug_variants, n_inner, seed)
    est = make_estimator(name, seed)
    grid = PARAM_GRIDS[name] if fixed_params is None else {}
    if fixed_params:
        est.set_params(**fixed_params)
    cv_results = None
    if grid:
        gs = GridSearchCV(est, grid, scoring="roc_auc", cv=splits, n_jobs=n_jobs, refit=True, error_score="raise")
        gs.fit(Xtr, ytr)
        best, best_params = gs.best_estimator_, gs.best_params_
        cv_results = pd.DataFrame(gs.cv_results_)
    else:
        best, best_params = clone(est).fit(Xtr, ytr), dict(fixed_params or {})
    model, method = best, "none"
    if calibration in ("sigmoid", "isotonic"):
        method = calibration
    elif name in NEEDS_PLATT:
        method = "sigmoid"
    if method != "none":
        # Explicitly requested post-hoc calibration fits ONE monotone mapping on inner out-of-fold scores
        # of the model refit on all training rows, so its ranking / ROC-AUC is kept. Without augmentation
        # this is CalibratedClassifierCV(ensemble=False); with augmented copies (which are never validated,
        # so the inner folds do not partition the rows) the same idea is implemented by oof_calibrated().
        # The default Platt wrapper for margin models in the model comparison averages inner-fold models.
        if calibration in ("sigmoid", "isotonic") and aug_variants:
            model = oof_calibrated(best, Xtr, ytr, splits, method)
        else:
            ensemble = calibration not in ("sigmoid", "isotonic")
            model = CalibratedClassifierCV(clone(best), method=method, cv=splits, ensemble=ensemble).fit(Xtr, ytr)
    return {"model": model, "best_params": best_params, "cv_results": cv_results, "calibration": method,
            "fit_seconds": time.time() - t0, "n_train_rows": int(len(train_all))}


class MappedProbability(BaseEstimator, ClassifierMixin):
    """A fitted classifier followed by a fitted monotone probability mapping."""

    def __init__(self, base, mapper, kind: str):
        self.base, self.mapper, self.kind = base, mapper, kind
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = np.clip(self.base.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
        q = (self.mapper.predict_proba(np.log(p / (1 - p))[:, None])[:, 1] if self.kind == "sigmoid"
             else self.mapper.predict(p))
        q = np.clip(q, 0.0, 1.0)
        return np.column_stack([1 - q, q])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def oof_calibrated(fitted_base, X, y, splits, kind: str) -> MappedProbability:
    """Out-of-fold scores of the ORIGINAL training windows (inner validation folds) -> one monotone
    mapping (Platt on the log-odds, or isotonic) applied on top of the model fitted on all rows."""
    oof = np.full(len(y), np.nan)
    for itr, iva in splits:
        oof[iva] = clone(fitted_base).fit(X[itr], y[itr]).predict_proba(X[iva])[:, 1]
    m = ~np.isnan(oof)
    p = np.clip(oof[m], 1e-6, 1 - 1e-6)
    if kind == "sigmoid":
        mapper = LogisticRegression(C=1e6, max_iter=1000).fit(np.log(p / (1 - p))[:, None], y[m])
        if mapper.coef_[0, 0] <= 0:
            raise RuntimeError("Platt mapping is not increasing; refusing a calibration that would invert scores")
    else:
        mapper = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True).fit(p, y[m])
    return MappedProbability(fitted_base, mapper, kind)
