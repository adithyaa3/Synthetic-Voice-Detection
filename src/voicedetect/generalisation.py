"""Generalisation experiments with the FINAL configuration (fixed hyperparameters, no re-tuning).

These run AFTER model selection and use all partitions; they never influence any choice.
* LOSO  : leave-one-source-speaker-out on user data (strictly group-safe).
* LOTO  : leave-one-target-voice-out on user data (RELAXED: other conversions of the same source
          recording stay in training, because every source was converted into every target voice).
* LOGO  : leave-one-generator-out on user + supplementary data (user voice conversion + 19 ASVspoof
          systems); the held-out generator's speakers are also removed from training.
* Cross-dataset: train on one data source, test on the other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import SEED
from .dataset import feature_columns
from .metrics import binary_metrics
from .splits import assert_no_leakage, leave_one_out_by
from .tuning import fit_tuned


def _fit_predict(cfg: dict, X, y, meta, tr, te):
    m = fit_tuned(cfg["model"], X, y, meta, tr, cfg["aug"], seed=SEED, fixed_params=cfg["params"],
                  calibration=cfg["calibration"])["model"]
    return m.predict_proba(X[te])[:, 1]


def _run_loo(cfg, table, rows, key, strict, label):
    from .manifest import generator_descriptions
    desc = generator_descriptions() if key == "generator" else {}
    meta = table.drop(columns=feature_columns("extended"))
    X, y = table[feature_columns(cfg["feature_set"])].to_numpy(), table.label.to_numpy()
    out = []
    for value, tr, te in leave_one_out_by(meta, rows, key, strict=strict, seed=SEED):
        if strict:
            assert_no_leakage(meta, tr, te)
        p = _fit_predict(cfg, X, y, meta, tr, te)
        yt = y[te]
        ai = yt == 1
        m = binary_metrics(yt, p, cfg["threshold"])
        out.append({"experiment": label, "held_out": value, "held_out_description": desc.get(value, ""),
                    "strict_groups": strict,
                    "n_test_ai": int(ai.sum()), "n_test_human": int((~ai).sum()),
                    "n_train": int(len(tr)), "recall_ai": m["recall"], "specificity_human": m["specificity"],
                    "balanced_accuracy": m["balanced_accuracy"], "roc_auc": m["roc_auc"],
                    "mean_p_ai_heldout_ai": float(p[ai].mean())})
    return out


def run(cfg: dict, table: pd.DataFrame) -> pd.DataFrame:
    orig = table.variant.eq("orig").values
    user = orig & table.origin.eq("user").values
    rows_user = np.flatnonzero(user)
    rows_all = np.flatnonzero(orig)
    results = []
    results += _run_loo(cfg, table, rows_user, "speaker", True, "LOSO: leave one source speaker out (user data)")
    results += _run_loo(cfg, table, rows_user, "target_voice", False, "LOTO: leave one target voice out (user data, relaxed)")
    results += _run_loo(cfg, table, rows_all, "generator", True, "LOGO: leave one generator out (user + supplementary)")
    return pd.DataFrame(results)


def cross_dataset(cfg: dict, table: pd.DataFrame) -> pd.DataFrame:
    meta = table.drop(columns=feature_columns("extended"))
    X, y = table[feature_columns(cfg["feature_set"])].to_numpy(), table.label.to_numpy()
    orig = table.variant.eq("orig").values
    user = np.flatnonzero(orig & table.origin.eq("user").values)
    supp = np.flatnonzero(orig & table.origin.eq("public").values)
    rows = []
    for name, tr, te in [("train user -> test ASVspoof", user, supp), ("train ASVspoof -> test user", supp, user)]:
        assert_no_leakage(meta, tr, te)
        p = _fit_predict(cfg, X, y, meta, tr, te)
        m = binary_metrics(y[te], p, cfg["threshold"])
        rows.append({"experiment": name, "n_train": int(len(tr)), "n_test": int(len(te)), "roc_auc": m["roc_auc"],
                     "balanced_accuracy": m["balanced_accuracy"], "recall_ai": m["recall"],
                     "specificity_human": m["specificity"]})
    return pd.DataFrame(rows)


def learning_curve(cfg: dict, table: pd.DataFrame, folds,
                   fractions=(0.1, 0.25, 0.5, 0.75, 1.0), repeats: int = 3) -> pd.DataFrame:
    """Train on a fraction of the training GROUPS (per class) of each outer fold; score the full
    validation fold. Shows whether more independent data would likely help."""
    meta = table.drop(columns=feature_columns("extended"))
    X, y = table[feature_columns(cfg["feature_set"])].to_numpy(), table.label.to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for k, (tr, va) in enumerate(folds):
        trm = meta.iloc[tr]
        for frac in fractions:
            for r in range(repeats if frac < 1 else 1):
                keep = []
                for lab in (0, 1):
                    gs = np.array(sorted(trm.loc[trm.label == lab, "group"].unique()))
                    n = max(1, int(round(frac * len(gs))))
                    keep += list(rng.choice(gs, n, replace=False))
                sub = tr[trm.group.isin(keep).values]
                if len(np.unique(y[sub])) < 2:
                    continue
                p = _fit_predict(cfg, X, y, meta, sub, va)
                rows.append({"fold": k, "fraction_of_training_groups": frac, "repeat": r,
                             "n_train_windows": int(len(sub)), "n_train_groups": int(len(keep)),
                             "roc_auc": roc_auc_score(y[va], p)})
    return pd.DataFrame(rows)
