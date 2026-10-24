"""Tuning / calibration behaviour on a small synthetic grouped problem."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from voicedetect.tuning import fit_tuned


def grouped_problem(n_groups=24, per=10, seed=0):
    rng = np.random.default_rng(seed)
    X, y, rows = [], [], []
    for g in range(n_groups):
        lab = g % 2
        shift = rng.normal(0, 0.6)
        for w in range(per):
            X.append(rng.normal(lab * 1.2 + shift, 1.0, size=6))
            y.append(lab)
            rows.append({"file_id": f"f{g}", "window_idx": w, "group": f"g{g}", "label": lab, "sha256": f"h{g}",
                         "variant": "orig"})
    return np.array(X), np.array(y), pd.DataFrame(rows)


def test_explicit_calibration_keeps_ranking_of_full_model():
    X, y, meta = grouped_problem()
    rows = np.arange(len(y))
    base = fit_tuned("logreg", X, y, meta, rows, fixed_params={"clf__C": 1.0}, calibration="none")["model"]
    cal = fit_tuned("logreg", X, y, meta, rows, fixed_params={"clf__C": 1.0}, calibration="sigmoid")
    Xt, yt, _ = grouped_problem(seed=1)
    pb, pc = base.predict_proba(Xt)[:, 1], cal["model"].predict_proba(Xt)[:, 1]
    assert cal["calibration"] == "sigmoid"
    assert roc_auc_score(yt, pb) == roc_auc_score(yt, pc)  # one monotone mapping: identical ranking


def test_calibration_with_augmented_rows_keeps_ranking_and_pickles(tmp_path):
    import joblib
    X, y, meta = grouped_problem()
    rng = np.random.default_rng(3)
    aug_meta = meta.assign(variant="aug_noise")
    X2 = np.vstack([X, X + rng.normal(0, 0.3, X.shape)])
    y2 = np.r_[y, y]
    meta2 = pd.concat([meta, aug_meta], ignore_index=True)
    rows = np.arange(len(y))  # evaluation rows = originals only
    base = fit_tuned("logreg", X2, y2, meta2, rows, ("aug_noise",), fixed_params={"clf__C": 1.0}, calibration="none")
    for kind in ("sigmoid", "isotonic"):
        cal = fit_tuned("logreg", X2, y2, meta2, rows, ("aug_noise",), fixed_params={"clf__C": 1.0}, calibration=kind)
        assert cal["n_train_rows"] == len(y2)  # augmented copies were used for training
        Xt, yt, _ = grouped_problem(seed=2)
        pb = base["model"].predict_proba(Xt)[:, 1]
        pc = cal["model"].predict_proba(Xt)[:, 1]
        if kind == "sigmoid":
            assert roc_auc_score(yt, pb) == roc_auc_score(yt, pc)
        assert np.all((pc >= 0) & (pc <= 1))
        joblib.dump(cal["model"], tmp_path / f"{kind}.joblib")
        again = joblib.load(tmp_path / f"{kind}.joblib")
        assert np.array_equal(again.predict_proba(Xt), cal["model"].predict_proba(Xt))


def test_margin_models_get_probabilities():
    X, y, meta = grouped_problem()
    res = fit_tuned("linear_svm", X, y, meta, np.arange(len(y)), fixed_params={"clf__C": 0.1})
    p = res["model"].predict_proba(X)[:, 1]
    assert res["calibration"] == "sigmoid" and np.all((p >= 0) & (p <= 1))


def test_inner_search_uses_group_folds_and_is_deterministic():
    X, y, meta = grouped_problem()
    a = fit_tuned("logreg", X, y, meta, np.arange(len(y)))
    b = fit_tuned("logreg", X, y, meta, np.arange(len(y)))
    assert a["best_params"] == b["best_params"]
    assert a["cv_results"] is not None and len(a["cv_results"]) == 5
