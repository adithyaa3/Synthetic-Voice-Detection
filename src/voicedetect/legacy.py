"""Reproduce the ORIGINAL pipeline (scripts/train.py as first committed) and show why its 100 %
test accuracy is not a valid estimate. Uses the original feature function verbatim: first 4 s,
22 050 Hz, 40 MFCC mean+std, no trimming; original split and original SVC."""
from __future__ import annotations

import json
import os
import warnings

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import FEATURES_DIR, REPORTS_DIR, SEED, USER_AI_DIR, USER_HUMAN_DIR
from .manifest import common_voice_runs, parse_user_ai_name
from .metrics import binary_metrics


def original_extract_features(audio_path: str) -> np.ndarray:
    """Verbatim copy of the original feature function (kept only for this reproduction)."""
    y, sr = librosa.load(audio_path, sr=22050, duration=4)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.concatenate((np.mean(mfcc, axis=1), np.std(mfcc, axis=1)))


def _original_dataset() -> pd.DataFrame:
    cache = FEATURES_DIR / "legacy_original_features.parquet"
    rows = []
    # identical listing logic and order to the original script
    for folder, label in ((USER_HUMAN_DIR, 0), (USER_AI_DIR, 1)):
        for f in os.listdir(folder):
            if f.lower().endswith((".wav", ".mp3")):
                rows.append({"file": f, "label": label, "path": str(folder / f)})
    df = pd.DataFrame(rows)
    if cache.exists():
        old = pd.read_parquet(cache)
        if list(old.file) == list(df.file):
            return old
    X = np.vstack([original_extract_features(p) for p in df.path])
    feats = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df = pd.concat([df.drop(columns="path"), feats], axis=1)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def _original_model():
    # probability=True is kept on purpose (verbatim reproduction of the original script). scikit-learn 1.9
    # deprecates it (removal planned for 1.11; requirements.txt pins 1.9.0). run() silences only that warning.
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=SEED))


def run() -> dict:
    with warnings.catch_warnings():  # the warning is raised in fit(), so the filter wraps the whole step
        warnings.filterwarnings("ignore", message="The `probability` parameter was deprecated", category=FutureWarning)
        return _run()


def _run() -> dict:
    df = _original_dataset()
    X = df.filter(regex=r"^f\d+$").to_numpy()
    y = df.label.to_numpy()
    runs = common_voice_runs(df.loc[df.label == 0, "file"].tolist())
    df["group"] = [runs[f] if l == 0 else f"vc_src_{parse_user_ai_name(f)[0]}" for f, l in zip(df.file, df.label)]

    # (a) the original single split, exactly as in scripts/train.py
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    model = _original_model().fit(X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    pred = model.predict(X[te])
    orig = binary_metrics(y[te], p)
    orig["accuracy_from_predict"] = float(np.mean(pred == y[te]))
    orig["majority_class_accuracy"] = float(max(np.mean(y[te] == 0), np.mean(y[te] == 1)))
    tr_groups = set(df.group.values[tr])
    te_ai = [i for i in te if y[i] == 1]
    leaked = [df.file.values[i] for i in te_ai if df.group.values[i] in tr_groups]
    leaked_h = [i for i in te if y[i] == 0 and df.group.values[i] in tr_groups]

    # (b) same features + model, group-aware 5-fold CV (repeated with 3 seeds)
    cv_rows = []
    for rep in range(3):
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + rep)
        for k, (a, b) in enumerate(cv.split(X, y, df.group.values)):
            m = _original_model().fit(X[a], y[a])
            cv_rows.append({"repeat": rep, "fold": k, **binary_metrics(y[b], m.predict_proba(X[b])[:, 1])})
    cv_df = pd.DataFrame(cv_rows)

    # (c) leave-one-source-speaker-out (AI) with human run-groups rotated over 8 folds
    srcs = sorted(df.loc[df.label == 1, "group"].unique())
    hum = np.random.default_rng(SEED).permutation(sorted(df.loc[df.label == 0, "group"].unique()))
    hfold = {g: i % len(srcs) for i, g in enumerate(hum)}
    loso = []
    for k, s in enumerate(srcs):
        test = ((df.label == 1) & (df.group == s)) | ((df.label == 0) & df.group.map(hfold).eq(k))
        m = _original_model().fit(X[~test.values], y[~test.values])
        pp = m.predict_proba(X[test.values])[:, 1]
        mm = binary_metrics(y[test.values], pp)
        loso.append({"held_out_source": s.replace("vc_src_", ""), **mm})
    loso_df = pd.DataFrame(loso)

    def ms(d, c):
        return {"mean": float(d[c].mean()), "std": float(d[c].std())}

    out = {
        "description": "Original scripts/train.py pipeline reproduced (features, split, model unchanged).",
        "n_files": int(len(y)), "n_human": int((y == 0).sum()), "n_ai": int((y == 1).sum()),
        "original_split": {**orig, "n_test": int(len(te)), "n_test_ai": int(len(te_ai)),
                           "test_ai_files": [df.file.values[i] for i in te_ai],
                           "test_ai_files_with_same_source_recording_in_train": leaked,
                           "test_human_clips_whose_run_group_is_in_train": int(len(leaked_h))},
        "group_cv_5fold_x3": {c: ms(cv_df, c) for c in ["accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc", "pr_auc"]},
        "leave_one_source_out": {c: ms(loso_df, c) for c in ["accuracy", "balanced_accuracy", "recall", "specificity", "roc_auc"]},
        "leave_one_source_out_per_source": loso_df[["held_out_source", "recall", "specificity", "roc_auc", "balanced_accuracy"]].to_dict("records"),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "legacy_reproduction.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    cv_df.to_csv(REPORTS_DIR / "legacy_group_cv_folds.csv", index=False)
    return out
