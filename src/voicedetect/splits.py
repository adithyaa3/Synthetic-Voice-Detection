"""Group-aware splitting and leakage guards.

Rows are analysis windows. Every window inherits the group of its file, and augmented copies
(variant != "orig") inherit the group of their source window. Validation/test indices only ever
contain "orig" rows; augmented rows are only added to the training side and only for training groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import SEED


class LeakageError(AssertionError):
    pass


def assert_no_leakage(meta: pd.DataFrame, train_idx, test_idx) -> None:
    """No group, no file content hash and no file id may appear on both sides."""
    tr, te = meta.iloc[np.asarray(train_idx)], meta.iloc[np.asarray(test_idx)]
    for col in ("group", "sha256", "file_id"):
        both = set(tr[col]) & set(te[col])
        if both:
            raise LeakageError(f"{len(both)} value(s) of '{col}' on both sides, e.g. {sorted(both)[:3]}")
    if (te["variant"] != "orig").any():
        raise LeakageError("augmented rows found on the evaluation side")


def with_augmented(meta: pd.DataFrame, train_idx, variants: tuple[str, ...]) -> np.ndarray:
    """Add augmented copies of the training windows (same files => same groups)."""
    train_idx = np.asarray(train_idx)
    if not variants:
        return train_idx
    keys = set(zip(meta.file_id.values[train_idx], meta.window_idx.values[train_idx]))
    is_aug = meta.variant.isin(variants).values
    aug = np.where(is_aug)[0]
    aug = aug[[(f, w) in keys for f, w in zip(meta.file_id.values[aug], meta.window_idx.values[aug])]]
    return np.concatenate([train_idx, aug])


def group_kfold(meta: pd.DataFrame, rows: np.ndarray, n_splits: int, seed: int = SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold over the given (orig) rows; returns positional indices into ``meta``."""
    rows = np.asarray(rows)
    sub = meta.iloc[rows]
    n_groups = sub.group.nunique()
    k = min(n_splits, n_groups)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for tr, va in cv.split(np.zeros(len(sub)), sub.label.values, sub.group.values):
        out.append((rows[tr], rows[va]))
    return out


def leave_one_out_by(meta: pd.DataFrame, rows: np.ndarray, key: str, strict: bool = True,
                     human_folds: int = 4, seed: int = SEED) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Hold out each value of ``key`` among AI rows (e.g. generator, source speaker, target voice).
    Human rows are split into ``human_folds`` group folds and rotated so each test fold also has
    unseen human groups. Training always excludes every AI row sharing the held-out value and the
    held-out human groups. With ``strict=True`` it also excludes every other row of any group present
    in the test fold (fully group-safe). ``strict=False`` is only for factors that cut across groups
    (target voice): it is labelled as relaxed wherever it is reported."""
    rows = np.asarray(rows)
    sub = meta.iloc[rows]
    ai_vals = sorted(sub.loc[sub.label == 1, key].unique())
    hum_groups = np.array(sorted(sub.loc[sub.label == 0, "group"].unique()))
    rng = np.random.default_rng(seed)
    hum_groups = rng.permutation(hum_groups)
    hum_fold = {g: i % human_folds for i, g in enumerate(hum_groups)}
    out = []
    for i, v in enumerate(ai_vals):
        hf = i % human_folds
        test_mask = ((sub.label == 1) & (sub[key] == v)) | ((sub.label == 0) & sub.group.map(hum_fold).eq(hf))
        held_human = set(sub.loc[test_mask & (sub.label == 0), "group"])
        blocked = set(sub.loc[test_mask, "group"]) if strict else held_human
        train_mask = ~test_mask & ~sub.group.isin(blocked) & ~((sub.label == 1) & (sub[key] == v))
        out.append((str(v), rows[train_mask.values], rows[test_mask.values]))
    return out


def summarize_split(meta: pd.DataFrame, idx) -> dict:
    s = meta.iloc[np.asarray(idx)]
    return {"rows": int(len(s)), "ai": int((s.label == 1).sum()), "human": int((s.label == 0).sum()),
            "groups": int(s.group.nunique())}
