import numpy as np
import pandas as pd
import pytest

from voicedetect.splits import LeakageError, assert_no_leakage, group_kfold, leave_one_out_by, with_augmented

from .conftest import requires_data


def toy_meta(n_groups=12, per_group=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        label = g % 2
        for w in range(per_group):
            rows.append({"file_id": f"f{g}", "window_idx": w, "group": f"g{g}", "label": label,
                         "sha256": f"h{g}", "variant": "orig", "generator": f"gen{g % 3}" if label else "human",
                         "speaker": f"s{g}"})
    meta = pd.DataFrame(rows)
    aug = meta.copy()
    aug["variant"] = "mp3cv"
    meta = pd.concat([meta, aug], ignore_index=True)
    return meta.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_group_kfold_never_splits_a_group():
    meta = toy_meta()
    rows = np.flatnonzero(meta.variant.eq("orig"))
    folds = group_kfold(meta, rows, 3, seed=1)
    seen = []
    for tr, va in folds:
        assert set(meta.group[tr]).isdisjoint(set(meta.group[va]))
        assert_no_leakage(meta, tr, va)
        seen += list(va)
    assert sorted(seen) == sorted(rows)  # every eval row validated exactly once


def test_augmented_rows_only_join_training_side_for_training_groups():
    meta = toy_meta()
    rows = np.flatnonzero(meta.variant.eq("orig"))
    tr, va = group_kfold(meta, rows, 3, seed=1)[0]
    tr_all = with_augmented(meta, tr, ("mp3cv",))
    added = meta.iloc[np.setdiff1d(tr_all, tr)]
    assert (added.variant == "mp3cv").all()
    assert set(added.group) <= set(meta.group[tr])
    assert_no_leakage(meta, tr_all, va)


def test_leakage_guard_detects_group_hash_and_augmented_overlap():
    meta = toy_meta()
    orig = np.flatnonzero(meta.variant.eq("orig"))
    g0 = np.flatnonzero((meta.group == "g0") & meta.variant.eq("orig"))
    with pytest.raises(LeakageError, match="group"):
        assert_no_leakage(meta, g0[:2], g0[2:])
    m2 = meta.copy()
    m2.loc[m2.group == "g1", "sha256"] = "h0"  # same content under a different group name
    a = np.flatnonzero((m2.group == "g0") & m2.variant.eq("orig"))
    b = np.flatnonzero((m2.group == "g1") & m2.variant.eq("orig"))
    with pytest.raises(LeakageError, match="sha256"):
        assert_no_leakage(m2, a, b)
    aug = np.flatnonzero(meta.variant.eq("mp3cv") & (meta.group == "g5"))
    rest = np.setdiff1d(orig, np.flatnonzero(meta.group == "g5"))
    with pytest.raises(LeakageError, match="augmented"):
        assert_no_leakage(meta, rest, aug)


def test_leave_one_generator_out_strict_is_leak_free():
    meta = toy_meta()
    rows = np.flatnonzero(meta.variant.eq("orig"))
    out = leave_one_out_by(meta, rows, "generator", strict=True, human_folds=2)
    assert {v for v, _, _ in out} == {"gen0", "gen1", "gen2"}
    for v, tr, te in out:
        assert_no_leakage(meta, tr, te)
        assert set(meta.generator[te][meta.label[te] == 1]) == {v}
        assert v not in set(meta.generator[tr])
        assert (meta.label[te] == 0).any()


@requires_data
def test_real_manifest_partitions_are_group_disjoint():
    from voicedetect.manifest import load_manifest
    m = load_manifest()
    dev, test = m[m.partition == "dev"], m[m.partition == "test"]
    assert set(dev.group).isdisjoint(set(test.group))
    assert set(dev.sha256).isdisjoint(set(test.sha256))
    assert m.file_id.is_unique and m.sha256.is_unique


@requires_data
def test_real_feature_table_cv_folds_are_leak_free():
    from voicedetect import dataset
    from voicedetect.evaluate import _check, eval_mask
    table = dataset.load_table()
    meta = table.drop(columns=dataset.feature_columns("extended"))
    for setting in ("user", "user_codec_matched", "combined"):
        rows = np.flatnonzero(eval_mask(table, setting, "dev"))
        for tr, va in group_kfold(meta, rows, 5):
            _check(meta, with_augmented(meta, tr, ("mp3cv",)), va, setting)
        test_rows = np.flatnonzero(eval_mask(table, setting, "test"))
        _check(meta, rows, test_rows, setting)
