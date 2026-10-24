"""The pre-declared selection rules, tested on small hand-made inputs."""
import json

from voicedetect.config import REPORTS_DIR
from voicedetect.pipeline import choose_robust

from .conftest import requires_model


def cand(name, augs, clean, robust):
    return {"candidate": name, "augmentations": augs, "auc_by_condition": {"clean": clean}, "robust_auc": robust}


def test_robust_rule_respects_clean_tolerance():
    rows = [cand("none", [], 0.966, 0.766), cand("mp3", ["mp3cv"], 0.961, 0.784),
            cand("noise+mic", ["n", "m"], 0.936, 0.880)]
    assert choose_robust(rows, tolerance=0.02)["candidate"] == "mp3"       # 0.936 < 0.966 - 0.02: ineligible
    assert choose_robust(rows, tolerance=0.05)["candidate"] == "noise+mic"


def test_robust_rule_tie_prefers_fewer_augmentations():
    rows = [cand("none", [], 0.96, 0.800), cand("a", ["x"], 0.96, 0.850), cand("ab", ["x", "y"], 0.96, 0.851)]
    assert choose_robust(rows, tie=0.002)["candidate"] == "a"
    assert choose_robust(rows, tie=0.0)["candidate"] == "ab"


def test_robust_rule_can_keep_no_augmentation():
    rows = [cand("none", [], 0.96, 0.90), cand("a", ["x"], 0.90, 0.95)]
    assert choose_robust(rows)["candidate"] == "none"


@requires_model
def test_saved_model_selection_matches_rule():
    """Re-apply the pre-declared selection rule to the saved CV artifacts; it must pick the same model."""
    import pandas as pd
    from voicedetect.pipeline import select_in_setting
    saved = json.loads((REPORTS_DIR / "selection" / "model_selection.json").read_text())
    summary = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    oof = pd.read_parquet(REPORTS_DIR / "oof_predictions.parquet")
    again = select_in_setting(summary, oof, "combined")
    assert again["chosen"]["model"] == saved["selection_by_setting"]["combined"]["chosen"]["model"]
    assert again["chosen"]["feature_set"] == saved["selection_by_setting"]["combined"]["chosen"]["feature_set"]


@requires_model
def test_saved_robust_choice_matches_rule():
    p = REPORTS_DIR / "selection" / "robustness_study.json"
    if p.exists():
        s = json.loads(p.read_text())
        assert choose_robust(s["candidates"])["candidate"] == s["chosen"]
