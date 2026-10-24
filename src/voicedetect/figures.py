"""Static report figures (light theme PNG) rendered from saved artifacts with the shared chart builders."""
from __future__ import annotations

import json

import pandas as pd

from . import charts as C
from .config import FIGURES_DIR, REPORTS_DIR

MODE = "light"


def _read(name):
    p = REPORTS_DIR / name
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def build_all() -> list[str]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((REPORTS_DIR.parent / "models" / "metadata.json").read_text(encoding="utf-8"))
    sel = _read("selection/model_selection.json")
    cal = _read("selection/calibration.json")
    test = _read("test_metrics.json")
    dep = sel["deploy"]
    figs = {
        "legacy_vs_honest": C.legacy_vs_honest(_read("legacy_reproduction.json"), MODE),
        "class_balance": C.class_balance(_read("feature_summary.json"), MODE),
        "durations": C.duration_box(_read("dataset_audit.csv"), MODE),
        "model_comparison_combined": C.model_comparison(_read("model_comparison.csv"), "combined",
                                                        (dep["model"], dep["feature_set"]) if dep["setting"] == "combined" else None, MODE),
        "model_comparison_user": C.model_comparison(_read("model_comparison.csv"), "user",
                                                    (sel["selection_by_setting"]["user"]["chosen"]["model"],
                                                     sel["selection_by_setting"]["user"]["chosen"]["feature_set"]), MODE),
        "model_comparison_codec_matched": C.model_comparison(_read("model_comparison.csv"), "user_codec_matched", None, MODE),
        "rbf_heatmap": C.rbf_heatmap(_read("rbf_svm_grid.csv"), dep["setting"], dep["feature_set"], MODE),
        "roc_test": C.curves(_read("test_curves.csv"), "deployed", "roc", MODE),
        "pr_test": C.curves(_read("test_curves.csv"), "deployed", "pr", MODE),
        "confusion_test": C.confusion(test["deployed"]["scopes"]["all test windows"], MODE),
        "calibration": C.reliability(_read("reliability_oof.csv"), cal["chosen_method"], MODE),
        "threshold_tradeoff": C.threshold_tradeoff(_read("threshold_tradeoff_oof.csv"), meta["threshold"],
                                                   (meta["threshold"] - meta["inconclusive_halfwidth"],
                                                    meta["threshold"] + meta["inconclusive_halfwidth"]), MODE),
        "robustness": C.robustness_dots(_read("robustness.csv"), MODE),
        "logo": C.generalisation_bars(_read("generalisation.csv"), "LOGO", MODE),
        "loso": C.generalisation_bars(_read("generalisation.csv"), "LOSO", MODE),
        "group_importance": C.importance_bars(_read("group_permutation_importance.csv"), "group", "auc_drop",
                                              "auc_drop_fold_std", MODE),
        "top_features": C.importance_bars(_read("permutation_importance.csv").head(15), "feature", "importance",
                                          "importance_fold_std", MODE),
        "projection_pca": C.projection(_read("projection_coords.csv"), "pca", MODE),
        "projection_tsne": C.projection(_read("projection_coords.csv"), "tsne", MODE),
        "learning_curve": C.learning_curve(_read("learning_curve.csv"), MODE),
    }
    if (REPORTS_DIR / "selection" / "robustness_study.json").exists():
        figs["robust_tradeoff"] = C.robust_tradeoff(_read("selection/robustness_study.json"), MODE)
    if (REPORTS_DIR / "robustness_primary.csv").exists():
        figs["robustness_primary"] = C.robustness_dots(_read("robustness_primary.csv"), MODE)
    written = []
    for name, fig in figs.items():
        out = FIGURES_DIR / f"{name}.png"
        fig.update_layout(width=900)
        fig.write_image(str(out), scale=2)
        written.append(out.name)
    return written
