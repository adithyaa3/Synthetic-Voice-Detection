"""End-to-end pipeline. Every number in the reports/UI is produced by one of these steps.

    python -m voicedetect.pipeline all          # everything, in order
    python -m voicedetect.pipeline <step> ...   # individual steps (see STEPS)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

from .config import REPORTS_DIR, SEED

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
SELECTION_DIR = REPORTS_DIR / "selection"
MODEL_SET = ["majority", "logreg", "linear_svm", "rbf_svm", "rbf_svm_original", "random_forest", "hist_gb"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _json(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")


# ----------------------------------------------------------------------------------------- data
def step_manifest():
    from . import manifest as M
    df = M.build_manifest()
    M.save_manifest(df)
    _log(f"manifest: {len(df)} files, hash {M.manifest_hash(df)}")


def step_audit():
    from . import data_audit, manifest as M
    rep = data_audit.run_audit(M.load_manifest())
    _log(f"audit: {rep['files_total']} files, unreadable {rep['unreadable_total']}, "
         f"rejected by loader {rep['rejected_by_loader_total']}, duplicates {len(rep['exact_duplicates'])}")


def step_features():
    from . import dataset as D, manifest as M
    D.build_table(M.load_manifest())


def step_legacy():
    from . import legacy
    out = legacy.run()
    o = out["original_split"]
    _log(f"legacy: original split accuracy {o['accuracy']:.3f} (majority {o['majority_class_accuracy']:.3f}); "
         f"group-CV balanced acc {out['group_cv_5fold_x3']['balanced_accuracy']['mean']:.3f}; "
         f"LOSO balanced acc {out['leave_one_source_out']['balanced_accuracy']['mean']:.3f}")


# ----------------------------------------------------------------------------- model comparison
def step_compare(settings=None, feature_sets=("baseline", "extended"), models=None):
    from . import dataset as D, evaluate as E
    table = D.load_table()
    settings = settings or E.SETTINGS
    models = models or MODEL_SET
    folds, oofs, grids = [], [], []
    for s in settings:
        for fs in feature_sets:
            t0 = time.time()
            r = E.nested_cv(table, s, fs, models, log=_log)
            folds.append(r["folds"]); oofs.append(r["oof"]); grids.append(r["grid"])
            _log(f"== {s}/{fs} done in {time.time() - t0:.0f}s")
    folds = pd.concat(folds, ignore_index=True)
    oof = pd.concat(oofs, ignore_index=True)
    grid = pd.concat(grids, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    folds.to_csv(REPORTS_DIR / "cv_fold_metrics.csv", index=False)
    oof.to_parquet(REPORTS_DIR / "oof_predictions.parquet", index=False)
    grid.to_csv(REPORTS_DIR / "rbf_svm_grid.csv", index=False)
    summary = E.summarize(folds)
    summary.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    E.pooled_ci(oof).to_csv(REPORTS_DIR / "pooled_auc_ci.csv", index=False)
    E.by_source(oof).to_csv(REPORTS_DIR / "oof_by_source.csv", index=False)
    show = summary[["setting", "feature_set", "model", "roc_auc_mean", "roc_auc_std", "balanced_accuracy_mean",
                    "balanced_accuracy_std", "roc_auc_mp3_mean"]]
    _log(show.round(3).to_string())


# ------------------------------------------------------------------------------------ selection
FEATURE_SIMPLICITY = ["baseline", "extended"]
DEPLOY_SETTING = "combined"
USER_DOMAIN_TOLERANCE = 0.02   # deploy the combined model unless it is this much worse on user-domain rows
AUG_TOLERANCE = 0.005          # augmentation adopted if clean AUC drops by less than this AND MP3 AUC improves


def _cands(summary, setting):
    return summary[(summary.setting == setting) & (summary.augment == "none") &
                   (summary.calibration == "default") & (summary.model != "majority")].copy()


def select_in_setting(summary: pd.DataFrame, oof: pd.DataFrame, setting: str) -> dict:
    """Pre-declared rule: highest mean outer-fold ROC-AUC; every candidate whose paired group-bootstrap
    95 % CI of (best - candidate) pooled OOF AUC includes 0 is 'tied'; among tied candidates pick the
    simplest (model simplicity first, then fewer features)."""
    from .metrics import paired_bootstrap_auc_diff
    from .models import SIMPLICITY
    c = _cands(summary, setting)
    best = c.loc[c.roc_auc_mean.idxmax()]
    o = oof[(oof.setting == setting) & (oof.augment == "none") & (oof.calibration == "default")]
    ref = o[(o.model == best.model) & (o.feature_set == best.feature_set)].set_index("row").sort_index()
    rows = []
    for r in c.itertuples():
        oth = o[(o.model == r.model) & (o.feature_set == r.feature_set)].set_index("row").loc[ref.index]
        if r.model == best.model and r.feature_set == best.feature_set:
            d, lo, hi = 0.0, 0.0, 0.0
        else:
            d, lo, hi = paired_bootstrap_auc_diff(ref.label.values, ref.p_ai.values, oth.p_ai.values,
                                                  ref.group.values, n_boot=1000, seed=SEED)
        rows.append({"model": r.model, "feature_set": r.feature_set, "roc_auc_mean": r.roc_auc_mean,
                     "roc_auc_std": r.roc_auc_std, "balanced_accuracy_mean": r.balanced_accuracy_mean,
                     "auc_diff_best_minus_this": d, "ci_low": lo, "ci_high": hi, "tied_with_best": bool(lo <= 0.0)})
    tab = pd.DataFrame(rows)
    tied = tab[tab.tied_with_best].copy()
    tied["_k"] = [(SIMPLICITY.index(m), FEATURE_SIMPLICITY.index(f)) for m, f in zip(tied.model, tied.feature_set)]
    chosen = tied.sort_values("_k").iloc[0]
    return {"setting": setting, "best_by_mean_auc": {"model": best.model, "feature_set": best.feature_set,
                                                      "roc_auc_mean": float(best.roc_auc_mean)},
            "chosen": {"model": chosen.model, "feature_set": chosen.feature_set,
                       "roc_auc_mean": float(chosen.roc_auc_mean), "roc_auc_std": float(chosen.roc_auc_std)},
            "table": tab.sort_values("roc_auc_mean", ascending=False).to_dict("records")}


def step_select():
    from sklearn.metrics import roc_auc_score
    from .models import SIMPLICITY
    summary = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    oof = pd.read_parquet(REPORTS_DIR / "oof_predictions.parquet")
    folds = pd.read_csv(REPORTS_DIR / "cv_fold_metrics.csv")
    sel = {s: select_in_setting(summary, oof, s) for s in ("combined", "user", "user_codec_matched")}
    # user-domain check: does adding supplementary data hurt on the user's own data?
    ch = sel["combined"]["chosen"]
    o = oof[(oof.augment == "none") & (oof.calibration == "default") & (oof.model == ch["model"]) &
            (oof.feature_set == ch["feature_set"])]
    comb_user = o[(o.setting == "combined") & (o.source != "asvspoof2019_la")]
    user_only = o[o.setting == "user"]
    auc_comb_user = roc_auc_score(comb_user.label, comb_user.p_ai)
    auc_user = roc_auc_score(user_only.label, user_only.p_ai)
    deploy = DEPLOY_SETTING if auc_comb_user >= auc_user - USER_DOMAIN_TOLERANCE else "user"
    from . import evaluate as E
    paired = pd.concat([E.paired_comparison(folds, oof, s, fs, ref)
                        for s in ("combined", "user", "user_codec_matched")
                        for fs, ref in [(sel[s]["chosen"]["feature_set"], sel[s]["chosen"]["model"])]], ignore_index=True)
    paired.to_csv(REPORTS_DIR / "paired_comparisons.csv", index=False)
    out = {
        "rule": ("Primary metric: mean outer-fold ROC-AUC (group-aware 5-fold CV, dev partition only). Candidates whose "
                 "paired group-bootstrap 95% CI for (best - candidate) pooled OOF AUC includes 0 are tied with the best; "
                 "among tied candidates the simplest is chosen (model: " + " < ".join(SIMPLICITY[1:]) +
                 "; features: baseline < extended). The deployed model is trained on user + supplementary data "
                 f"unless its out-of-fold AUC on user-domain rows is more than {USER_DOMAIN_TOLERANCE} below the user-only model's."),
        "selection_by_setting": sel,
        "user_domain_check": {"model": ch["model"], "feature_set": ch["feature_set"],
                              "oof_auc_on_user_rows_trained_combined": auc_comb_user,
                              "oof_auc_on_user_rows_trained_user_only": auc_user,
                              "tolerance": USER_DOMAIN_TOLERANCE, "deploy_setting": deploy},
        "deploy": {"setting": deploy, **sel[deploy]["chosen"]},
    }
    _json(SELECTION_DIR / "model_selection.json", out)
    _log(json.dumps({k: out[k] for k in ("user_domain_check", "deploy")}, indent=2))
    for s in sel:
        _log(f"{s}: best={sel[s]['best_by_mean_auc']} chosen={sel[s]['chosen']}")


def _selection():
    return json.loads((SELECTION_DIR / "model_selection.json").read_text(encoding="utf-8"))


def step_augment():
    from . import dataset as D, evaluate as E
    dep = _selection()["deploy"]
    table = D.load_table()
    r = E.nested_cv(table, dep["setting"], dep["feature_set"], [dep["model"]], aug_variants=("mp3cv",), log=_log)
    base = pd.read_csv(REPORTS_DIR / "cv_fold_metrics.csv")
    base = base[(base.setting == dep["setting"]) & (base.feature_set == dep["feature_set"]) &
                (base.model == dep["model"]) & (base.augment == "none")]
    r["folds"].to_csv(REPORTS_DIR / "ablation_augmentation_folds.csv", index=False)
    r["oof"].to_parquet(REPORTS_DIR / "ablation_augmentation_oof.parquet", index=False)
    a, b = r["folds"], base
    res = {"config": dep,
           "no_augmentation": {"roc_auc_mean": b.roc_auc.mean(), "roc_auc_std": b.roc_auc.std(),
                               "roc_auc_mp3_mean": b.roc_auc_mp3.mean(), "balanced_accuracy_mean": b.balanced_accuracy.mean(),
                               "balanced_accuracy_mp3_mean": b.balanced_accuracy_mp3.mean()},
           "mp3_augmentation": {"roc_auc_mean": a.roc_auc.mean(), "roc_auc_std": a.roc_auc.std(),
                                "roc_auc_mp3_mean": a.roc_auc_mp3.mean(), "balanced_accuracy_mean": a.balanced_accuracy.mean(),
                                "balanced_accuracy_mp3_mean": a.balanced_accuracy_mp3.mean()},
           "rule": f"adopt augmentation if clean CV AUC drops by < {AUG_TOLERANCE} and CV AUC on MP3 versions of the validation windows improves"}
    res["adopted"] = bool(a.roc_auc.mean() >= b.roc_auc.mean() - AUG_TOLERANCE and a.roc_auc_mp3.mean() > b.roc_auc_mp3.mean())
    _json(SELECTION_DIR / "augmentation.json", res)
    _log(json.dumps(res, indent=2, default=float))


# ------------------------------------------------------------------- robustness-aware augmentation
# Added after the test-set robustness analysis revealed strong noise sensitivity of the model chosen by
# the clean-data rule. Decided on DEVELOPMENT data only; rule fixed before this study was run.
ROBUST_CANDIDATES = {"none": (), "mp3": ("mp3cv",), "noise+mic": ("aug_noise", "aug_mic"),
                     "mp3+noise+mic": ("mp3cv", "aug_noise", "aug_mic")}
ROBUST_CONDITIONS = ["clean", "mp3", "white20", "room20", "mic"]
ROBUST_CLEAN_TOLERANCE = 0.02
ROBUST_TIE = 0.002


def choose_robust(rows: list[dict], tolerance: float = ROBUST_CLEAN_TOLERANCE, tie: float = ROBUST_TIE) -> dict:
    """rows[0] must be the no-augmentation candidate. Eligible = clean AUC within ``tolerance`` of it;
    highest mean AUC wins; candidates within ``tie`` of the best are tied -> fewest augmentation types,
    then candidate order."""
    base_clean = rows[0]["auc_by_condition"]["clean"]
    eligible = [r for r in rows if r["auc_by_condition"]["clean"] >= base_clean - tolerance]
    best = max(r["robust_auc"] for r in eligible)
    tied = [r for r in eligible if r["robust_auc"] >= best - tie]
    return min(tied, key=lambda r: (len(r["augmentations"]), rows.index(r)))


def _clean_fold_params(dep) -> dict:
    """Per-fold hyperparameters of the selected configuration from the (deterministic) clean inner search."""
    import ast
    src = pd.read_csv(REPORTS_DIR / "cv_fold_metrics.csv")
    src = src[(src.setting == dep["setting"]) & (src.feature_set == dep["feature_set"]) & (src.model == dep["model"]) &
              (src.augment == "none") & (src.calibration == "default")]
    return {int(r.fold): ast.literal_eval(r.best_params) for r in src.itertuples()}


def step_features_aug():
    from . import dataset as D
    D.build_aug_table(D.load_table(include_aug=False))


def step_robust():
    from . import dataset as D, evaluate as E
    dep = _selection()["deploy"]
    table = D.load_table()
    fold_params = _clean_fold_params(dep)
    perts = tuple(E.EVAL_PERTURBATIONS)
    folds, rows = [], []
    for name, aug in ROBUST_CANDIDATES.items():
        r = E.nested_cv(table, dep["setting"], dep["feature_set"], [dep["model"]], aug_variants=aug, calibration="none",
                        fold_params=fold_params, perturbations=perts, log=_log)
        f = r["folds"].assign(candidate=name)
        folds.append(f)
        aucs = {c: float(f["roc_auc" if c == "clean" else f"roc_auc_{c}"].mean()) for c in ROBUST_CONDITIONS}
        spec = {c: float(f["specificity" if c == "clean" else f"specificity_{c}"].mean()) for c in ROBUST_CONDITIONS}
        rec = {c: float(f["recall" if c == "clean" else f"recall_{c}"].mean()) for c in ROBUST_CONDITIONS}
        rows.append({"candidate": name, "augmentations": list(aug), "auc_by_condition": aucs,
                     "robust_auc": float(np.mean(list(aucs.values()))),
                     "specificity_at_0.5_by_condition": spec, "recall_at_0.5_by_condition": rec})
    chosen = choose_robust(rows)
    pd.concat(folds).to_csv(REPORTS_DIR / "robust_study_folds.csv", index=False)
    out = {"config": dep, "candidates": rows, "chosen": chosen["candidate"], "chosen_augmentations": chosen["augmentations"],
           "rule": (f"Maximise mean validation ROC-AUC over conditions {ROBUST_CONDITIONS} (validation windows perturbed; "
                    f"development data only), among candidates whose clean AUC is within {ROBUST_CLEAN_TOLERANCE} of "
                    f"no augmentation; candidates within {ROBUST_TIE} of the best are tied and the one with fewer "
                    "augmentation types wins. Augmented copies are used in training folds only."),
           "motivation": ("The test-set robustness analysis of the model chosen by the clean-data rule showed that added "
                          "noise or microphone-like processing made it label almost all human speech as AI. This study "
                          "was run afterwards, so the deployed model's test results are post-hoc (see VERIFICATION_REPORT)."),
           "hyperparameters": "per-fold values from the clean inner search (not re-tuned); the final model is re-tuned"}
    _json(SELECTION_DIR / "robustness_study.json", out)
    for r in rows:
        _log(f"{r['candidate']:<14} robust AUC {r['robust_auc']:.3f} " +
             " ".join(f"{c}={v:.3f}" for c, v in r["auc_by_condition"].items()))
    _log(f"chosen: {chosen['candidate']}")


def _configs() -> dict:
    """'primary' = the model chosen by the pre-declared clean-data rules; 'deployed' = with the
    robustness-aware augmentation choice (identical to primary when that choice is 'none')."""
    sel = _selection()["deploy"]
    base = {"setting": sel["setting"], "model": sel["model"], "feature_set": sel["feature_set"]}
    primary_aug = ("mp3cv",) if json.loads((SELECTION_DIR / "augmentation.json").read_text())["adopted"] else ()
    rs = SELECTION_DIR / "robustness_study.json"
    deployed_aug = tuple(json.loads(rs.read_text())["chosen_augmentations"]) if rs.exists() else primary_aug
    cfgs = {"deployed": {**base, "aug": deployed_aug}}
    if deployed_aug != primary_aug:
        cfgs["primary"] = {**base, "aug": primary_aug}
    return cfgs


def _sfx(which: str) -> str:
    return "" if which == "deployed" else f"_{which}"


def step_calibrate():
    for which, cfg in _configs().items():
        _calibrate_config(which, cfg)


def _calibrate_config(which: str, cfg: dict):
    from . import calibration as C, dataset as D, evaluate as E
    from .config import DECISION
    from .metrics import ece
    from .models import NEEDS_PLATT
    dep, aug = cfg, cfg["aug"]
    sfx = _sfx(which)
    table = D.load_table()
    fold_params = _clean_fold_params(dep)
    methods = ["sigmoid", "isotonic"] if dep["model"] in NEEDS_PLATT else ["none", "sigmoid", "isotonic"]
    runs, rel = {}, []
    for meth in methods:
        r = E.nested_cv(table, dep["setting"], dep["feature_set"], [dep["model"]], aug_variants=aug,
                        calibration=meth, fold_params=fold_params, perturbations=tuple(E.EVAL_PERTURBATIONS), log=_log)
        runs[meth] = r
        o = r["oof"]
        t = C.reliability_table(o.label, o.p_ai)
        t.insert(0, "method", meth)
        rel.append(t)
    summ = {m: {"log_loss_mean": r["folds"].log_loss.mean(), "brier_mean": r["folds"].brier.mean(),
                "ece_mean": r["folds"].ece.mean(), "roc_auc_mean": r["folds"].roc_auc.mean(),
                "pooled_ece": ece(r["oof"].label, r["oof"].p_ai)}
            for m, r in runs.items()}
    chosen = min(methods, key=lambda m: (round(summ[m]["log_loss_mean"], 4), summ[m]["brier_mean"]))
    o = runs[chosen]["oof"]
    decision = C.decide(o.label.values, o.p_ai.values, DECISION.cost_false_alarm, DECISION.cost_miss,
                        DECISION.inconclusive_halfwidth)
    per_domain = {}
    for src, d in o.groupby(o.source == "asvspoof2019_la"):
        name = "supplementary (ASVspoof)" if src else "user data"
        per_domain[name] = C.decide(d.label.values, d.p_ai.values, DECISION.cost_false_alarm, DECISION.cost_miss,
                                    DECISION.inconclusive_halfwidth)
    pd.concat(rel).to_csv(REPORTS_DIR / f"reliability_oof{sfx}.csv", index=False)
    C.threshold_table(o.label.values, o.p_ai.values, DECISION.cost_false_alarm, DECISION.cost_miss).to_csv(
        REPORTS_DIR / f"threshold_tradeoff_oof{sfx}.csv", index=False)
    o.to_parquet(REPORTS_DIR / f"oof_final_config{sfx}.parquet", index=False)
    pd.concat([r["folds"] for r in runs.values()]).to_csv(REPORTS_DIR / f"calibration_folds{sfx}.csv", index=False)
    # dev-only robustness of the chosen calibration at the chosen threshold (validation windows perturbed)
    f = runs[chosen]["folds"]
    dev_rob = {}
    for c in ROBUST_CONDITIONS:
        col = "p_ai" if c == "clean" else f"p_ai_{c}"
        if col in o and o[col].notna().all():
            mm = C.binary_metrics(o.label.values, o[col].values, decision["threshold"])
            dev_rob[c] = {k: mm[k] for k in ("roc_auc", "balanced_accuracy", "recall", "specificity")}
    out = {"role": which, "config": dep, "augmentation": list(aug), "methods": summ, "chosen_method": chosen,
           "dev_robustness_at_threshold": dev_rob, "folds_mean_auc": float(f.roc_auc.mean()),
           "selection_rule": "lowest mean outer-fold log loss (Brier as tie-break), on validation folds only",
           "decision": {**decision, "inconclusive_halfwidth": DECISION.inconclusive_halfwidth},
           "decision_by_domain_for_information": per_domain,
           "cost_rationale": ("A false alarm (a real person's voice labelled synthetic) is treated as twice as costly as a "
                              "missed synthetic voice, because acting on a false accusation harms a real person, whereas "
                              "a miss leaves the status quo. The threshold minimising expected cost on out-of-fold "
                              "validation predictions is used; an inconclusive band of +/- "
                              f"{DECISION.inconclusive_halfwidth} around it is shown in the UI.")}
    _json(SELECTION_DIR / f"calibration{sfx}.json", out)
    _log(json.dumps({"role": which, "methods": summ, "chosen": chosen, "threshold": decision["threshold"],
                     "inconclusive": decision["inconclusive"], "dev_robustness_at_threshold": dev_rob},
                    indent=2, default=float))


# ---------------------------------------------------------------------------- final model + test
def _final_config(which: str = "deployed") -> dict:
    cfg = _configs()[which]
    cal = json.loads((SELECTION_DIR / f"calibration{_sfx(which)}.json").read_text())
    return {**cfg, "calibration": cal["chosen_method"], "threshold": cal["decision"]["threshold"],
            "halfwidth": cal["decision"]["inconclusive_halfwidth"]}


def _fit_final(table, cfg):
    from .dataset import feature_columns
    from .evaluate import eval_mask
    from .tuning import fit_tuned
    meta = table.drop(columns=feature_columns("extended"))
    X, y = table[feature_columns(cfg["feature_set"])].to_numpy(), table.label.to_numpy()
    rows = np.flatnonzero(eval_mask(table, cfg["setting"], "dev"))
    res = fit_tuned(cfg["model"], X, y, meta, rows, cfg["aug"], seed=SEED, calibration=cfg["calibration"])
    return res, rows


def step_finalize():
    from . import dataset as D, manifest as M
    from .calibration import decide
    from .config import DECISION, MODELS_DIR, N_INNER_SPLITS, N_OUTER_SPLITS
    from .predict import save_bundle
    table = D.load_table()
    man = M.load_manifest()
    summary = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    roles = {"deployed": "deployed model (robustness-aware augmentation choice, decided on development data)",
             "primary": "pre-registered model chosen by the clean-data rules (not deployed)"}
    for which in _configs():
        cfg = _final_config(which)
        res, rows = _fit_final(table, cfg)
        cal = json.loads((SELECTION_DIR / f"calibration{_sfx(which)}.json").read_text())
        s = summary[(summary.setting == cfg["setting"]) & (summary.model == cfg["model"]) &
                    (summary.feature_set == cfg["feature_set"]) & (summary.augment == "none")].iloc[0]
        tr = table.iloc[rows]
        extra = {
            "role": roles[which] if len(_configs()) > 1 else "deployed model",
            "training_setting": cfg["setting"], "model_name": cfg["model"], "best_params": res["best_params"],
            "augmentation": list(cfg["aug"]), "calibration": res["calibration"], "seed": SEED,
            "cv": {"outer": N_OUTER_SPLITS, "inner": N_INNER_SPLITS, "type": "StratifiedGroupKFold"},
            "data_manifest_hash": M.manifest_hash(man), "feature_cache_key": D.cache_key(),
            "training_windows": int(len(tr)), "training_files": int(tr.file_id.nunique()),
            "training_groups": int(tr.group.nunique()),
            "training_windows_by_source": {f"{a}/{c}": int(n) for (a, c), n in tr.groupby(["source", "class_name"]).size().items()},
            "cv_metrics": {k: float(s[k]) for k in s.index if k.endswith(("_mean", "_std")) and not k.startswith("fit")},
            "cv_metrics_note": "clean-data nested CV of this model family / features without augmentation",
            "calibration_selection": {k: cal[k] for k in ("methods", "chosen_method")},
            "dev_robustness_at_threshold": cal.get("dev_robustness_at_threshold", {}),
            "decision_costs": {"false_alarm": DECISION.cost_false_alarm, "miss": DECISION.cost_miss},
        }
        paths = {} if which == "deployed" else {"model_path": MODELS_DIR / f"{which}_model.joblib",
                                                "metadata_path": MODELS_DIR / f"{which}_metadata.json"}
        meta = save_bundle(res["model"], cfg["feature_set"], {"threshold": cfg["threshold"],
                                                             "inconclusive_halfwidth": cfg["halfwidth"]}, extra, **paths)
        _log(f"{which} model saved: {meta['model_version']} {cfg} params={res['best_params']}")

    # comparison model trained on USER DATA ONLY (selected by the same rule within the 'user' setting)
    ucfg = _selection()["selection_by_setting"]["user"]["chosen"]
    ucfg = {"setting": "user", "model": ucfg["model"], "feature_set": ucfg["feature_set"], "aug": (), "calibration": None}
    ures, urows = _fit_final(table, ucfg)
    oof = pd.read_parquet(REPORTS_DIR / "oof_predictions.parquet")
    uo = oof[(oof.setting == "user") & (oof.model == ucfg["model"]) & (oof.feature_set == ucfg["feature_set"]) &
             (oof.augment == "none")]
    udec = decide(uo.label.values, uo.p_ai.values, DECISION.cost_false_alarm, DECISION.cost_miss, DECISION.inconclusive_halfwidth)
    save_bundle(ures["model"], ucfg["feature_set"], {"threshold": udec["threshold"],
                                                    "inconclusive_halfwidth": DECISION.inconclusive_halfwidth},
                {"training_setting": "user", "model_name": ucfg["model"], "best_params": ures["best_params"],
                 "calibration": ures["calibration"], "seed": SEED, "data_manifest_hash": M.manifest_hash(man),
                 "role": "comparison model trained on user data only (not deployed)"},
                model_path=MODELS_DIR / "user_only_model.joblib", metadata_path=MODELS_DIR / "user_only_metadata.json")
    _log(f"user-only comparison model saved: {ucfg} params={ures['best_params']} threshold={udec['threshold']}")


def _test_rows(table, setting):
    from .evaluate import eval_mask
    return np.flatnonzero(eval_mask(table, setting, "test"))


def step_test():
    """The ONLY step that touches the test partition for metrics; run once after finalisation."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    from . import dataset as D, errors as ER
    from .config import MODELS_DIR
    from .dataset import feature_columns
    from .metrics import binary_metrics, cluster_bootstrap_ci
    from .predict import load_bundle
    table = D.load_table()
    out, preds, curves = {}, [], []
    models = [("deployed", MODELS_DIR / "final_model.joblib")]
    if "primary" in _configs():
        models.append(("primary", MODELS_DIR / "primary_model.joblib"))
    models.append(("user_only", MODELS_DIR / "user_only_model.joblib"))
    for tag, path in models:
        b = load_bundle(path)
        setting = b.metadata["training_setting"]
        rows = _test_rows(table, setting)
        d = table.iloc[rows][["file_id", "path", "start", "end", "label", "source", "generator", "group"]].copy()
        d["p_ai"] = b.pipeline.predict_proba(table.iloc[rows][feature_columns(b.feature_set)].to_numpy())[:, 1]
        d["model"] = tag
        preds.append(d)
        res = {"setting": setting, "model": b.metadata["model_name"], "feature_set": b.feature_set,
               "threshold": b.threshold, "scopes": {}}
        scopes = [("all test windows", d)] + [(f"source: {s}", x) for s, x in d.groupby("source")]
        scopes += [("user data (both classes)", d[d.source != "asvspoof2019_la"])]
        if (d.source == "asvspoof2019_la").any():
            scopes += [("supplementary: ASVspoof eval", d[d.source == "asvspoof2019_la"])]
        for name, x in scopes:
            m = binary_metrics(x.label.values, x.p_ai.values, b.threshold)
            if x.label.nunique() == 2:
                _, lo, hi = cluster_bootstrap_ci(x.label.values, x.p_ai.values, x.group.values, n_boot=1000, seed=SEED)
                m["roc_auc_ci95"] = [lo, hi]
            fl = ER.file_level(x)
            m["file_level"] = binary_metrics(fl.label.values, fl.p_ai.values, b.threshold)
            res["scopes"][name] = m
        out[tag] = res
        if d.label.nunique() == 2:
            for scope, x in [("all", d), ("user", d[d.source != "asvspoof2019_la"]), ("asvspoof", d[d.source == "asvspoof2019_la"])]:
                if len(x) == 0 or x.label.nunique() < 2:
                    continue
                fpr, tpr, thr = roc_curve(x.label, x.p_ai)
                pr, rc, _ = precision_recall_curve(x.label, x.p_ai)
                curves.append(pd.DataFrame({"model": tag, "scope": scope, "curve": "roc", "x": fpr, "y": tpr}))
                curves.append(pd.DataFrame({"model": tag, "scope": scope, "curve": "pr", "x": rc, "y": pr}))
    pd.concat(preds).to_csv(REPORTS_DIR / "test_predictions.csv", index=False)
    pd.concat(curves).to_csv(REPORTS_DIR / "test_curves.csv", index=False)
    _json(REPORTS_DIR / "test_metrics.json", out)
    for tag, r in out.items():
        for scope, m in r["scopes"].items():
            _log(f"{tag:<10} {scope:<35} n={m['n']:<5} AUC={m['roc_auc']:.3f} bAcc={m['balanced_accuracy']:.3f} "
                 f"recall={m['recall']:.3f} spec={m['specificity']:.3f}")


# ------------------------------------------------------------------------------------- analyses
def _deployed_cfg(bundle) -> dict:
    m = bundle.metadata
    return {"setting": m["training_setting"], "model": m["model_name"], "feature_set": bundle.feature_set,
            "params": m["best_params"], "aug": tuple(m.get("augmentation", [])),
            "calibration": m.get("calibration"), "threshold": bundle.threshold}


def step_robustness():
    from . import dataset as D, robustness as R
    from .config import MODELS_DIR
    from .predict import load_bundle
    table = D.load_table()
    for which in _configs():
        b = load_bundle(MODELS_DIR / ("final_model.joblib" if which == "deployed" else f"{which}_model.joblib"))
        rows = _test_rows(table, b.metadata["training_setting"])
        test = table.iloc[rows][["file_id", "path", "start", "end", "label", "source", "generator", "group"]]
        summary, preds = R.run(b, test)
        summary.to_csv(REPORTS_DIR / f"robustness{_sfx(which)}.csv", index=False)
        preds.to_parquet(REPORTS_DIR / f"robustness_predictions{_sfx(which)}.parquet", index=False)
        _log(f"== {which}")
        _log(summary[summary.scope == "all test windows"][["condition", "n", "balanced_accuracy", "roc_auc",
                                                           "recall_ai", "specificity_human"]].round(3).to_string())


def step_generalisation():
    from . import dataset as D, generalisation as G
    from .predict import load_bundle
    b = load_bundle()
    cfg = _deployed_cfg(b)
    table = D.load_table()
    g = G.run(cfg, table)
    g.to_csv(REPORTS_DIR / "generalisation.csv", index=False)
    x = G.cross_dataset(cfg, table)
    x.to_csv(REPORTS_DIR / "cross_dataset.csv", index=False)
    _log(g.groupby("experiment")[["recall_ai", "specificity_human", "balanced_accuracy", "roc_auc"]].agg(["mean", "std"]).round(3).to_string())
    _log(x.round(3).to_string())


def step_explain():
    from . import dataset as D, explain as X
    from .predict import load_bundle
    b = load_bundle()
    cfg = _deployed_cfg(b)
    table = D.load_table()
    # importance / ablation use the base model with the final hyperparameters (calibration=None) and
    # rank-based ROC-AUC, which a monotone calibration step leaves essentially unchanged
    pf, pg = X.permutation_analysis(table, cfg["setting"], cfg["feature_set"], cfg["model"], cfg["params"],
                                    cfg["aug"], None)
    pf.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)
    pg.to_csv(REPORTS_DIR / "group_permutation_importance.csv", index=False)
    ab = X.group_ablation(table, cfg["setting"], cfg["model"], cfg["params"], cfg["aug"], None)
    ab.to_csv(REPORTS_DIR / "feature_group_ablation.csv", index=False)
    pr = X.projections(table, cfg["setting"], cfg["feature_set"])
    pr.to_csv(REPORTS_DIR / "projection_coords.csv", index=False)
    _json(REPORTS_DIR / "projection_info.json", {"pca_explained_variance": pr.attrs["pca_explained_variance"],
                                                 "setting": cfg["setting"], "feature_set": cfg["feature_set"],
                                                 "n_points": int(len(pr))})
    _log(pg.round(4).to_string())
    _log(ab.round(4).to_string())


def step_errors():
    from . import errors as ER
    from .predict import load_bundle
    b = load_bundle()
    oof = pd.read_parquet(REPORTS_DIR / "oof_final_config.parquet")
    test = pd.read_csv(REPORTS_DIR / "test_predictions.csv")
    test = test[test.model == "deployed"]
    from . import dataset as D
    table = D.load_table()
    loc = table.iloc[oof.row.to_numpy()]  # attach path/start/end so the UI can replay each window
    oof = oof.assign(path=loc.path.values, start=loc.start.values, end=loc.end.values)
    for name, d in [("oof", oof), ("test", test)]:
        ER.misclassified(d, b.threshold, b.halfwidth).to_csv(REPORTS_DIR / f"errors_{name}_windows.csv", index=False)
        fl = ER.file_level(d)
        ER.misclassified(fl, b.threshold, b.halfwidth).to_csv(REPORTS_DIR / f"errors_{name}_files.csv", index=False)
        ER.summary(d, b.threshold).to_csv(REPORTS_DIR / f"error_summary_{name}.csv", index=False)
    _log(ER.summary(test, b.threshold).round(3).to_string())
    _log(f"misclassified test files: {len(ER.misclassified(ER.file_level(test), b.threshold, b.halfwidth))} of {test.file_id.nunique()}")


def step_learning():
    from . import dataset as D, generalisation as G
    from .config import N_OUTER_SPLITS
    from .dataset import feature_columns
    from .evaluate import eval_mask
    from .predict import load_bundle
    from .splits import group_kfold
    b = load_bundle()
    cfg = _deployed_cfg(b)
    table = D.load_table()
    meta = table.drop(columns=feature_columns("extended"))
    rows = np.flatnonzero(eval_mask(table, cfg["setting"], "dev"))
    lc = G.learning_curve({**cfg, "calibration": None}, table, group_kfold(meta, rows, N_OUTER_SPLITS, SEED))
    lc.to_csv(REPORTS_DIR / "learning_curve.csv", index=False)
    _log(lc.groupby("fraction_of_training_groups")[["n_train_windows", "n_train_groups", "roc_auc"]].mean().round(3).to_string())


def step_shortcuts():
    """Single-feature AUCs per data source (development partition only)."""
    from . import dataset as D, explain as X
    from .evaluate import eval_mask
    t = D.load_table()
    asv = t.source.eq("asvspoof2019_la").values
    scopes = {"user data (as stored)": eval_mask(t, "user", "dev"),
              "user data (codec-matched)": eval_mask(t, "user_codec_matched", "dev"),
              "ASVspoof 2019 LA": eval_mask(t, "combined", "dev") & asv}
    u = X.univariate_auc(t, scopes)
    u.to_csv(REPORTS_DIR / "shortcut_univariate_auc.csv", index=False)
    summary = {s: {"n_windows": int(d.n.iloc[0]), "max_single_feature_auc": float(d.auc_folded.max()),
                   "best_feature": d.iloc[0].feature, "features_auc_ge_0.90": int((d.auc_folded >= 0.90).sum()),
                   "features_auc_ge_0.95": int((d.auc_folded >= 0.95).sum())}
               for s, d in u.groupby("scope")}
    _json(REPORTS_DIR / "shortcut_summary.json", summary)
    _log(json.dumps(summary, indent=2))


def step_figures():
    from . import figures
    written = figures.build_all()
    _log(f"figures: {len(written)} PNGs in reports/figures")


def step_report():
    from . import report
    _log(f"report: {report.build()}")


def step_readme():
    from . import readme
    _log(f"README results block {'updated' if readme.update_readme() else 'already up to date'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", help=f"one or more of: {', '.join(STEPS)} or 'all'")
    args = ap.parse_args(argv)
    steps = list(STEPS) if args.steps == ["all"] else args.steps
    for s in steps:
        if s not in STEPS:
            ap.error(f"unknown step {s!r}")
    for s in steps:
        t0 = time.time()
        _log(f"\n### step: {s}")
        STEPS[s]()
        _log(f"### step {s} finished in {time.time() - t0:.0f}s")


STEPS = {
    "manifest": step_manifest,
    "audit": step_audit,
    "features": step_features,
    "features_aug": step_features_aug,
    "legacy": step_legacy,
    "compare": step_compare,
    "shortcuts": step_shortcuts,
    "select": step_select,
    "augment": step_augment,
    "robust": step_robust,
    "calibrate": step_calibrate,
    "finalize": step_finalize,
    "test": step_test,
    "robustness": step_robustness,
    "generalisation": step_generalisation,
    "explain": step_explain,
    "errors": step_errors,
    "learning": step_learning,
    "figures": step_figures,
    "report": step_report,
    "readme": step_readme,
}

if __name__ == "__main__":
    sys.exit(main())
