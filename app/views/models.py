import pandas as pd
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.evaluate import SETTING_LABELS

METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("balanced_accuracy", "Balanced acc."), ("f1", "F1"),
           ("precision", "Precision"), ("recall", "Recall"), ("accuracy", "Accuracy"), ("brier", "Brier"),
           ("log_loss", "Log loss"), ("ece", "ECE")]


@lib.safe_page
def render():
    lib.page_header("Model comparison", "Six model families under identical group-aware folds and features; "
                                        "hyperparameters tuned inside each training fold only.")
    m = lib.mode()
    summ = lib.t("model_comparison.csv")
    sel = lib.j("selection/model_selection.json")
    aug = lib.j("selection/augmentation.json")
    meta = lib.model_metadata()

    setting = st.segmented_control("Data setting", list(SETTING_LABELS), default="combined",
                                   format_func=lambda s: SETTING_LABELS[s], key="mc_setting") or "combined"
    chosen = sel["selection_by_setting"][setting]["chosen"]
    lib.chart(charts.model_comparison(summ, setting, (chosen["model"], chosen["feature_set"]), m), f"mc_bars_{setting}",
              caption="Dots: mean ROC-AUC over 5 outer group folds; whiskers: ± 1 std; vertical line: chance (0.5). The "
                      "highlighted dot is the model the pre-declared rule selects in this setting.")

    s = summ[(summ.setting == setting) & (summ.augment == "none") & (summ.calibration == "default")].copy()
    out = pd.DataFrame({"Model": s.model.map(lib.LABELS).fillna(s.model), "Features": s.feature_set})
    for k, lab in METRICS:
        out[lab] = [lib.fmt_pm(a, b) for a, b in zip(s[f"{k}_mean"], s[f"{k}_std"])]
    if "roc_auc_mp3_mean" in s and s.roc_auc_mp3_mean.notna().any():
        out["ROC-AUC on MP3 copies"] = s.roc_auc_mp3_mean.map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/a")
    out["Fit time / fold (s)"] = s.fit_seconds_mean.round(1)
    st.dataframe(out, hide_index=True, width="stretch")
    st.download_button("Download this table (CSV)", out.to_csv(index=False).encode(), f"model_comparison_{setting}.csv",
                       "text/csv", icon=":material/download:", key="mc_dl")
    st.caption("'ROC-AUC on MP3 copies' scores the same validation windows after re-encoding them to 32 kHz / 48 kbps "
               "MP3 (the Common Voice format): a development-only check of sensitivity to compression.")

    st.subheader("Why this model")
    rule_tab = pd.DataFrame(sel["selection_by_setting"][setting]["table"])
    rule_tab["Model"] = rule_tab.model.map(lib.LABELS).fillna(rule_tab.model)
    best = sel["selection_by_setting"][setting]["best_by_mean_auc"]
    is_ref = (rule_tab.model == best["model"]) & (rule_tab.feature_set == best["feature_set"])
    rule_tab["AUC diff vs best (95% CI)"] = ["reference (best mean AUC)" if ref else f"{-d:+.4f} ({-hi:+.4f} to {-lo:+.4f})"
                                             for ref, d, lo, hi in zip(is_ref, rule_tab.auc_diff_best_minus_this,
                                                                       rule_tab.ci_low, rule_tab.ci_high)]
    st.markdown(f"**Rule (fixed before looking at results):** {sel['rule']}")
    st.markdown(f"In this setting the highest mean ROC-AUC is **{lib.LABELS.get(best['model'])} / {best['feature_set']}** "
                f"({best['roc_auc_mean']:.3f}). The rule selects **{lib.LABELS.get(chosen['model'])} / {chosen['feature_set']}** "
                f"({lib.fmt_pm(chosen['roc_auc_mean'], chosen['roc_auc_std'])}).")
    st.dataframe(rule_tab[["Model", "feature_set", "roc_auc_mean", "roc_auc_std", "AUC diff vs best (95% CI)", "tied_with_best"]]
                 .rename(columns={"feature_set": "Features", "roc_auc_mean": "ROC-AUC mean", "roc_auc_std": "std",
                                  "tied_with_best": "Tied with best"}),
                 hide_index=True, width="stretch")
    udc = sel["user_domain_check"]
    st.markdown(
        f"**Deployed model:** {lib.LABELS.get(meta['model_name'])} / {meta['feature_set']} trained on "
        f"**{SETTING_LABELS[meta['training_setting']]}**. On the user's own validation windows, training with the "
        f"supplementary data gives out-of-fold ROC-AUC {udc['oof_auc_on_user_rows_trained_combined']:.3f} versus "
        f"{udc['oof_auc_on_user_rows_trained_user_only']:.3f} for user data alone (tolerance {udc['tolerance']}). "
        f"First augmentation rule (MP3 only, clean-AUC tolerance 0.005): clean CV AUC {aug['no_augmentation']['roc_auc_mean']:.3f} -> "
        f"{aug['mp3_augmentation']['roc_auc_mean']:.3f}, AUC on MP3 copies {aug['no_augmentation']['roc_auc_mp3_mean']:.3f} -> "
        f"{aug['mp3_augmentation']['roc_auc_mp3_mean']:.3f}; adopted: **{'yes' if aug['adopted'] else 'no'}**. "
        f"Deployed model's training augmentation: **{', '.join(meta['augmentation']) or 'none'}** (see the robustness study below).")

    try:
        rs = lib.j("selection/robustness_study.json")
    except lib.MissingArtifact:
        rs = None
    if rs:
        st.subheader("Robustness-aware augmentation study (development data only)")
        st.warning(rs["motivation"], icon=":material/history:")
        lib.chart(charts.robust_tradeoff(rs, m), "mc_robust")
        tab = pd.DataFrame([{"Candidate": r["candidate"], **{f"AUC {k}": round(v, 3) for k, v in r["auc_by_condition"].items()},
                             "Mean (robust AUC)": round(r["robust_auc"], 3)} for r in rs["candidates"]])
        st.dataframe(tab, hide_index=True, width="stretch")
        st.markdown(f"**Rule (fixed before the study ran):** {rs['rule']}  \n**Chosen:** `{rs['chosen']}`.")
        st.caption("Conditions: validation windows re-encoded as 48 kbps MP3, with white or pink 'room' noise at "
                   "20 dB SNR, or passed through a microphone-like chain. Hyperparameters: " + rs["hyperparameters"] + ".")

    with st.expander("Paired comparisons (same folds)"):
        pc = lib.t("paired_comparisons.csv")
        st.dataframe(pc[pc.setting == setting], hide_index=True, width="stretch")
        st.caption("Wilcoxon signed-rank over 5 folds cannot reach p < 0.05 (minimum two-sided p = 0.0625), so the "
                   "paired group-bootstrap CI of the pooled out-of-fold AUC difference is the main comparison.")

    st.subheader("RBF SVM tuning surface")
    grid = lib.t("rbf_svm_grid.csv")
    fs = st.segmented_control("Feature set", ["baseline", "extended"], default="baseline", key="mc_fs") or "baseline"
    lib.chart(charts.rbf_heatmap(grid, setting, fs, m), f"mc_heat_{setting}_{fs}",
              caption="Inner-CV ROC-AUC averaged over the 5 outer folds. The original script used C=10, gamma='scale' "
                      "(= 1/n_features after standardisation: 0.0125 for 80 features, 0.0048 for 210).")


render()
