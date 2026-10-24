import numpy as np
import pandas as pd
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.calibration import threshold_table
from voicedetect.metrics import binary_metrics


@lib.safe_page
def render():
    lib.page_header("Evaluation", "Performance on the held-out test set (see the order of events below), calibration "
                                  "of the probabilities, and the decision threshold.")
    m = lib.mode()
    meta = lib.model_metadata()
    test = lib.j("test_metrics.json")
    cal = lib.j("selection/calibration.json")

    st.subheader("Held-out test set")
    rows = []
    for tag, label in (("deployed", "Deployed model"), ("primary", "Pre-registered model (not deployed)"),
                       ("user_only", "User-data-only comparison")):
        if tag not in test:
            continue
        for scope, r in test[tag]["scopes"].items():
            ci = r.get("roc_auc_ci95")

            def f(v, na="n/a (one class)"):
                return na if v is None or pd.isna(v) else f"{v:.3f}"
            rows.append({"Model": label, "Scope": scope, "Windows": r["n"], "AI / human": f"{r['n_ai']} / {r['n_human']}",
                         "ROC-AUC": f(r["roc_auc"]), "95% CI": f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "",
                         "PR-AUC": f(r["pr_auc"]), "Balanced acc.": f(r["balanced_accuracy"]),
                         "AI recall": f(r["recall"]) if r["n_ai"] else "n/a (no AI)",
                         "Human specificity": f(r["specificity"]) if r["n_human"] else "n/a (no human)",
                         "Precision": f(r["precision"]) if r["n_ai"] else "n/a", "F1": f(r["f1"]) if r["n_ai"] else "n/a",
                         "Brier": f(r["brier"]), "File-level bal. acc.": f(r["file_level"]["balanced_accuracy"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Threshold {meta['threshold']:.2f} for the deployed model. CIs resample whole groups (1,000 bootstrap "
               "draws). File-level = mean probability over a file's windows, which is what the live demo reports.")
    if "primary" in test:
        st.info("**Order of events, for transparency:** the *pre-registered* model was chosen by rules fixed before any "
                "test-set evaluation, and its test results are the primary result. Its test-set robustness analysis then "
                "revealed noise sensitivity, which prompted a robustness study on development data (Model comparison "
                "page). The *deployed* model comes from that study, so its test results are **post-hoc**: the test set "
                "had already been looked at once.", icon=":material/info:")

    c1, c2 = st.columns([1.3, 1], gap="large")
    curves = lib.t("test_curves.csv")
    with c1:
        tab_roc, tab_pr = st.tabs(["ROC curve", "Precision-recall curve"])
        with tab_roc:
            lib.chart(charts.curves(curves, "deployed", "roc", m), "ev_roc")
        with tab_pr:
            lib.chart(charts.curves(curves, "deployed", "pr", m), "ev_pr")
    with c2:
        lib.chart(charts.confusion(test["deployed"]["scopes"]["all test windows"], m, title="All test windows"), "ev_cm")

    st.subheader("Calibration")
    rel = lib.t("reliability_oof.csv")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        lib.chart(charts.reliability(rel, cal["chosen_method"], m), "ev_rel", data=rel,
                  caption="Out-of-fold validation predictions (development data only), 10 equal-width bins.")
    with c2:
        mt = pd.DataFrame(cal["methods"]).T.reset_index().rename(columns={
            "index": "Method", "log_loss_mean": "Log loss", "brier_mean": "Brier", "ece_mean": "ECE (fold mean)",
            "roc_auc_mean": "ROC-AUC", "pooled_ece": "ECE (pooled)"})
        st.dataframe(mt.round(4), hide_index=True, width="stretch")
        st.markdown(f"Chosen: **{cal['chosen_method']}** ({cal['selection_rule']}).")

    st.subheader("Choose a threshold")
    st.markdown(cal["cost_rationale"])
    src = st.segmented_control("Predictions", ["Validation (out-of-fold)", "Test"], default="Validation (out-of-fold)",
                               key="ev_src") or "Validation (out-of-fold)"
    if src.startswith("Validation"):
        pr = lib.t("oof_final_config.parquet")[["label", "p_ai", "source"]]
    else:
        tp = lib.t("test_predictions.csv")
        pr = tp[tp.model == "deployed"][["label", "p_ai", "source"]]
    domain = st.segmented_control("Scope", ["All", "User data", "ASVspoof"], default="All", key="ev_dom") or "All"
    if domain != "All":
        pr = pr[(pr.source == "asvspoof2019_la") == (domain == "ASVspoof")]
    thr = st.slider("Decision threshold on P(AI)", 0.02, 0.98, float(meta["threshold"]), 0.01, key="ev_thr",
                    help="Recomputed live from the saved predictions; this does not change the deployed model.")
    if len(pr) == 0 or pr.label.nunique() < 2:
        st.warning("This selection contains only one class, so threshold metrics are not defined.")
        return
    mm = binary_metrics(pr.label.values, pr.p_ai.values, thr)
    k = st.columns(5)
    k[0].metric("AI recall", f"{mm['recall']:.3f}", border=True)
    k[1].metric("Human specificity", f"{mm['specificity']:.3f}", border=True)
    k[2].metric("Precision", f"{mm['precision']:.3f}", border=True)
    k[3].metric("F1", f"{mm['f1']:.3f}", border=True)
    k[4].metric("False alarms / misses", f"{mm['fp']} / {mm['fn']}", border=True)
    c1, c2 = st.columns([1.3, 1], gap="large")
    tt = threshold_table(pr.label.values, pr.p_ai.values, meta["decision_costs"]["false_alarm"], meta["decision_costs"]["miss"])
    with c1:
        lib.chart(charts.threshold_tradeoff(tt, thr, (thr - meta["inconclusive_halfwidth"], thr + meta["inconclusive_halfwidth"]), m),
                  "ev_tt", data=tt.round(4))
    with c2:
        lib.chart(charts.confusion(mm, m, title=f"At threshold {thr:.2f}"), "ev_cm_thr")
        band = np.abs(pr.p_ai.values - thr) < meta["inconclusive_halfwidth"]
        st.caption(f"{band.mean():.1%} of these windows would be shown as *inconclusive* (|P(AI) - threshold| < "
                   f"{meta['inconclusive_halfwidth']}). Deployed threshold: {meta['threshold']:.2f}.")


render()
