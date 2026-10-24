import pandas as pd
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.config import REPORTS_DIR
from voicedetect.evaluate import SETTING_LABELS


@lib.safe_page
def render():
    lib.page_header("Synthetic voice detection",
                    "Is a voice recording human or AI-generated? A predictive-analysis study with leakage-safe "
                    "validation, honest uncertainty and a live demo.")
    meta = lib.model_metadata()
    test = lib.j("test_metrics.json")
    fsum = lib.j("feature_summary.json")
    audit = lib.j("dataset_audit.json")
    allm = test["deployed"]["scopes"]["all test windows"]
    usr = test["deployed"]["scopes"].get("user data (both classes)")
    cv = meta["cv_metrics"]

    c = st.columns(4)
    c[0].metric("Audio files analysed", f"{audit['files_total']:,}", help="User data plus the supplementary ASVspoof subset",
                border=True)
    c[1].metric("Analysis windows (4 s)", f"{fsum['windows_orig']:,}", border=True,
                help=f"{fsum['files_skipped']} files skipped by the loader (too short or silent)")
    ci = allm.get("roc_auc_ci95")
    c[2].metric("Test ROC-AUC", f"{allm['roc_auc']:.3f}", border=True,
                help=(f"All held-out test windows (unseen groups); 95% group-bootstrap CI {ci[0]:.3f}-{ci[1]:.3f}"
                      if ci else "All held-out test windows (unseen groups)"))
    c[3].metric("Test balanced accuracy", f"{allm['balanced_accuracy']:.3f}", border=True,
                help=f"AI recall {allm['recall']:.3f} · human specificity {allm['specificity']:.3f} at threshold {meta['threshold']:.2f}")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Final model")
        cal = lib.j("selection/calibration.json")
        aug_txt = ", ".join({"mp3cv": "MP3 re-encoding", "aug_noise": "random noise", "aug_mic": "random microphone chain"}
                            .get(a, a) for a in meta["augmentation"]) or "none"
        dev_auc = cal["methods"][cal["chosen_method"]]["roc_auc_mean"]
        st.markdown(
            f"- **{lib.LABELS.get(meta['model_name'], meta['model_name'])}** on **{meta['feature_set']}** features, "
            f"trained on **{SETTING_LABELS[meta['training_setting']]}** ({meta['training_windows']:,} windows, "
            f"{meta['training_groups']} independent groups)\n"
            f"- Training augmentation: **{aug_txt}**; probability calibration: **{meta['calibration']}**\n"
            f"- Decision threshold **{meta['threshold']:.2f}** (cost-based, chosen on validation data), "
            f"inconclusive band **{meta['threshold'] - meta['inconclusive_halfwidth']:.2f}-{meta['threshold'] + meta['inconclusive_halfwidth']:.2f}**\n"
            f"- Cross-validated ROC-AUC of this configuration on development data: **{dev_auc:.3f}** "
            f"(model-family comparison without augmentation: {lib.fmt_pm(cv['roc_auc_mean'], cv['roc_auc_std'])})\n"
            f"- Model version `{meta['model_version']}`")
        mic = cal.get("dev_robustness_at_threshold", {}).get("mic")
        if mic and mic["specificity"] < 0.8:
            st.error(f"**Microphone input is unreliable.** On development data passed through a microphone-like chain, "
                     f"only **{mic['specificity']:.0%}** of real human windows are cleared at the chosen threshold, so a "
                     "real person recorded on a laptop microphone is likely to be flagged as AI. See the robustness study "
                     "on the Model comparison page for the noise-robust training variant and its trade-off.",
                     icon=":material/mic_off:")
        elif mic:
            st.info(f"On development data passed through a microphone-like chain, {mic['specificity']:.0%} of human "
                    f"windows are cleared and {mic['recall']:.0%} of AI windows flagged at the chosen threshold.",
                    icon=":material/mic:")
        if usr:
            st.markdown(f"On the user's own held-out data (unseen source speakers + unseen Common Voice groups) the "
                        f"test ROC-AUC is **{usr['roc_auc']:.3f}** with balanced accuracy **{usr['balanced_accuracy']:.3f}**.")
        asv = test["deployed"]["scopes"].get("supplementary: ASVspoof eval")
        if asv:
            st.warning(
                f"**Unseen generators are the weak spot.** On the ASVspoof evaluation partition (unseen speakers and "
                f"mostly unseen attack systems) ROC-AUC falls to **{asv['roc_auc']:.3f}**, and at the chosen threshold "
                f"only **{asv['recall']:.0%}** of synthetic clips are flagged (human specificity {asv['specificity']:.0%}). "
                "Treat a 'human' verdict on audio from an unknown generator with caution.", icon=":material/warning:")
        st.subheader("How it was validated")
        st.markdown(
            "- **Group-aware splits** (StratifiedGroupKFold): all re-voicings of one source recording, all clips of one "
            "Common Voice recording session and all clips of one ASVspoof speaker stay on the same side.\n"
            "- **Nested CV**: hyperparameters tuned inside each training fold only; 5 outer folds, mean ± std reported.\n"
            "- **Held-out test set** chosen by group before modelling: unseen source speakers, unseen Common Voice "
            "groups, and the ASVspoof evaluation partition (unseen speakers and attack systems). It was first used for "
            "the pre-registered model; the deployed model's test numbers are post-hoc (see *Evaluation*).\n"
            "- **Automated leakage tests** assert that no group or file hash crosses a split.")
    with right:
        st.subheader("The original 100 % was leakage")
        legacy = lib.j("legacy_reproduction.json")
        lib.chart(charts.legacy_vs_honest(legacy, lib.mode()), "ov_legacy",
                  caption=f"{len(legacy['original_split']['test_ai_files_with_same_source_recording_in_train'])} of "
                          f"{legacy['original_split']['n_test_ai']} AI test files in the original split shared their "
                          "source recording with training files.")

    st.subheader("Data sources")
    rows = []
    for src, v in audit["per_source"].items():
        rows.append({"Source": lib.SOURCE_LABELS.get(src, src), "Origin": v["origin"], "Files": v["files"],
                     "Windows": v["windows_total"], "Groups": v["groups"], "Group confidence": v["group_confidence"],
                     "Sample rates (Hz)": ", ".join(v["sample_rates"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Key limitations")
    st.markdown(
        "- The user's AI audio is **one voice-conversion tool, 8 source recordings**; the human audio is crowd-sourced "
        "Common Voice MP3. Recording conditions differ between the classes, so some of what a model learns may be "
        "channel, not synthesis. The codec-matched control and the ASVspoof data address this only partly.\n"
        "- Detection of **unseen generators** is much weaker than detection of seen ones (see *Robustness & generalisation*).\n"
        "- **Microphone recordings** differ from all training data; treat live results as indicative.\n"
        "- The output is a **probability, not proof**. Never use it as the sole evidence against a person.")

    st.subheader("Download")
    c1, c2, c3 = st.columns(3)
    rep = REPORTS_DIR / "report.html"
    if rep.exists():
        c1.download_button("Full report (HTML)", rep.read_bytes(), "voice_detection_report.html", "text/html",
                           icon=":material/description:", type="primary")
    else:
        c1.caption("Report not generated yet (pipeline step `report`).")
    c2.download_button("Model comparison (CSV)", lib.t("model_comparison.csv").to_csv(index=False).encode(),
                       "model_comparison.csv", "text/csv", icon=":material/table:")
    c3.download_button("Test predictions (CSV)", lib.t("test_predictions.csv").to_csv(index=False).encode(),
                       "test_predictions.csv", "text/csv", icon=":material/table:")


render()
