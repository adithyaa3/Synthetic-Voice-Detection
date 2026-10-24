import streamlit as st

import lib
from voicedetect import charts


@lib.safe_page
def render():
    lib.page_header("Robustness & generalisation", "How the final model holds up under noise, compression, short "
                                                   "clips and microphone-like audio, and on generators and speakers it never saw.")
    m = lib.mode()
    rob = lib.t("robustness.csv")
    gen = lib.t("generalisation.csv")
    xd = lib.t("cross_dataset.csv")

    st.subheader("Perturbations of the test windows")
    try:
        rob_primary = lib.t("robustness_primary.csv")
    except lib.MissingArtifact:
        rob_primary = None
    if rob_primary is not None:
        which = st.segmented_control("Model", ["Deployed model", "Pre-registered model"], default="Deployed model",
                                     key="rb_model") or "Deployed model"
        if which == "Pre-registered model":
            rob = rob_primary
        st.caption("The pre-registered model's noise sensitivity motivated the development-data robustness study "
                   "(Model comparison page) that produced the deployed model.")
    scopes = list(dict.fromkeys(rob.scope))
    scope = st.selectbox("Scope", scopes, format_func=lambda s: s.replace("source: ", "").replace(
        "asvspoof2019_la", "ASVspoof eval").replace("user_commonvoice", "User Common Voice (human only)").replace(
        "user_voice_conversion", "User voice conversion (AI only)"), key="rb_scope")
    fams = st.pills("Families", sorted(rob.family.unique()), selection_mode="multi", default=sorted(rob.family.unique()),
                    key="rb_fam")
    view = rob[(rob.scope == scope) & rob.family.isin(fams or [])]
    if view.empty:
        st.info("Select at least one family.")
    else:
        lib.chart(charts.robustness_dots(view, m, scope), f"rb_dots_{scope}",
                  data=view[["description", "n", "balanced_accuracy", "roc_auc", "recall_ai", "specificity_human",
                             "mean_p_ai_human", "mean_p_ai_ai"]].round(3),
                  caption="Each condition is applied to every test window before feature extraction; the thin vertical "
                          "line marks clean balanced accuracy. MP3 encoding uses libsndfile's built-in LAME encoder.")

    st.subheader("Unseen generators, speakers and voices")
    st.markdown("These experiments re-train the final configuration (fixed hyperparameters) on everything except the "
                "held-out factor. They run after model selection and never influenced it.")
    summary = gen.groupby("experiment").agg(folds=("held_out", "size"), ai_recall=("recall_ai", "mean"),
                                            human_specificity=("specificity_human", "mean"),
                                            balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean"),
                                            roc_auc_std=("roc_auc", "std")).reset_index().round(3)
    st.dataframe(summary, hide_index=True, width="stretch")
    t1, t2, t3, t4 = st.tabs(["Leave one generator out", "Leave one source speaker out", "Leave one target voice out",
                              "Cross-dataset"])
    with t1:
        lib.chart(charts.generalisation_bars(gen, "LOGO", m), "rb_logo",
                  data=gen[gen.experiment.str.startswith("LOGO")].round(3),
                  caption="user_vc = the user's voice-conversion data; A01-A19 = ASVspoof 2019 systems (type from the "
                          "dataset's README). The held-out generator's speakers are removed from training too.")
        uv = gen[gen.experiment.str.startswith("LOGO") & (gen.held_out == "user_vc")]
        if len(uv):
            r0 = uv.iloc[0]
            st.warning(f"Trained on all 19 ASVspoof systems but not on the user's conversion tool, the model flags "
                       f"**{r0.recall_ai:.0%}** of the user's {int(r0.n_test_ai)} AI windows (ROC-AUC {r0.roc_auc:.3f}). "
                       "Detection does not transfer to an unseen generator.", icon=":material/warning:")
    with t2:
        lib.chart(charts.generalisation_bars(gen, "LOSO", m), "rb_loso",
                  data=gen[gen.experiment.str.startswith("LOSO")].round(3),
                  caption="User data only: all 7 conversions of the held-out source recording are unseen.")
    with t3:
        lib.chart(charts.generalisation_bars(gen, "LOTO", m), "rb_loto",
                  data=gen[gen.experiment.str.startswith("LOTO")].round(3),
                  caption="Relaxed: other conversions of the same source recordings remain in training (each source was "
                          "converted into every target voice), so this measures timbre generalisation only.")
    with t4:
        st.dataframe(xd.round(3), hide_index=True, width="stretch")
        st.caption("Train on one data source, test on the other: the hardest and most realistic check of domain shift.")

    st.subheader("Would more data help?")
    lc = lib.t("learning_curve.csv")
    lib.chart(charts.learning_curve(lc, m), "rb_lc",
              data=lc.groupby("fraction_of_training_groups")[["n_train_groups", "n_train_windows", "roc_auc"]].mean().round(3).reset_index(),
              caption="Training on a random fraction of the training groups of each outer fold (3 repeats), scoring the "
                      "full validation fold.")


render()
