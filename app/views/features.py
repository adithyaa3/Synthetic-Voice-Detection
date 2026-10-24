import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.features import feature_names


@lib.safe_page
def render():
    lib.page_header("Feature insights", "Which acoustic properties separate the classes, and how much each feature "
                                        "group contributes under honest validation.")
    m = lib.mode()
    p = charts.P(m)
    meta = lib.model_metadata()

    st.subheader("Feature distributions")
    table = lib.feature_table()
    dev = table[(table.variant == "orig") & (table.partition == "dev")]
    c1, c2, c3 = st.columns([1.2, 1, 1])
    family = c1.selectbox("Feature family", ["MFCC mean", "MFCC std", "Delta-MFCC std", "Spectral / energy"], key="fi_family")
    scope = c2.segmented_control("Data", ["User data", "ASVspoof", "All"], default="User data", key="fi_scope")
    names = feature_names("extended")
    if family == "Spectral / energy":
        pick = c3.selectbox("Feature", [n for n in names if n.split("_")[0] in
                                        ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms")], key="fi_spec")
    else:
        k = c3.slider("Coefficient", 0, 39, 1, key="fi_coef")
        pick = {"MFCC mean": f"mfcc{k:02d}_mean", "MFCC std": f"mfcc{k:02d}_std", "Delta-MFCC std": f"dmfcc{k:02d}_std"}[family]
    d = dev if scope == "All" else dev[(dev.source == "asvspoof2019_la") == (scope == "ASVspoof")]
    fig = go.Figure()
    for cls, col, lab in (("human", p["human"], "Human"), ("ai", p["ai"], "AI-generated")):
        v = d.loc[d.class_name == cls, pick]
        fig.add_violin(x=v, y=[lab] * len(v), name=lab, orientation="h", line_color=col, fillcolor=charts._rgba(col, 0.25),
                       box_visible=True, meanline_visible=True, points=False, spanmode="hard",
                       hovertemplate=f"{lab}<br>{pick}: %{{x:.3f}}<extra></extra>")
    fig.update_xaxes(title=pick)
    stats = d.groupby("class_name")[pick].describe()[["count", "mean", "std", "25%", "50%", "75%"]].round(3).reset_index()
    lib.chart(charts.style(fig, m, height=300), "fi_violin", data=stats,
              caption=f"Development windows only ({len(d):,}); violins show the full distribution, boxes the quartiles.")

    st.subheader("2-D projections")
    coords = lib.t("projection_coords.csv")
    info = lib.j("projection_info.json")
    meth = st.segmented_control("Method", ["t-SNE", "PCA"], default="t-SNE", key="fi_proj")
    meth = "tsne" if meth != "PCA" else "pca"
    lib.chart(charts.projection(coords, meth, m), f"fi_proj_{meth}",
              caption=(f"{info['n_points']:,} development windows, {info['feature_set']} features, standardised. "
                       f"PCA explains {sum(info['pca_explained_variance']):.0%} of variance in 2 components. "
                       "Colour = class, shape = dataset. t-SNE preserves local neighbourhoods only; distances between "
                       "clusters are not meaningful."))

    st.subheader("What the final model relies on")
    c1, c2 = st.columns(2, gap="large")
    gi = lib.t("group_permutation_importance.csv")
    pi = lib.t("permutation_importance.csv")
    with c1:
        lib.chart(charts.importance_bars(gi, "group", "auc_drop", "auc_drop_fold_std", m), "fi_gimp", data=gi,
                  caption="Joint permutation of each feature group on validation folds (5 repeats x 5 folds): drop in ROC-AUC. "
                          "Correlated features make single-feature importance unreliable, so groups are the main view.")
    with c2:
        top = pi.head(15)
        lib.chart(charts.importance_bars(top, "feature", "importance", "importance_fold_std", m), "fi_top", data=pi.head(40),
                  caption="Top 15 single features (sklearn permutation importance, ROC-AUC).")

    st.subheader("Feature-group ablation")
    ab = lib.t("feature_group_ablation.csv")
    ab = ab.assign(**{"ROC-AUC (mean ± std)": [lib.fmt_pm(a, b) for a, b in zip(ab.roc_auc_mean, ab.roc_auc_std)]})
    st.dataframe(ab[["config", "n_features", "ROC-AUC (mean ± std)"]], hide_index=True, width="stretch")
    st.caption(f"Same folds and fixed hyperparameters of the final model ({lib.LABELS.get(meta['model_name'])}); "
               "'drop X' removes one group, 'only X' keeps one group.")


render()
