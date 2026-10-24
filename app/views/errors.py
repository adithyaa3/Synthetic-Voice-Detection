import pandas as pd
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.config import ROOT


@lib.safe_page
def render():
    lib.page_header("Error analysis", "Which clips the model gets wrong, how confidently, and which sources or "
                                      "generators fool it.")
    m = lib.mode()
    meta = lib.model_metadata()
    which = st.segmented_control("Predictions", ["Test set", "Validation (out-of-fold)"], default="Test set",
                                 key="er_which") or "Test set"
    tag = "test" if which == "Test set" else "oof"
    summ = lib.t(f"error_summary_{tag}.csv")
    st.subheader("Error rate by source and generator")
    st.dataframe(summ.assign(source=summ.source.map(lib.SOURCE_LABELS).fillna(summ.source)).round(3),
                 hide_index=True, width="stretch")
    st.caption(f"At the deployed threshold {meta['threshold']:.2f}, window level.")

    st.subheader("Misclassified clips")
    level = st.segmented_control("Level", ["Windows", "Files (mean over windows)"], default="Windows", key="er_level") or "Windows"
    err = lib.t(f"errors_{tag}_{'windows' if level == 'Windows' else 'files'}.csv")
    if err.empty:
        st.success("No misclassified items at this level.")
        return
    c1, c2, c3 = st.columns(3)
    et = c1.multiselect("Error type", sorted(err.error_type.unique()), default=sorted(err.error_type.unique()), key="er_type")
    src = c2.multiselect("Source", sorted(err.source.unique()), default=sorted(err.source.unique()),
                         format_func=lambda s: lib.SOURCE_LABELS.get(s, s), key="er_src")
    min_conf = c3.slider("Minimum distance from threshold", 0.0, 0.5, 0.0, 0.05, key="er_conf",
                         help="0 shows everything; larger values keep only confident mistakes.")
    view = err[err.error_type.isin(et) & err.source.isin(src) & (err.confidence_distance >= min_conf)].copy()
    cols = [c for c in ["file_id", "source", "generator", "truth", "predicted", "p_ai", "error_type", "inconclusive",
                        "n_windows", "start", "path", "end"] if c in view.columns]
    view = view[cols].reset_index(drop=True)
    st.caption(f"{len(view)} of {len(err)} misclassified items match the filters. Select a row to listen to it.")
    ev = st.dataframe(view.drop(columns=[c for c in ("path", "end") if c in view.columns]).round(3), hide_index=True,
                      width="stretch", on_select="rerun", selection_mode="single-row", key=f"er_table_{tag}_{level}",
                      height=min(460, 38 + 35 * max(1, len(view))))
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if rows:
        r = view.iloc[rows[0]]
        st.markdown(f"**{r.file_id}** · truth **{r.truth}**, predicted **{r.predicted}** with P(AI) = {r.p_ai:.3f}")
        if "start" in r and pd.notna(r.get("start")) and (ROOT / r.path).exists():
            with st.spinner("Loading audio ..."):
                y, sr = lib.window_audio(r.path, int(r.start), int(r.end))
            st.audio(lib.wav_bytes(y, sr), format="audio/wav")
            lib.chart(charts.mel_spectrogram(y, sr, m, height=220), f"er_mel_{r.file_id}_{r.start}")
        elif "start" not in r or pd.isna(r.get("start")):
            st.caption("File-level rows aggregate several windows; switch to 'Windows' to listen to a specific one.")
        else:
            st.warning("This audio file is not present on this machine.")


render()
