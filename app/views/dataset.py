import pandas as pd
import streamlit as st

import lib
from voicedetect import charts
from voicedetect.config import ROOT


@lib.safe_page
def render():
    lib.page_header("Dataset explorer", "What the model was trained and tested on, where it came from, and how "
                                        "independent the samples really are.")
    audit = lib.j("dataset_audit.json")
    fsum = lib.j("feature_summary.json")
    adf = lib.t("dataset_audit.csv")
    m = lib.mode()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.subheader("Class balance")
        lib.chart(charts.class_balance(fsum, m), "ds_balance",
                  data=pd.DataFrame([{"source/class": k, "windows": v} for k, v in fsum["windows_by_source_class"].items()]),
                  caption="Each long AI file contributes up to 10 non-overlapping 4 s windows; every window keeps "
                          "its file's group, so windows of one recording never sit on both sides of a split.")
    with c2:
        st.subheader("Durations")
        lib.chart(charts.duration_box(adf, m), "ds_dur",
                  caption="User AI files are 80-600 s long; human clips are a few seconds. Silence is trimmed "
                          "before analysis (this also removes the known leading-silence shortcut in ASVspoof 2019).")

    st.subheader("Sources, generators and groups")
    man = adf.copy()
    tab = (man.groupby(["source", "class_name", "generator", "partition"]).agg(files=("file_id", "size"),
                                                                              groups=("group", "nunique"))
           .reset_index().rename(columns={"class_name": "class"}))
    tab["source"] = tab.source.map(lib.SOURCE_LABELS).fillna(tab.source)
    f1, f2 = st.columns([1, 2])
    src_pick = f1.multiselect("Filter by source", sorted(tab.source.unique()), default=sorted(tab.source.unique()),
                              key="ds_src")
    part_pick = f2.segmented_control("Partition", ["dev", "test"], selection_mode="multi", default=["dev", "test"],
                                     key="ds_part")
    view = tab[tab.source.isin(src_pick) & tab.partition.isin(part_pick or [])]
    st.dataframe(view, hide_index=True, width="stretch", height=min(420, 38 + 35 * max(1, len(view))))
    st.caption(f"{len(view)} rows · 'dev' = used for model development (cross-validation); 'test' = held out until the end.")

    st.subheader("Is the dataset adequate?")
    rows = []
    for key, a in audit["adequacy"].items():
        for check, ok in a["checks"].items():
            rows.append({"dataset": a["dataset"], "check": check.replace("_", " "), "result": "pass" if ok else "fail"})
    ad = pd.DataFrame(rows).pivot(index="check", columns="dataset", values="result").reset_index()
    st.dataframe(ad, hide_index=True, width="stretch")
    st.markdown(
        f"User data alone has only **{audit['adequacy']['user_only']['independent_groups_ai']} independent AI source "
        f"recordings** from **one** generator, so the ASVspoof 2019 LA subset (ODC-By licence) was added and every "
        f"result is reported separately for user data and for user + supplementary data.")
    gi = audit["group_inference"]
    with st.expander("How groups were inferred (evidence)"):
        cv = gi["user_commonvoice"]["coherence_check"]
        st.markdown(
            f"- **Voice conversion (user AI):** {gi['user_voice_conversion']['basis']}. All 7 conversions of each source "
            f"have one identical duration ({gi['user_voice_conversion']['distinct_durations_per_source_group']}), so they "
            f"are re-voicings of the same recording. Confidence: {gi['user_voice_conversion']['confidence']}.\n"
            f"- **Common Voice (user human):** {gi['user_commonvoice']['basis']}. Acoustic check: the nearest neighbour of "
            f"{cv['nearest_neighbour_same_group']:.0%} of clips lies in the same inferred group (chance {cv['chance_level']:.1%}). "
            f"Confidence: {gi['user_commonvoice']['confidence']}.\n"
            f"- **ASVspoof:** {gi['asvspoof2019_la']['basis']} (confidence {gi['asvspoof2019_la']['confidence']}).")

    st.subheader("Listen and look: human vs AI examples")
    win = lib.feature_table()
    win = win[(win.variant == "orig") & (win.window_idx == 0)][["file_id", "path", "start", "end", "class_name", "source", "generator"]]
    if not (ROOT / "data").exists():
        lib.empty_state("Example audio (needs the local data/ folder and feature table)", "python -m voicedetect.pipeline features")
        return
    cols = st.columns(2, gap="large")
    for col, cls, label in ((cols[0], "human", "Human"), (cols[1], "ai", "AI-generated")):
        with col:
            options = win[win.class_name == cls]
            srcs = sorted(options.source.unique())
            s = st.selectbox(f"{label} source", srcs, format_func=lambda x: lib.SOURCE_LABELS.get(x, x), key=f"ex_src_{cls}")
            opts = options[options.source == s].sort_values("file_id").head(40)
            f = st.selectbox(f"{label} example", opts.file_id.tolist(), key=f"ex_file_{cls}")
            r = opts[opts.file_id == f].iloc[0]
            if not (ROOT / r.path).exists():
                st.warning("This audio file is not present on this machine.")
                continue
            with st.spinner("Loading audio ..."):
                y, sr = lib.window_audio(r.path, r.start, r.end)
            st.audio(lib.wav_bytes(y, sr), format="audio/wav")
            st.caption(f"First analysis window ({len(y) / sr:.1f} s, mono, 22.05 kHz, silence trimmed) · generator: {r.generator}")
            lib.chart(charts.waveform(y, sr, m, color=charts.P(m)["human" if cls == "human" else "ai"], title="Waveform"),
                      f"ex_wave_{cls}")
            lib.chart(charts.mel_spectrogram(y, sr, m, title="Mel-spectrogram"), f"ex_mel_{cls}")


render()
