import json

import pandas as pd
import streamlit as st

import lib
from voicedetect.config import ASVSPOOF_DIR


@lib.safe_page
def render():
    lib.page_header("About & limitations", "Methods, data sources and licences, ethics, and what this detector "
                                           "cannot do.")
    meta = lib.model_metadata()
    prov_path = ASVSPOOF_DIR / "PROVENANCE.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.exists() else None

    st.subheader("Methods in brief")
    st.markdown(
        "1. **Audio handling:** decode (libsndfile; MP3 without ffmpeg) -> mono -> resample to 22.05 kHz -> trim leading/"
        "trailing silence -> reject silent, undecodable, too-short (< 1 s of sound) or too-long input.\n"
        "2. **Windows:** consecutive 4 s windows; low-activity windows skipped; at most 10 per file; a file's prediction "
        "is the mean over its windows.\n"
        "3. **Features:** baseline = the original 40 MFCC means + 40 standard deviations; extended adds delta / "
        "delta-delta MFCC spread, spectral centroid / bandwidth / roll-off / flatness / contrast, zero-crossing rate, RMS "
        "and chroma (210 total). One feature function is shared by training, evaluation and prediction.\n"
        "4. **Validation:** group-aware nested 5-fold CV on the development partition; an untouched test partition "
        "chosen by group before modelling; leakage assertions on every split.\n"
        "5. **Models:** majority class, logistic regression, linear SVM, RBF SVM (the original), random forest and "
        "histogram gradient boosting, all as scikit-learn pipelines (scaling fit on training folds only).\n"
        "6. **Selection, calibration, threshold:** pre-declared rule on validation AUC with a simplicity tie-break; "
        "calibration method by validation log loss; threshold minimising validation cost with false alarms weighted 2x.")

    st.subheader("Data sources and licences")
    rows = [
        {"Data": "Human speech: Mozilla Common Voice (English) clips, supplied by the user", "Origin": "user",
         "Licence / terms": "Mozilla states Common Voice exists to create public-domain datasets (common-voice/common-voice "
                            "README); the exact release and download terms accepted by the user are not recorded here.",
         "Used as": "human class"},
        {"Data": "AI speech: 56 voice-conversion files (8 source recordings x 7 target voices), supplied by the user",
         "Origin": "user", "Licence / terms": "Unknown (tool and source recordings not documented). They imitate public "
                                             "figures' voices: use for research/education only and do not redistribute.",
         "Used as": "AI class"},
        {"Data": "ASVspoof 2019 Logical Access subset (bona fide VCTK speech + 19 TTS/VC systems)", "Origin": "public",
         "Licence / terms": "Open Data Commons Attribution License (ODC-By) v1.0, verified from the dataset's README and "
                            "licence files. Citation: Todisco et al., Interspeech 2019; doi:10.7488/ds/2555.",
         "Used as": "supplementary human + AI"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if prov:
        st.caption(f"ASVspoof subset: {prov['n_files']} files selected with seed {prov['selection_seed']}, retrieved "
                   f"{prov['retrieved_utc']} via HTTP range requests (CRC-checked).")

    st.subheader("Known limitations")
    st.markdown(
        "- **Small, narrow AI data from the user:** 8 independent source recordings and one unknown conversion tool. "
        "Windows from the same recording are correlated; reported uncertainty (fold std, group-bootstrap CIs) reflects "
        "this, but estimates remain coarse.\n"
        "- **Confounded recording conditions:** user human clips are crowd-sourced 48 kbps MP3s, user AI clips are "
        "uncompressed conversions of broadcast-style recordings. The codec-matched control removes the file-format "
        "difference but not every channel difference.\n"
        "- **Generators covered:** one modern voice-conversion tool plus 2019-era ASVspoof systems. Newer TTS/VC systems "
        "are not represented and may evade detection entirely.\n"
        "- **Domain shift:** accents, languages, recording devices and rooms that differ from the training data reduce "
        "reliability. Accent/recording bias was not measured.\n"
        "- **Microphone input** is processed by the browser and differs from all training audio.\n"
        "- **Detectors can be fooled:** re-recording, added noise, compression or deliberate adversarial edits can change "
        "the output.\n"
        "- **Human groups are inferred** (Common Voice speaker IDs unavailable), so some speaker overlap between "
        "development and test cannot be ruled out.")

    st.subheader("Ethics")
    st.markdown(
        "- The output is a **probability, not proof**. Do not use it as the sole basis for accusing anyone of fabricating "
        "audio, or for any decision about a person.\n"
        "- A false alarm (a real person's voice labelled synthetic) was treated as **twice as costly** as a miss when "
        "choosing the threshold, and borderline scores are shown as **inconclusive**.\n"
        "- Uploaded and recorded audio is processed in memory and **not stored** unless you switch storage on.\n"
        "- The user's AI data imitates real public figures; keep it private and use it only for detection research.")

    st.subheader("Reproducibility")
    st.code("python -m venv .venv && .venv\\Scripts\\activate\npip install -r requirements.txt && pip install -e .\n"
            "python -m voicedetect.pipeline all     # or: python scripts/train.py\n"
            "python -m pytest\nstreamlit run app/streamlit_app.py", language="bash")
    lv = meta.get("library_versions", {})
    st.caption(f"Model {meta['model_version']} · seed {meta['seed']} · data manifest {meta['data_manifest_hash']} · "
               + " · ".join(f"{k} {v}" for k, v in lv.items()))


render()
