import hashlib
import io
from datetime import datetime
from pathlib import Path

import streamlit as st

import lib
from voicedetect import audio_io, charts
from voicedetect.audio_io import AudioError
from voicedetect.config import AUDIO, PREDICT, ROOT
from voicedetect.predict import predict_loaded

UPLOAD_DIR = ROOT / "uploads"
FRIENDLY = {
    "unsupported_format": "This file type isn't supported. Please use WAV, MP3, FLAC or OGG audio.",
    "too_large": f"This file is larger than {PREDICT.ui_max_upload_mb} MB. Export a shorter excerpt and try again.",
    "empty": "The file is empty.",
    "undecodable": "This audio couldn't be decoded. The file may be corrupt, or it isn't really audio "
                   "(for example a renamed document).",
    "too_short": f"The recording is too short: at least {AUDIO.min_duration_s:.0f} s of sound is needed after silence "
                 f"is removed (about {AUDIO.window_seconds:.0f} s or more gives the most reliable result).",
    "too_long": f"The audio is longer than {PREDICT.max_duration_s / 60:.0f} minutes. Please upload a shorter excerpt.",
    "silent": "The audio is silent or almost silent. Check that the right file or microphone was used and that the "
              "input level isn't muted.",
    "invalid_samples": "The audio contains invalid sample values and can't be analysed.",
}
VERDICT = {"human": ("Human", "human", "Most consistent with a real human voice."),
           "ai": ("AI-generated", "ai", "Most consistent with synthetic (AI-generated or converted) speech."),
           "inconclusive": ("Inconclusive", "neutral", "Too close to the decision threshold to call either way.")}


def _store(data: bytes, name: str) -> Path:
    UPLOAD_DIR.mkdir(exist_ok=True)
    out = UPLOAD_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{Path(name).name}"
    out.write_bytes(data)
    return out


def _verdict_card(r: dict, m: str) -> None:
    p = charts.P(m)
    label, colour_key, explain = VERDICT[r["label"]]
    stripe = p[colour_key]
    style = (f"--vd-stripe:{stripe};--vd-surface:{p['surface']};--vd-border:{p['grid']};--vd-ink:{p['ink']};"
             f"--vd-ink2:{p['ink2']};--vd-muted:{p['muted']}")
    side = "AI-generated" if r["raw_label"] == "ai" else "human"
    extra = f" (leaning {side})" if r["inconclusive"] else ""
    st.markdown(
        f"<div class='vd-verdict' style='{style}' role='status' aria-live='polite'>"
        f"<div class='vd-kicker'>Verdict</div>"
        f"<div class='vd-label'><span class='vd-dot' aria-hidden='true'></span>{label}{extra}</div>"
        f"<div class='vd-detail'>{explain} Calibrated probability of AI-generated speech: "
        f"<b>{r['probability']:.1%}</b> · threshold {r['threshold']:.2f} · inconclusive band "
        f"{r['inconclusive_band'][0]:.2f}-{r['inconclusive_band'][1]:.2f}</div></div>",
        unsafe_allow_html=True)


def _show_result(r: dict, la: audio_io.LoadedAudio, key: str, mic: bool = False) -> None:
    m = lib.mode()
    _verdict_card(r, m)
    lib.chart(charts.probability_meter(r["probability"], r["threshold"], tuple(r["inconclusive_band"]), m), f"meter_{key}")
    c = st.columns(4)
    c[0].metric("P(AI-generated)", f"{r['probability']:.3f}", border=True)
    c[1].metric("Threshold", f"{r['threshold']:.2f}", border=True)
    c[2].metric("Windows analysed", r["n_windows"], border=True)
    c[3].metric("Sound analysed", f"{r['analysed_duration_s']:.1f} s", border=True,
                help=f"Input: {r['duration_s']:.1f} s, {r['input_sample_rate']} Hz, {r['input_channels']} channel(s)")
    st.markdown("<div class='vd-note'>This is a model probability, not proof. Detectors make mistakes, especially on "
                "voices, microphones and generators unlike their training data. Do not use it as the sole evidence "
                "against anyone.</div>", unsafe_allow_html=True)
    notes = list(r["warnings"])
    if mic:
        notes.insert(0, "Microphone audio differs from the training data (room acoustics, background noise, browser "
                        "processing and compression), so this result is less reliable than for studio-quality files.")
    for w in notes:
        st.warning(w, icon=":material/warning:")
    if r["n_windows"] > 1:
        lib.chart(charts.window_probs(r["window_probabilities"], r["window_spans_s"], r["threshold"], m), f"wp_{key}")
    y = la.samples[: int(60 * la.sample_rate)]  # plot at most the first minute
    c1, c2 = st.columns(2, gap="large")
    with c1:
        lib.chart(charts.waveform(y, la.sample_rate, m, title="Waveform (silence trimmed, mono, 22.05 kHz)"), f"wave_{key}")
    with c2:
        lib.chart(charts.mel_spectrogram(y, la.sample_rate, m, title="Mel-spectrogram"), f"mel_{key}")
    st.caption(f"Model version {r['model_version']}.")


def _run(kind: str, data: bytes, name: str, bundle) -> None:
    """Classify bytes (upload) or a WAV recording (microphone) and render; results are cached per content."""
    digest = hashlib.sha1(data).hexdigest()
    cache = st.session_state.setdefault("vd_results", {})
    try:
        if digest not in cache:
            with st.spinner("Analysing audio ..."):
                if kind == "upload":
                    if Path(name).suffix.lower() not in PREDICT.allowed_extensions:
                        raise AudioError("unsupported_format")
                    if len(data) / 1e6 > PREDICT.ui_max_upload_mb:
                        raise AudioError("too_large")
                    la = audio_io.load_bytes(data, AUDIO, PREDICT.max_duration_s)
                else:  # microphone: decode the browser's WAV (coded errors) and use the in-memory-array path
                    y, sr, _, _ = audio_io.decode(io.BytesIO(data), PREDICT.max_duration_s)
                    la = audio_io.load_array(y, sr, AUDIO)
                cache[digest] = (predict_loaded(la, bundle), la)
        r, la = cache[digest]
    except AudioError as e:
        st.error(FRIENDLY.get(e.code, str(e)), icon=":material/error:")
        return
    if st.session_state.get("vd_store"):
        saved = st.session_state.setdefault("vd_saved", {})
        if digest not in saved:  # save each distinct recording once, not on every rerun
            saved[digest] = _store(data, name).relative_to(ROOT).as_posix()
        st.caption(f"Saved a copy to `{saved[digest]}` (storage is switched on in Privacy settings).")
    _show_result(r, la, f"{kind}_{digest[:8]}", mic=(kind == "mic"))


@lib.safe_page
def render():
    lib.page_header("Live prediction", "Upload a recording or use your microphone. The audio goes through exactly the "
                                       "same loading, windowing and features as the training data.")
    bundle = lib.get_bundle()
    with st.expander("Privacy settings: audio is not stored by default", icon=":material/lock:"):
        st.toggle("Keep a copy of uploaded/recorded audio on this computer", value=False, key="vd_store",
                  help=f"Off by default. When on, audio is saved to {UPLOAD_DIR.relative_to(ROOT).as_posix()}/ in the project folder.")
        st.caption("With storage off, audio is processed in memory only and discarded; results are kept only in this "
                   "browser session.")
    up_tab, mic_tab = st.tabs(["Upload a file", "Record with microphone"], key="live_tabs")
    with up_tab:
        # the widget accepts up to 200 MB so that the app (not just a red chip) explains a >50 MB rejection
        f = st.file_uploader(f"Audio file (WAV, MP3, FLAC or OGG, up to {PREDICT.ui_max_upload_mb} MB)",
                             type=["wav", "mp3", "flac", "ogg"], key="vd_upload", max_upload_size=200,
                             help="Decoded in memory; nothing is written to disk unless you switch storage on in "
                                  "Privacy settings above.")
        if f is not None:
            _run("upload", f.getvalue(), f.name, bundle)
        else:
            st.caption("Tip: 4 s or more of clear speech gives the most reliable result. Long files are analysed in up "
                       "to 10 windows and averaged.")
    with mic_tab:
        mic_dev = lib.j("selection/calibration.json").get("dev_robustness_at_threshold", {}).get("mic")
        if mic_dev and mic_dev["specificity"] < 0.8:
            st.warning(f"**Expect false alarms on microphone audio.** In development tests with a simulated laptop "
                       f"microphone, only {mic_dev['specificity']:.0%} of real human recordings were cleared, so a real "
                       "voice recorded here may well be labelled AI-generated. Treat this tab as a demonstration of "
                       "domain shift, not as a reliable check. Speak for at least 4-5 seconds.", icon=":material/mic:")
        else:
            st.info("Microphone audio differs from the training data (room noise, device and browser processing), so live "
                    "results are less reliable. Speak for at least 4-5 seconds.", icon=":material/mic:")
        rec = st.audio_input("Record a voice sample", sample_rate=44100, key="vd_mic",
                             help="Click the microphone to start and the stop button to finish. Use play to listen back; "
                                  "record again, or use the clear (trash) button above the recorder, to start over.")
        if rec is not None:
            if st.button("Classify recording", type="primary", icon=":material/graphic_eq:", key="vd_classify"):
                st.session_state["vd_mic_digest"] = hashlib.sha1(rec.getvalue()).hexdigest()
            if st.session_state.get("vd_mic_digest") == hashlib.sha1(rec.getvalue()).hexdigest():
                _run("mic", rec.getvalue(), "recording.wav", bundle)
            else:
                st.caption("Recording ready. Listen back with the play button, re-record if needed, then press "
                           "**Classify recording**.")
        with st.expander("Microphone not working?"):
            st.markdown(
                "- The browser must be allowed to use the microphone. If you clicked *Block*, use the padlock / site "
                "settings icon next to the address bar to allow it, then reload.\n"
                "- Microphone access only works in a **secure context**: `http://localhost` or `https://`. Opening the app "
                "via a LAN IP over plain http will not offer the microphone.\n"
                "- On Windows also check *Settings > Privacy & security > Microphone* (allow desktop apps / your browser).\n"
                "- If no microphone is connected, the recorder cannot start; use the *Upload a file* tab instead.")


render()
