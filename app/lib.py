"""Shared helpers for the Streamlit app: cached artifact loading, theming, safe rendering."""
from __future__ import annotations

import functools
import io
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

from voicedetect.config import FINAL_MODEL_PATH, MODEL_LABELS, REPORTS_DIR, ROOT

log = logging.getLogger("voicedetect.app")
ASSETS = Path(__file__).parent / "assets"
LABELS = {**MODEL_LABELS, "rbf_svm": "RBF SVM (tuned)", "rbf_svm_original": "RBF SVM (original settings)"}
SOURCE_LABELS = {"user_commonvoice": "User: Common Voice (human)", "user_voice_conversion": "User: voice conversion (AI)",
                 "asvspoof2019_la": "Supplementary: ASVspoof 2019 LA"}
PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def mode() -> str:
    """Chart palette. Always the validated dark palette: st.context.theme is documented as unreliable on
    the first load of a session, which rendered light charts on the dark default theme. If a viewer
    switches Streamlit to light mode, the charts stay dark 'cards' (still fully legible)."""
    return "dark"


def inject_css() -> None:
    st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


class MissingArtifact(FileNotFoundError):
    pass


def _path(name: str) -> Path:
    p = REPORTS_DIR / name
    if not p.exists():
        raise MissingArtifact(name)
    return p


@st.cache_data(show_spinner=False)
def _json(name: str, mtime: float) -> dict:
    return json.loads(_path(name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _table(name: str, mtime: float) -> pd.DataFrame:
    p = _path(name)
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def j(name: str) -> dict:
    return _json(name, _path(name).stat().st_mtime)


def t(name: str) -> pd.DataFrame:
    return _table(name, _path(name).stat().st_mtime)


def model_metadata() -> dict:
    p = ROOT / "models" / "metadata.json"
    if not p.exists():
        raise MissingArtifact("models/metadata.json")
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading model ...")
def bundle(mtime: float):
    from voicedetect.predict import load_bundle
    return load_bundle(FINAL_MODEL_PATH)


def get_bundle():
    if not FINAL_MODEL_PATH.exists():
        raise MissingArtifact("models/final_model.joblib")
    return bundle(FINAL_MODEL_PATH.stat().st_mtime)


@st.cache_data(show_spinner="Loading feature table ...")
def _feature_table(mtime: float) -> pd.DataFrame:
    from voicedetect.dataset import load_table
    return load_table()


def feature_table() -> pd.DataFrame:
    from voicedetect.config import FEATURES_DIR
    from voicedetect.dataset import cache_key
    p = FEATURES_DIR / f"windows_{cache_key()}.parquet"
    if not p.exists():
        raise MissingArtifact(f"features/{p.name}")
    return _feature_table(p.stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=64)
def window_audio(path: str, start: int, end: int) -> tuple[np.ndarray, int]:
    """Re-create an analysis window exactly as the pipeline saw it (mono, 22.05 kHz, trimmed)."""
    from voicedetect.dataset import window_audio as _wa
    from voicedetect.config import AUDIO
    return _wa(path, int(start), int(end)), AUDIO.sample_rate


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1, 1), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def chart(fig, key: str, data: pd.DataFrame | None = None, caption: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG, key=key)
    if caption:
        st.caption(caption)
    if data is not None:
        with st.expander("Show data as a table"):
            st.dataframe(data, hide_index=True, width="stretch")


def empty_state(what: str, how: str = "python -m voicedetect.pipeline all") -> None:
    st.info(f"**{what} is not available yet.** Generate it with `{how}` and reload this page.", icon=":material/info:")


def safe_page(fn):
    """Render a page; missing artifacts show an empty state, any other error a friendly message
    (details go to the server log only)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except MissingArtifact as e:
            empty_state(f"The artifact `{e.args[0]}`")
        except Exception:  # noqa: BLE001 - last-resort guard so users never see a traceback
            log.error("page error\n%s", traceback.format_exc())
            st.error("Something went wrong while rendering this page. The details were written to the server log.",
                     icon=":material/error:")
    return wrapper


def fmt_pm(m: float, s: float, d: int = 3) -> str:
    return f"{m:.{d}f} ± {s:.{d}f}"


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='vd-sub'>{subtitle}</p>", unsafe_allow_html=True)
