"""Synthetic Voice Detection - Streamlit app.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))  # for `import lib`
try:
    import voicedetect  # noqa: F401
except ImportError:  # running without `pip install -e .`
    sys.path.insert(0, str(APP_DIR.parent / "src"))

import lib  # noqa: E402

# local icon file: a Material-icon favicon would be fetched from fonts.gstatic.com at runtime
st.set_page_config(page_title="Synthetic Voice Detection", page_icon=str(APP_DIR / "assets" / "icon.png"),
                   layout="wide", initial_sidebar_state="auto")  # "auto": collapsed on phones, open on desktop
lib.inject_css()

V = APP_DIR / "views"
pages = {
    "Project": [
        st.Page(V / "overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page(V / "dataset.py", title="Dataset explorer", icon=":material/library_music:"),
        st.Page(V / "features.py", title="Feature insights", icon=":material/insights:"),
    ],
    "Results": [
        st.Page(V / "models.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page(V / "evaluation.py", title="Evaluation", icon=":material/fact_check:"),
        st.Page(V / "robustness.py", title="Robustness & generalisation", icon=":material/shield:"),
        st.Page(V / "errors.py", title="Error analysis", icon=":material/report:"),
    ],
    "Try it": [
        st.Page(V / "live.py", title="Live prediction", icon=":material/mic:"),
        st.Page(V / "about.py", title="About & limitations", icon=":material/info:"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("All numbers on these pages are loaded from saved pipeline artifacts (reports/, models/).")
nav.run()
