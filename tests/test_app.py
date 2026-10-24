"""Headless app tests with streamlit.testing.AppTest (needs the trained model + reports)."""
import pytest

from voicedetect.config import ROOT

from .conftest import requires_data, requires_model, speech_like, wav_bytes

APP = str(ROOT / "app" / "streamlit_app.py")
PAGES = ["views/overview.py", "views/dataset.py", "views/features.py", "views/models.py", "views/evaluation.py",
         "views/robustness.py", "views/errors.py", "views/live.py", "views/about.py"]

pytestmark = [requires_model, requires_data]


def _app(page: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    if page != PAGES[0]:
        at.switch_page(page).run()
    return at


def _assert_clean(at):
    assert not at.exception, [e.value for e in at.exception]
    errors = [e.value for e in at.error]
    assert not any("Something went wrong" in e for e in errors), errors
    assert not any("is not available yet" in i.value for i in at.info)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _app(page)
    _assert_clean(at)
    assert at.title, "page has a title"


def test_upload_valid_and_invalid_audio():
    at = _app("views/live.py")
    at.file_uploader(key="vd_upload").upload("speech.wav", wav_bytes(speech_like(6, 44100), 44100), "audio/wav").run()
    _assert_clean(at)
    html = " ".join(m.value for m in at.markdown)
    assert "class='vd-verdict'" in html and "Calibrated probability of AI-generated speech" in html
    assert any(m.label == "P(AI-generated)" for m in at.metric)
    at.file_uploader(key="vd_upload").upload("fake.wav", b"not audio at all" * 100, "audio/wav").run()
    assert any("couldn't be decoded" in e.value for e in at.error)
    at.file_uploader(key="vd_upload").upload("quiet.wav", wav_bytes(speech_like(6, 44100) * 0, 44100), "audio/wav").run()
    assert any("silent" in e.value for e in at.error)


def test_microphone_recording_flow():
    at = _app("views/live.py")
    verdict_text = "Calibrated probability of AI-generated speech"  # (the CSS also mentions the card's class name)
    at.audio_input(key="vd_mic").upload("recording.wav", wav_bytes(speech_like(6, 44100), 44100)).run()
    assert not any(verdict_text in m.value for m in at.markdown), "must not classify before the button is pressed"
    at.button(key="vd_classify").click().run()
    _assert_clean(at)
    html = " ".join(m.value for m in at.markdown)
    assert verdict_text in html and "class='vd-verdict'" in html
    assert any("Microphone audio differs" in w.value for w in at.warning)


def test_corrupt_microphone_recording_gives_friendly_message():
    at = _app("views/live.py")
    at.audio_input(key="vd_mic").upload("recording.wav", b"RIFF-but-not-really-audio" * 50).run()
    at.button(key="vd_classify").click().run()
    assert not at.exception
    assert any("couldn't be decoded" in e.value for e in at.error)


def test_threshold_slider_recomputes_metrics():
    at = _app("views/evaluation.py")
    before = {m.label: m.value for m in at.metric}
    at.slider(key="ev_thr").set_value(0.9).run()
    after = {m.label: m.value for m in at.metric}
    assert before["AI recall"] != after["AI recall"]


def test_storage_is_off_by_default_and_nothing_is_saved(tmp_path):
    uploads = ROOT / "uploads"
    before = set(uploads.glob("*")) if uploads.exists() else set()
    at = _app("views/live.py")
    assert at.toggle(key="vd_store").value is False
    at.file_uploader(key="vd_upload").upload("s.wav", wav_bytes(speech_like(5, 22050, seed=3), 22050), "audio/wav").run()
    after = set(uploads.glob("*")) if uploads.exists() else set()
    assert after == before
