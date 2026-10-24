import json

import numpy as np
import pytest

from voicedetect.audio_io import AudioError
from voicedetect.predict import (ModelConfigMismatch, ModelNotFound, load_bundle, predict_array, predict_bytes,
                                 predict_file)

from .conftest import speech_like, wav_bytes

KEYS = {"label", "probability", "threshold", "inconclusive", "model_version", "warnings"}


def test_predict_file_contract(audio_dir, tiny_bundle):
    bundle, _ = tiny_bundle
    r = predict_file(audio_dir / "mono44k.wav", bundle)
    assert KEYS <= set(r)
    assert 0.0 <= r["probability"] <= 1.0
    assert r["label"] in {"human", "ai", "inconclusive"}
    assert r["threshold"] == 0.5 and r["model_version"] == bundle.version
    assert r["inconclusive"] == (abs(r["probability"] - 0.5) < 0.1)
    assert r["n_windows"] == 1 and len(r["window_probabilities"]) == 1


def test_long_file_averages_windows(audio_dir, tiny_bundle):
    bundle, _ = tiny_bundle
    r = predict_file(audio_dir / "long12s.wav", bundle)
    assert r["n_windows"] == 3
    assert r["probability"] == pytest.approx(np.mean(r["window_probabilities"]))
    assert any("windows" in w for w in r["warnings"])


@pytest.mark.parametrize("name,code", [("nope.wav", "not_found"), ("notes.txt", "unsupported_format"),
                                       ("empty.wav", "empty"), ("corrupt.wav", "undecodable"),
                                       ("silent.wav", "silent"), ("too_short.wav", "too_short")])
def test_predict_file_validation(audio_dir, tiny_bundle, name, code):
    with pytest.raises(AudioError) as e:
        predict_file(audio_dir / name, tiny_bundle[0])
    assert e.value.code == code


def test_predict_bytes_validation(audio_dir, tiny_bundle):
    bundle, _ = tiny_bundle
    with pytest.raises(AudioError) as e:
        predict_bytes(b"hello", "x.exe", bundle)
    assert e.value.code == "unsupported_format"
    with pytest.raises(AudioError) as e:
        predict_bytes((audio_dir / "mono44k.wav").read_bytes(), "a.wav", bundle, max_mb=0.01)
    assert e.value.code == "too_large"
    with pytest.raises(AudioError) as e:
        predict_bytes(b"not really a wav file at all", "fake.wav", bundle)
    assert e.value.code == "undecodable"


def test_predict_bytes_does_not_write_files(audio_dir, tiny_bundle, tmp_path, monkeypatch):
    """Uploads are decoded in memory: the working directory and temp dir stay unchanged."""
    import tempfile
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    before = set(tmp_path.rglob("*"))
    predict_bytes(wav_bytes(speech_like(5, 44100), 44100), "rec.wav", tiny_bundle[0])
    assert set(tmp_path.rglob("*")) == before


def test_predict_array_microphone_like_input(tiny_bundle):
    bundle, _ = tiny_bundle
    y = (speech_like(5, 48000) * 32767).astype(np.int16)
    r = predict_array(np.c_[y, y], 48000, bundle)  # stereo int16 at 48 kHz, as browsers often deliver
    assert KEYS <= set(r) and r["input_sample_rate"] == 48000 and r["input_channels"] == 2
    with pytest.raises(AudioError) as e:
        predict_array(np.zeros(48000 * 3, dtype=np.int16), 48000, bundle)
    assert e.value.code == "silent"


def test_model_round_trip_and_mismatch_refusal(tiny_bundle, tmp_path, audio_dir):
    import joblib
    bundle, path = tiny_bundle
    again = load_bundle(path)
    a = predict_file(audio_dir / "clip.flac", bundle)["probability"]
    b = predict_file(audio_dir / "clip.flac", again)["probability"]
    assert a == b
    obj = joblib.load(path)
    obj["metadata"]["feature_signature_hash"] = "deadbeefdeadbeef"
    bad = tmp_path / "bad.joblib"
    joblib.dump(obj, bad)
    with pytest.raises(ModelConfigMismatch):
        load_bundle(bad)
    with pytest.raises(ModelNotFound):
        load_bundle(tmp_path / "missing.joblib")
    meta = json.loads(path.with_suffix(".json").read_text())
    assert {"feature_signature", "library_versions", "threshold", "model_version"} <= set(meta)
