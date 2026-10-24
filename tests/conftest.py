"""Deterministic synthetic audio fixtures, so the suite never depends on the private dataset."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voicedetect.config import FINAL_MODEL_PATH, MANIFEST_PATH


def speech_like(seconds: float, sr: int, f0: float = 140.0, seed: int = 0) -> np.ndarray:
    """Harmonic signal with a wandering pitch, 4 Hz syllable envelope and a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    pitch = f0 * (1 + 0.08 * np.sin(2 * np.pi * 0.7 * t) + 0.01 * rng.standard_normal(len(t)).cumsum() / np.sqrt(len(t)))
    phase = 2 * np.pi * np.cumsum(pitch) / sr
    y = sum((0.6 / k) * np.sin(k * phase) for k in range(1, 12))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t) ** 2
    y = y * env + 0.01 * rng.standard_normal(len(t))
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


@pytest.fixture(scope="session")
def audio_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("audio")
    y22 = speech_like(6.0, 22050, seed=1)
    sf.write(d / "mono22k.wav", y22, 22050)
    y44 = speech_like(6.0, 44100, seed=1)
    sf.write(d / "mono44k.wav", y44, 44100)
    sf.write(d / "stereo44k_identical.wav", np.c_[y44, y44], 44100)
    sf.write(d / "stereo44k_different.wav", np.c_[y44, speech_like(6.0, 44100, f0=200, seed=2)], 44100)
    sf.write(d / "clip.flac", speech_like(5.0, 16000, seed=3), 16000)
    sf.write(d / "clip.mp3", speech_like(5.0, 32000, seed=4), 32000, format="MP3")
    sf.write(d / "long12s.wav", speech_like(12.5, 22050, seed=5), 22050)
    sf.write(d / "short3s.wav", speech_like(3.0, 22050, seed=6), 22050)
    sf.write(d / "too_short.wav", speech_like(0.5, 22050, seed=7), 22050)
    sf.write(d / "silent.wav", np.zeros(22050 * 3, dtype=np.float32), 22050)
    sf.write(d / "near_silent.wav", (1e-5 * np.random.default_rng(0).standard_normal(22050 * 3)).astype(np.float32), 22050)
    (d / "corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk" + bytes(range(256)) * 20)
    (d / "notes.txt").write_text("not audio")
    (d / "empty.wav").write_bytes(b"")
    return d


@pytest.fixture(scope="session")
def tiny_bundle(tmp_path_factory, audio_dir):
    """A small but real model bundle trained on synthetic features, saved and re-loaded via the API."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from voicedetect import audio_io
    from voicedetect.config import AUDIO, FeatureConfig
    from voicedetect.features import extract
    from voicedetect.predict import load_bundle, save_bundle

    X, y = [], []
    for i in range(16):
        for label, f0 in ((0, 110 + 5 * i), (1, 230 + 5 * i)):
            w = speech_like(4.0, AUDIO.sample_rate, f0=f0, seed=100 + i)
            X.append(extract(w, AUDIO.sample_rate, FeatureConfig("baseline")))
            y.append(label)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.array(X), np.array(y))
    d = tmp_path_factory.mktemp("model")
    save_bundle(pipe, "baseline", {"threshold": 0.5, "inconclusive_halfwidth": 0.1},
                {"training_setting": "synthetic-test", "model_name": "logreg", "best_params": {}},
                model_path=d / "m.joblib", metadata_path=d / "m.json")
    return load_bundle(d / "m.joblib"), d / "m.joblib"


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


requires_data = pytest.mark.skipif(not MANIFEST_PATH.exists(), reason="real dataset/manifest not present")
requires_model = pytest.mark.skipif(not FINAL_MODEL_PATH.exists(), reason="trained model not present")
