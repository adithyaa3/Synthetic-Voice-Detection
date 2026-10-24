"""The training path and every prediction path must produce identical features / probabilities."""
import numpy as np
import pytest
import soundfile as sf

from voicedetect import audio_io, dataset
from voicedetect.config import AUDIO, FeatureConfig
from voicedetect.features import extract_many
from voicedetect.predict import predict_array, predict_bytes, predict_file


@pytest.mark.parametrize("name", ["mono44k.wav", "clip.mp3", "long12s.wav", "short3s.wav"])
def test_training_path_equals_prediction_path(audio_dir, tmp_path, name, tiny_bundle):
    bundle, _ = tiny_bundle
    path = audio_dir / name
    train = dataset._process_file(str(path), f"test-{name}", tmp_path, AUDIO)  # training feature builder
    la = audio_io.load(str(path))
    spans = audio_io.select_windows(la.samples)
    pred_feats = extract_many([la.samples[s:e] for s, e in spans], AUDIO.sample_rate, FeatureConfig("baseline"))
    assert train["status"] == "ok"
    assert np.array_equal(train["starts"], [s for s, _ in spans])
    assert np.array_equal(train["X_orig"][:, :80], pred_feats)
    # and the probability returned by the public API equals the model applied to training-path features
    p_train = float(np.mean(bundle.pipeline.predict_proba(train["X_orig"][:, :80])[:, 1]))
    assert predict_file(path, bundle)["probability"] == pytest.approx(p_train, abs=1e-12)


def test_cached_features_are_reused_identically(audio_dir, tmp_path):
    a = dataset._process_file(str(audio_dir / "mono22k.wav"), "same-sha", tmp_path, AUDIO)
    b = dataset._process_file(str(audio_dir / "mono22k.wav"), "same-sha", tmp_path, AUDIO)
    assert a["cached"] is False and b["cached"] is True
    assert np.array_equal(a["X_orig"], b["X_orig"]) and np.array_equal(a["X_mp3cv"], b["X_mp3cv"])


def test_cache_key_changes_with_config():
    from voicedetect.config import AudioConfig
    assert dataset.cache_key(AUDIO) != dataset.cache_key(AudioConfig(window_seconds=3.0))
    assert dataset.cache_key(AUDIO) == dataset.cache_key(AudioConfig())


@pytest.mark.parametrize("name", ["mono44k.wav", "stereo44k_different.wav", "clip.flac", "clip.mp3"])
def test_predict_file_bytes_array_agree(audio_dir, name, tiny_bundle):
    bundle, _ = tiny_bundle
    path = audio_dir / name
    a = predict_file(path, bundle)
    b = predict_bytes(path.read_bytes(), name, bundle)
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    c = predict_array(y, sr, bundle)
    assert a["probability"] == pytest.approx(b["probability"], abs=1e-12)
    assert a["probability"] == pytest.approx(c["probability"], abs=1e-12)
    assert a["label"] == b["label"] == c["label"]
    assert a["n_windows"] == b["n_windows"] == c["n_windows"]
