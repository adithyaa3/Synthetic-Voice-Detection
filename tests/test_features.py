import librosa
import numpy as np
import pytest

from voicedetect import audio_io
from voicedetect.config import AUDIO, FeatureConfig
from voicedetect.features import FeatureError, extract, feature_group, feature_names
from voicedetect.legacy import original_extract_features

from .conftest import speech_like


@pytest.mark.parametrize("fs,n", [("baseline", 80), ("extended", 210)])
def test_fixed_length_and_finite(fs, n):
    for secs in (1.0, 2.5, 4.0):
        v = extract(speech_like(secs, 22050, seed=3), 22050, FeatureConfig(fs))
        assert v.shape == (n,) and v.dtype == np.float64
        assert np.all(np.isfinite(v))
    assert len(feature_names(fs)) == n == len(set(feature_names(fs)))


def test_deterministic():
    w = speech_like(4.0, 22050, seed=9)
    a = extract(w, 22050, FeatureConfig("extended"))
    b = extract(w.copy(), 22050, FeatureConfig("extended"))
    assert np.array_equal(a, b)


def test_baseline_is_prefix_of_extended():
    w = speech_like(4.0, 22050, seed=11)
    assert np.array_equal(extract(w, 22050, FeatureConfig("baseline")), extract(w, 22050, FeatureConfig("extended"))[:80])


def test_baseline_matches_original_feature_code_exactly(audio_dir):
    """The original script's feature function on the same file == baseline on the same samples."""
    p = str(audio_dir / "mono22k.wav")
    y, sr = librosa.load(p, sr=22050, duration=4)
    assert np.array_equal(original_extract_features(p), extract(y, sr, FeatureConfig("baseline")))


@pytest.mark.filterwarnings("ignore:Trying to estimate tuning from empty frequency set")
def test_silent_and_constant_windows_stay_finite():
    """Degenerate windows (rejected earlier by the loader in practice) still give finite features."""
    for w in (np.zeros(22050 * 2, dtype=np.float32), np.full(22050 * 2, 0.1, dtype=np.float32)):
        assert np.all(np.isfinite(extract(w, 22050, FeatureConfig("extended"))))


def test_rejects_too_short_or_non_mono():
    with pytest.raises(FeatureError):
        extract(np.zeros(100, dtype=np.float32), 22050)
    with pytest.raises(FeatureError):
        extract(np.zeros((2, 22050), dtype=np.float32), 22050)


def test_every_extended_feature_has_a_group():
    groups = {feature_group(n) for n in feature_names("extended")}
    assert groups == {"mfcc_mean", "mfcc_std", "delta_mfcc", "delta2_mfcc", "spectral_shape", "zcr_rms",
                      "spectral_contrast", "chroma"}


def test_same_file_same_features_across_formats_is_close(audio_dir):
    """Lossless formats of the same signal give (nearly) the same features after resampling."""
    a = audio_io.load(str(audio_dir / "mono44k.wav"))
    b = audio_io.load(str(audio_dir / "stereo44k_identical.wav"))
    fa = extract(a.samples[: int(4 * AUDIO.sample_rate)], AUDIO.sample_rate)
    fb = extract(b.samples[: int(4 * AUDIO.sample_rate)], AUDIO.sample_rate)
    np.testing.assert_allclose(fa, fb, rtol=1e-4, atol=1e-3)
