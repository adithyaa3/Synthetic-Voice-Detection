import numpy as np
import pytest

from voicedetect import audio_io
from voicedetect.audio_io import AudioError
from voicedetect.config import AUDIO, AudioConfig


def test_mono_is_resampled_to_analysis_rate(audio_dir):
    la = audio_io.load(str(audio_dir / "mono44k.wav"))
    assert la.sample_rate == AUDIO.sample_rate == 22050
    assert la.original_sample_rate == 44100 and la.original_channels == 1
    assert la.samples.dtype == np.float32 and la.samples.ndim == 1


def test_stereo_with_identical_channels_equals_mono(audio_dir):
    mono = audio_io.load(str(audio_dir / "mono44k.wav"))
    stereo = audio_io.load(str(audio_dir / "stereo44k_identical.wav"))
    assert stereo.original_channels == 2
    assert any("mixed down to mono" in w for w in stereo.warnings)
    np.testing.assert_allclose(stereo.samples, mono.samples, atol=1e-6)


def test_stereo_with_different_channels_is_averaged(audio_dir):
    la = audio_io.load(str(audio_dir / "stereo44k_different.wav"))
    assert la.samples.ndim == 1 and la.original_channels == 2


@pytest.mark.parametrize("name", ["clip.flac", "clip.mp3", "mono22k.wav"])
def test_supported_formats_decode(audio_dir, name):
    la = audio_io.load(str(audio_dir / name))
    assert la.trimmed_duration_s > 3


def test_mp3_decodes_in_memory(audio_dir):
    la = audio_io.load_bytes((audio_dir / "clip.mp3").read_bytes())
    assert la.original_sample_rate == 32000


@pytest.mark.parametrize("name,code", [
    ("too_short.wav", "too_short"),
    ("silent.wav", "silent"),
    ("near_silent.wav", "silent"),
    ("corrupt.wav", "undecodable"),
])
def test_bad_audio_raises_coded_error(audio_dir, name, code):
    with pytest.raises(AudioError) as e:
        audio_io.load(str(audio_dir / name))
    assert e.value.code == code


def test_check_path_rejects_missing_unsupported_empty_and_large(audio_dir):
    with pytest.raises(AudioError) as e:
        audio_io.check_path(audio_dir / "nope.wav")
    assert e.value.code == "not_found"
    with pytest.raises(AudioError) as e:
        audio_io.check_path(audio_dir / "notes.txt")
    assert e.value.code == "unsupported_format"
    with pytest.raises(AudioError) as e:
        audio_io.check_path(audio_dir / "empty.wav")
    assert e.value.code == "empty"
    with pytest.raises(AudioError) as e:
        audio_io.check_path(audio_dir / "long12s.wav", max_file_mb=0.1)
    assert e.value.code == "too_large"


def test_too_long_is_rejected(audio_dir):
    with pytest.raises(AudioError) as e:
        audio_io.load(str(audio_dir / "long12s.wav"), max_duration_s=5)
    assert e.value.code == "too_long"


def test_empty_bytes_rejected():
    with pytest.raises(AudioError) as e:
        audio_io.load_bytes(b"")
    assert e.value.code == "empty"


def test_short_clip_is_one_shorter_window_with_warning(audio_dir):
    la = audio_io.load(str(audio_dir / "short3s.wav"))
    spans = audio_io.select_windows(la.samples)
    assert len(spans) == 1 and spans[0] == (0, len(la.samples))
    assert any("shorter than" in w for w in la.warnings)


def test_long_clip_gives_full_non_overlapping_windows(audio_dir):
    la = audio_io.load(str(audio_dir / "long12s.wav"))
    spans = audio_io.select_windows(la.samples)
    W = int(AUDIO.window_seconds * AUDIO.sample_rate)
    assert len(spans) == 3
    assert all(e - s == W for s, e in spans)
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_max_windows_is_respected_and_evenly_spread():
    cfg = AudioConfig(max_windows_per_file=3)
    y = np.random.default_rng(0).standard_normal(22050 * 40).astype(np.float32) * 0.1
    spans = audio_io.select_windows(y, cfg)
    assert len(spans) == 3
    assert spans[0][0] == 0 and spans[-1][1] == 22050 * 40


def test_silent_windows_are_skipped():
    sr = 22050
    rng = np.random.default_rng(0)
    loud = (0.2 * rng.standard_normal(sr * 4)).astype(np.float32)
    quiet = np.zeros(sr * 4, dtype=np.float32)
    y = np.concatenate([loud, quiet, loud])
    spans = audio_io.select_windows(y)
    assert [s for s, _ in spans] == [0, 2 * sr * 4]


def test_load_array_accepts_int16_and_channel_first(audio_dir):
    import soundfile as sf
    y, sr = sf.read(audio_dir / "mono22k.wav", dtype="int16")
    a = audio_io.load_array(y, sr)
    b = audio_io.load_array(np.vstack([y, y]), sr)  # (channels, samples) layout
    np.testing.assert_allclose(a.samples, b.samples, atol=1e-4)
    with pytest.raises(AudioError):
        audio_io.load_array(np.array([]), sr)
    with pytest.raises(AudioError) as e:
        audio_io.load_array(np.full(sr * 2, np.nan, dtype=np.float32), sr)
    assert e.value.code == "invalid_samples"
