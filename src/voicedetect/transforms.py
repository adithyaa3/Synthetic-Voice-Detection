"""Audio perturbations shared by augmentation (training folds only) and the robustness suite.

All functions take/return mono float32 at sample rate ``sr`` and never modify their input.
MP3 encoding uses libsndfile's built-in LAME/mpg123 support (no ffmpeg needed).
"""
from __future__ import annotations

import io
from functools import lru_cache

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


def _match_length(z: np.ndarray, n: int) -> np.ndarray:
    if len(z) >= n:
        return z[:n]
    return np.pad(z, (0, n - len(z)))


MPEG_BITRATES = (8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256, 320)


@lru_cache(maxsize=None)
def _mp3_level_table(codec_sr: int) -> tuple[tuple[float, float], ...]:
    """Measure the CBR bitrate (kbps) each libsndfile compression level produces at ``codec_sr``.
    The measured size includes header frames, so it is snapped to the standard MPEG bitrate ladder."""
    rng = np.random.default_rng(0)
    seconds = 4
    probe = (0.1 * rng.standard_normal(codec_sr * seconds)).astype(np.float32)
    table = []
    for level in np.round(np.arange(0.0, 0.99, 0.02), 2):
        buf = io.BytesIO()
        try:
            sf.write(buf, probe, codec_sr, format="MP3", compression_level=float(level), bitrate_mode="CONSTANT")
        except Exception:
            continue
        measured = buf.tell() * 8 / seconds / 1000
        table.append((float(level), float(min(MPEG_BITRATES, key=lambda b: abs(b - measured)))))
    if not table:
        raise RuntimeError("MP3 encoding is not available in this libsndfile build")
    return tuple(table)


def mp3_level_for(kbps: float, codec_sr: int) -> tuple[float, float]:
    """(compression_level, achieved_kbps) closest to the requested bitrate."""
    return min(_mp3_level_table(codec_sr), key=lambda t: abs(t[1] - kbps))


def mp3_roundtrip(y: np.ndarray, sr: int, kbps: float = 48, codec_sr: int | None = None) -> np.ndarray:
    """Encode to CBR MP3 at ~``kbps`` (optionally at ``codec_sr``) and decode back to ``sr``."""
    codec_sr = codec_sr or sr
    x = y if codec_sr == sr else librosa.resample(y, orig_sr=sr, target_sr=codec_sr, res_type="soxr_hq")
    level, _ = mp3_level_for(kbps, codec_sr)
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1, 1).astype(np.float32), codec_sr, format="MP3",
             compression_level=level, bitrate_mode="CONSTANT")
    buf.seek(0)
    z, zsr = sf.read(buf, dtype="float32")
    if z.ndim == 2:
        z = z.mean(axis=1)
    if zsr != sr:
        z = librosa.resample(z, orig_sr=zsr, target_sr=sr, res_type="soxr_hq")
    return _match_length(z, len(y)).astype(np.float32)


def colored_noise(n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n)
    if kind == "white":
        return white
    spec = np.fft.rfft(white)
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / (np.sqrt(f) if kind == "pink" else f)  # pink 1/f power, brown 1/f^2 power
    out = np.fft.irfft(spec, n)
    return out / (np.std(out) + 1e-12)


def add_noise(y: np.ndarray, snr_db: float, kind: str = "white", seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(np.square(y, dtype=np.float64)))
    noise = colored_noise(len(y), kind, rng)
    noise *= np.sqrt(p_sig / (10 ** (snr_db / 10)) / (np.mean(noise ** 2) + 1e-12))
    return (y + noise).astype(np.float32)


def resample_roundtrip(y: np.ndarray, sr: int, low_sr: int) -> np.ndarray:
    z = librosa.resample(y, orig_sr=sr, target_sr=low_sr, res_type="soxr_hq")
    z = librosa.resample(z, orig_sr=low_sr, target_sr=sr, res_type="soxr_hq")
    return _match_length(z, len(y)).astype(np.float32)


def lowpass(y: np.ndarray, sr: int, cutoff_hz: float, order: int = 8) -> np.ndarray:
    sos = signal.butter(order, cutoff_hz, btype="low", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def gain(y: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(y * factor, -1.0, 1.0).astype(np.float32)


def truncate(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    return y[: int(round(seconds * sr))].copy()


def reverb(y: np.ndarray, sr: int, rt60: float = 0.3, seed: int = 0, wet: float = 0.3) -> np.ndarray:
    """Synthetic small-room reverb: exponentially decaying noise impulse response."""
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)  # -60 dB at rt60
    ir[0] = 1.0
    ir /= np.sqrt(np.sum(ir ** 2))
    wet_sig = signal.fftconvolve(y, ir)[: len(y)]
    out = (1 - wet) * y + wet * wet_sig
    return (out * (np.max(np.abs(y)) / (np.max(np.abs(out)) + 1e-12))).astype(np.float32)


def random_noise(y: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """Training augmentation: white or pink noise at a random SNR in [10, 30] dB."""
    rng = np.random.default_rng(seed)
    return add_noise(y, float(rng.uniform(10, 30)), str(rng.choice(["white", "pink"])), seed)


def random_mic(y: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """Training augmentation: a randomised microphone/room chain (band limit 5-8 kHz, reverb RT60
    0.2-0.6 s, pink noise 15-30 dB SNR, 16 kHz capture, gain 0.3-1.0)."""
    rng = np.random.default_rng(seed)
    z = lowpass(y, sr, float(rng.uniform(5000, 8000)))
    z = reverb(z, sr, rt60=float(rng.uniform(0.2, 0.6)), seed=seed)
    z = add_noise(z, float(rng.uniform(15, 30)), "pink", seed=seed)
    z = resample_roundtrip(z, sr, 16000)
    return gain(z, float(rng.uniform(0.3, 1.0)))


def mic_like(y: np.ndarray, sr: int, seed: int = 0) -> np.ndarray:
    """Approximates a laptop/phone microphone in a room: band-limit to ~7 kHz, small-room reverb,
    pink 'room' noise at 20 dB SNR, 16 kHz capture, and lower level."""
    z = lowpass(y, sr, 7000)
    z = reverb(z, sr, rt60=0.35, seed=seed)
    z = add_noise(z, 20, "pink", seed=seed)
    z = resample_roundtrip(z, sr, 16000)
    return gain(z, 0.6)
