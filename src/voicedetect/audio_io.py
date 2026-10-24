"""Safe audio loading and analysis-window selection.

Policy (documented in README / docs):
* Any channel count -> mono by averaging channels; any sample rate -> AudioConfig.sample_rate (soxr_hq).
* Leading/trailing silence is trimmed (``trim_top_db`` below peak).
* Trimmed audio shorter than ``min_duration_s`` is rejected (no padding: padding with zeros would
  change the MFCC statistics). Clips between ``min_duration_s`` and one window length are analysed
  as a single, shorter window.
* Longer audio is cut into consecutive non-overlapping full windows; windows with too little active
  speech are skipped; at most ``max_windows_per_file`` evenly spaced windows are kept.
* Silent / non-finite / undecodable input raises :class:`AudioError` with a machine-readable code.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from .config import AUDIO, PREDICT, AudioConfig


class AudioError(Exception):
    """Raised for any input the pipeline refuses to analyse. ``code`` is stable for the UI."""

    MESSAGES = {
        "not_found": "The file does not exist.",
        "unsupported_format": "Unsupported file type.",
        "too_large": "The file is too large.",
        "empty": "The file is empty.",
        "undecodable": "The audio could not be decoded (corrupt or not really audio).",
        "too_short": "The audio is too short to analyse.",
        "too_long": "The audio is too long.",
        "silent": "The audio is silent or almost silent.",
        "invalid_samples": "The audio contains invalid sample values.",
    }

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{self.MESSAGES.get(code, code)} {detail}".strip())


@dataclass
class LoadedAudio:
    samples: np.ndarray            # mono float32 at ``sample_rate``, trimmed
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    original_duration_s: float
    trimmed_duration_s: float
    warnings: list = field(default_factory=list)


def _to_mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def check_path(path: str | os.PathLike, max_file_mb: float = PREDICT.max_file_mb,
               allowed_extensions: tuple = PREDICT.allowed_extensions) -> Path:
    p = Path(path)
    if not p.is_file():
        raise AudioError("not_found", str(p.name))
    if p.suffix.lower() not in allowed_extensions:
        raise AudioError("unsupported_format", f"'{p.suffix}' (allowed: {', '.join(allowed_extensions)})")
    size_mb = p.stat().st_size / 1e6
    if size_mb == 0:
        raise AudioError("empty")
    if size_mb > max_file_mb:
        raise AudioError("too_large", f"({size_mb:.1f} MB > {max_file_mb:.0f} MB)")
    return p


def decode(source, max_duration_s: float | None = PREDICT.max_duration_s) -> tuple[np.ndarray, int, int, float]:
    """Decode a path or a file-like object. Returns (mono float32, sr, channels, duration_s)."""
    try:
        with sf.SoundFile(source) as f:
            sr, channels, frames = f.samplerate, f.channels, f.frames
            duration = frames / sr if sr else 0.0
            if max_duration_s is not None and duration > max_duration_s:
                raise AudioError("too_long", f"({duration:.0f} s > {max_duration_s:.0f} s)")
            data = f.read(dtype="float32", always_2d=True)
    except AudioError:
        raise
    except Exception as exc:  # libsndfile raises several types for corrupt / non-audio input
        raise AudioError("undecodable", type(exc).__name__) from exc
    if data.size == 0 or sr <= 0:
        raise AudioError("empty")
    y = _to_mono(data)
    if not np.all(np.isfinite(y)):
        raise AudioError("invalid_samples")
    return y, int(sr), int(channels), float(len(y) / sr)


def is_silent(y: np.ndarray, cfg: AudioConfig = AUDIO) -> bool:
    if y.size == 0:
        return True
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    rms_db = 20 * np.log10(max(rms, 1e-12))
    return rms_db < cfg.silence_dbfs or float(np.max(np.abs(y))) < cfg.silence_peak


def prepare(y: np.ndarray, sr: int, cfg: AudioConfig = AUDIO) -> tuple[np.ndarray, list]:
    """Resample to the analysis rate, check silence, trim, check duration."""
    warnings = []
    if sr != cfg.sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.sample_rate, res_type="soxr_hq")
    y = np.ascontiguousarray(y, dtype=np.float32)
    if is_silent(y, cfg):
        raise AudioError("silent")
    yt, _ = librosa.effects.trim(y, top_db=cfg.trim_top_db, frame_length=cfg.frame_length,
                                 hop_length=cfg.hop_length)
    dur = len(yt) / cfg.sample_rate
    if dur < cfg.min_duration_s:
        raise AudioError("too_short", f"({dur:.2f} s of sound after trimming silence; minimum {cfg.min_duration_s:.1f} s)")
    if dur < cfg.window_seconds:
        warnings.append(f"Only {dur:.1f} s of sound: analysed as one window shorter than the {cfg.window_seconds:.0f} s used in training.")
    return yt, warnings


def load(source, cfg: AudioConfig = AUDIO, max_duration_s: float | None = PREDICT.max_duration_s) -> LoadedAudio:
    """Decode + prepare. ``source`` may be a path or a binary file-like object."""
    y, sr, ch, dur = decode(source, max_duration_s=max_duration_s)
    warnings = []
    if ch > 1:
        warnings.append(f"{ch}-channel audio was mixed down to mono.")
    if sr < 16000:
        warnings.append(f"Low sample rate ({sr} Hz): high-frequency detail is missing; results may be less reliable.")
    yt, w2 = prepare(y, sr, cfg)
    return LoadedAudio(yt, cfg.sample_rate, sr, ch, dur, len(yt) / cfg.sample_rate, warnings + w2)


def load_bytes(data: bytes, cfg: AudioConfig = AUDIO, max_duration_s: float | None = PREDICT.max_duration_s) -> LoadedAudio:
    """Decode audio fully in memory (nothing is written to disk)."""
    if not data:
        raise AudioError("empty")
    return load(io.BytesIO(data), cfg, max_duration_s)


def load_array(samples: np.ndarray, sample_rate: int, cfg: AudioConfig = AUDIO) -> LoadedAudio:
    """Prepare in-memory audio (e.g. a microphone recording). Accepts (n,), (n, ch) float or int PCM."""
    x = np.asarray(samples)
    if x.size == 0:
        raise AudioError("empty")
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)
    ch = 1
    if x.ndim == 2:
        # accept (n, ch) and the common (ch, n) layout
        if x.shape[0] < x.shape[1] and x.shape[0] <= 8:
            x = x.T
        ch = x.shape[1]
        x = x.mean(axis=1)
    if not np.all(np.isfinite(x)):
        raise AudioError("invalid_samples")
    if sample_rate <= 0:
        raise AudioError("undecodable", "invalid sample rate")
    dur = len(x) / sample_rate
    if dur > PREDICT.max_duration_s:
        raise AudioError("too_long", f"({dur:.0f} s > {PREDICT.max_duration_s:.0f} s)")
    warnings = [f"{ch}-channel audio was mixed down to mono."] if ch > 1 else []
    yt, w2 = prepare(x, int(sample_rate), cfg)
    return LoadedAudio(yt, cfg.sample_rate, int(sample_rate), ch, dur, len(yt) / cfg.sample_rate, warnings + w2)


def frame_activity_db(y: np.ndarray, cfg: AudioConfig = AUDIO) -> np.ndarray:
    rms = librosa.feature.rms(y=y, frame_length=cfg.frame_length, hop_length=cfg.hop_length, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-10))


def select_windows(y: np.ndarray, cfg: AudioConfig = AUDIO, max_windows: int | None = None) -> list[tuple[int, int]]:
    """Return (start, end) sample indices of the analysis windows for an already-prepared signal."""
    max_windows = cfg.max_windows_per_file if max_windows is None else max_windows
    W = int(round(cfg.window_seconds * cfg.sample_rate))
    n = len(y)
    if n <= W:
        return [(0, n)]
    starts = np.arange(0, n - W + 1, W)
    db = frame_activity_db(y, cfg)
    thr = np.percentile(db, 95) - cfg.active_rel_db
    frac = []
    for s in starts:
        f0, f1 = s // cfg.hop_length, (s + W) // cfg.hop_length
        frac.append(float(np.mean(db[f0:f1] > thr)) if f1 > f0 else 0.0)
    frac = np.asarray(frac)
    keep = np.where(frac >= cfg.min_active_fraction)[0]
    if keep.size == 0:
        keep = np.array([int(np.argmax(frac))])
    if keep.size > max_windows:
        keep = keep[np.unique(np.round(np.linspace(0, keep.size - 1, max_windows)).astype(int))]
    return [(int(starts[i]), int(starts[i] + W)) for i in keep]


def windows(audio: LoadedAudio | np.ndarray, cfg: AudioConfig = AUDIO, max_windows: int | None = None) -> list[np.ndarray]:
    y = audio.samples if isinstance(audio, LoadedAudio) else audio
    return [y[s:e] for s, e in select_windows(y, cfg, max_windows)]
