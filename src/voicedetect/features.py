"""THE feature function. Training, evaluation, robustness, prediction and the UI all call
:func:`extract` on analysis windows produced by :mod:`voicedetect.audio_io`.

baseline (80)  : 40 MFCC means + 40 MFCC standard deviations; identical to the original script
                 (librosa defaults n_fft=2048, hop=512, 128 mel bands).
extended (210) : baseline + std of delta and delta-delta MFCCs (80) + mean/std of spectral centroid,
                 bandwidth, roll-off, flatness, zero-crossing rate, RMS (12) + spectral contrast
                 mean/std (14) + chroma mean/std (24).
(Means of delta MFCCs are ~0 by construction and carry no information, so only their std is used.)
"""
from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from .config import FeatureConfig


class FeatureError(Exception):
    pass


FEATURE_SETS = ("baseline", "extended")


@lru_cache(maxsize=None)
def feature_names(feature_set: str = "baseline", n_mfcc: int = 40) -> tuple[str, ...]:
    names = [f"mfcc{i:02d}_mean" for i in range(n_mfcc)] + [f"mfcc{i:02d}_std" for i in range(n_mfcc)]
    if feature_set == "baseline":
        return tuple(names)
    if feature_set != "extended":
        raise ValueError(f"unknown feature set {feature_set!r}")
    names += [f"dmfcc{i:02d}_std" for i in range(n_mfcc)] + [f"ddmfcc{i:02d}_std" for i in range(n_mfcc)]
    for s in ("centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"):
        names += [f"{s}_mean", f"{s}_std"]
    names += [f"contrast{i}_mean" for i in range(7)] + [f"contrast{i}_std" for i in range(7)]
    names += [f"chroma{i:02d}_mean" for i in range(12)] + [f"chroma{i:02d}_std" for i in range(12)]
    return tuple(names)


def feature_group(name: str) -> str:
    """Coarse group used for ablation and importance summaries."""
    if name.startswith("mfcc") and name.endswith("_mean"):
        return "mfcc_mean"
    if name.startswith("mfcc") and name.endswith("_std"):
        return "mfcc_std"
    if name.startswith("ddmfcc"):
        return "delta2_mfcc"
    if name.startswith("dmfcc"):
        return "delta_mfcc"
    if name.split("_")[0] in ("centroid", "bandwidth", "rolloff", "flatness"):
        return "spectral_shape"
    if name.split("_")[0] in ("zcr", "rms"):
        return "zcr_rms"
    if name.startswith("contrast"):
        return "spectral_contrast"
    if name.startswith("chroma"):
        return "chroma"
    raise ValueError(name)


def _ms(x: np.ndarray) -> np.ndarray:
    """Mean then std over time; scalar tracks give 2 values, (k, t) matrices give 2k."""
    return np.concatenate([np.atleast_1d(x.mean(axis=-1)), np.atleast_1d(x.std(axis=-1))])


def extract(window: np.ndarray, sr: int, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Feature vector (float64) for ONE analysis window of mono audio at ``sr``."""
    y = np.ascontiguousarray(window, dtype=np.float32)
    if y.ndim != 1 or y.size < cfg.n_fft:
        raise FeatureError(f"window too short or not mono (shape {y.shape})")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft,
                                hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    parts = [mfcc.mean(axis=1), mfcc.std(axis=1)]
    if cfg.feature_set == "extended":
        width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
        parts += [librosa.feature.delta(mfcc, width=width, order=1).std(axis=1),
                  librosa.feature.delta(mfcc, width=width, order=2).std(axis=1)]
        S = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
        parts += [
            _ms(librosa.feature.spectral_centroid(S=S, sr=sr)[0]),
            _ms(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]),
            _ms(librosa.feature.spectral_rolloff(S=S, sr=sr)[0]),
            _ms(librosa.feature.spectral_flatness(S=S)[0]),
            _ms(librosa.feature.zero_crossing_rate(y, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]),
            _ms(librosa.feature.rms(S=S, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]),
            _ms(librosa.feature.spectral_contrast(S=S, sr=sr)),
            _ms(librosa.feature.chroma_stft(S=S ** 2, sr=sr)),
        ]
    elif cfg.feature_set != "baseline":
        raise ValueError(f"unknown feature set {cfg.feature_set!r}")
    vec = np.concatenate(parts).astype(np.float64)
    if vec.shape[0] != len(feature_names(cfg.feature_set, cfg.n_mfcc)):
        raise FeatureError(f"feature length {vec.shape[0]} does not match names")
    if not np.all(np.isfinite(vec)):
        raise FeatureError("non-finite feature values")
    return vec


def extract_many(windows: list[np.ndarray], sr: int, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    return np.vstack([extract(w, sr, cfg) for w in windows])
