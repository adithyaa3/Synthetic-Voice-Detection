"""Robustness of the FINAL model on the untouched test windows under audio perturbations."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from joblib.externals.loky.process_executor import BrokenProcessPool

from . import audio_io, transforms as T
from .config import AUDIO, ROOT, FeatureConfig
from .features import extract
from .metrics import binary_metrics

SR = AUDIO.sample_rate

# (name, family, description, function)
CONDITIONS = [
    ("clean", "reference", "Unmodified test windows", lambda y, s: y),
    ("white_noise_30dB", "noise", "White noise, SNR 30 dB", lambda y, s: T.add_noise(y, 30, "white", s)),
    ("white_noise_20dB", "noise", "White noise, SNR 20 dB", lambda y, s: T.add_noise(y, 20, "white", s)),
    ("white_noise_10dB", "noise", "White noise, SNR 10 dB", lambda y, s: T.add_noise(y, 10, "white", s)),
    ("white_noise_5dB", "noise", "White noise, SNR 5 dB", lambda y, s: T.add_noise(y, 5, "white", s)),
    ("room_noise_20dB", "noise", "Pink (room-like) noise, SNR 20 dB", lambda y, s: T.add_noise(y, 20, "pink", s)),
    ("room_noise_10dB", "noise", "Pink (room-like) noise, SNR 10 dB", lambda y, s: T.add_noise(y, 10, "pink", s)),
    ("mp3_128k", "compression", "MP3 128 kbps (32 kHz)", lambda y, s: T.mp3_roundtrip(y, SR, 128, 32000)),
    ("mp3_64k", "compression", "MP3 64 kbps (32 kHz)", lambda y, s: T.mp3_roundtrip(y, SR, 64, 32000)),
    ("mp3_48k", "compression", "MP3 48 kbps (32 kHz) = Common Voice format", lambda y, s: T.mp3_roundtrip(y, SR, 48, 32000)),
    ("mp3_32k", "compression", "MP3 32 kbps (22.05 kHz)", lambda y, s: T.mp3_roundtrip(y, SR, 32, 22050)),
    ("resample_16k", "resampling", "Down to 16 kHz and back", lambda y, s: T.resample_roundtrip(y, SR, 16000)),
    ("resample_8k", "resampling", "Down to 8 kHz (telephone) and back", lambda y, s: T.resample_roundtrip(y, SR, 8000)),
    ("truncate_1s", "length", "First 1 s only", lambda y, s: T.truncate(y, SR, 1.0)),
    ("truncate_2s", "length", "First 2 s only", lambda y, s: T.truncate(y, SR, 2.0)),
    ("truncate_4s", "length", "First 4 s only (= window length; ~clean)", lambda y, s: T.truncate(y, SR, 4.0)),
    ("volume_0.1x", "volume", "Volume x0.1 (-20 dB)", lambda y, s: T.gain(y, 0.1)),
    ("volume_0.5x", "volume", "Volume x0.5 (-6 dB)", lambda y, s: T.gain(y, 0.5)),
    ("volume_2x_clip", "volume", "Volume x2 (+6 dB) with clipping", lambda y, s: T.gain(y, 2.0)),
    ("lowpass_4k", "microphone", "Low-pass 4 kHz", lambda y, s: T.lowpass(y, SR, 4000)),
    ("reverb_room", "microphone", "Small-room reverb (RT60 0.35 s)", lambda y, s: T.reverb(y, SR, 0.35, s)),
    ("mic_like", "microphone", "Mic-like: low-pass 7 kHz + reverb + room noise 20 dB + 16 kHz + quieter", lambda y, s: T.mic_like(y, SR, s)),
]


def _file_features(path: str, spans: list[tuple[int, int]], seeds: list[int], fcfg: FeatureConfig) -> list[dict]:
    la = audio_io.load(str(ROOT / path), AUDIO, max_duration_s=None)
    out = []
    for (s, e), seed in zip(spans, seeds):
        w = la.samples[s:e]
        for name, _, _, fn in CONDITIONS:
            z = fn(w, seed)
            try:
                out.append({"start": s, "condition": name, "x": extract(z, SR, fcfg)})
            except Exception as exc:  # e.g. a 1 s truncation of a window that is already shorter
                out.append({"start": s, "condition": name, "x": None, "error": str(exc)})
    return out


def run(bundle, test_rows: pd.DataFrame, n_jobs: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """test_rows: orig test windows (file_id, path, start, end, label, source, generator, group)."""
    fcfg = bundle.feature_config
    jobs = []
    for path, d in test_rows.groupby("path", sort=False):
        spans = list(zip(d.start.astype(int), d.end.astype(int)))
        seeds = [zlib.crc32(f"{path}:{int(s)}".encode()) for s, _ in spans]  # stable across runs
        jobs.append((path, spans, seeds))
    try:
        res = Parallel(n_jobs=n_jobs)(delayed(_file_features)(p, sp, se, fcfg) for p, sp, se in jobs)
    except BrokenProcessPool:
        # Observed once when this step ran in the same process after a grid search had spawned the shared
        # loky pool; the identical call succeeds with fresh workers, so reset the pool and retry once.
        print("worker pool broken; retrying robustness feature extraction with a fresh pool", flush=True)
        get_reusable_executor().shutdown(wait=True)
        res = Parallel(n_jobs=n_jobs)(delayed(_file_features)(p, sp, se, fcfg) for p, sp, se in jobs)
    rows = []
    key = test_rows.set_index(["path", "start"])
    for (path, _, _), items in zip(jobs, res):
        for it in items:
            if it["x"] is None:
                continue
            r = key.loc[(path, it["start"])]
            rows.append({"path": path, "start": it["start"], "condition": it["condition"], "x": it["x"],
                         "label": int(r.label), "source": r.source, "generator": r.generator, "file_id": r.file_id})
    df = pd.DataFrame(rows)
    X = np.vstack(df.x.values)
    df["p_ai"] = bundle.pipeline.predict_proba(X)[:, 1]
    df = df.drop(columns="x")
    t = bundle.threshold
    meta = {c[0]: (c[1], c[2]) for c in CONDITIONS}
    summary = []
    for cond, d in df.groupby("condition", sort=False):
        for scope, dd in [("all test windows", d)] + [(f"source: {s}", x) for s, x in d.groupby("source")]:
            m = binary_metrics(dd.label.values, dd.p_ai.values, t)
            summary.append({"condition": cond, "family": meta[cond][0], "description": meta[cond][1], "scope": scope,
                            "n": m["n"], "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                            "roc_auc": m["roc_auc"], "recall_ai": m["recall"], "specificity_human": m["specificity"],
                            "mean_p_ai_human": float(dd.loc[dd.label == 0, "p_ai"].mean()) if (dd.label == 0).any() else np.nan,
                            "mean_p_ai_ai": float(dd.loc[dd.label == 1, "p_ai"].mean()) if (dd.label == 1).any() else np.nan})
    order = {c[0]: i for i, c in enumerate(CONDITIONS)}
    summary = pd.DataFrame(summary)
    summary["_o"] = summary.condition.map(order)
    summary = summary.sort_values(["_o", "scope"]).drop(columns="_o").reset_index(drop=True)
    return summary, df
