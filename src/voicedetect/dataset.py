"""Window-level feature table built from the manifest, with a content+config keyed cache.

For every readable file: load -> select analysis windows -> extended features (the baseline
features are exactly its first 80 columns) for each window, in two variants:
  orig   : the audio as stored
  mp3cv  : the window re-encoded as 32 kHz / 48 kbps CBR MP3, i.e. the exact Common Voice format.
           Used (a) for the codec-matched control experiment and (b) as training-fold augmentation.
Cache key: features/cache/<config hash>/<file sha256>.npz, so changing the audio/feature config, the
feature code version or the file content invalidates the entry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import audio_io, transforms
from .config import AUDIO, FEATURE_CACHE_DIR, FEATURES_DIR, REPORTS_DIR, ROOT, AudioConfig, FeatureConfig, config_hash
from .features import extract, feature_names

EXTRACT_CFG = FeatureConfig("extended")
VARIANTS = ("orig", "mp3cv")
MP3CV = {"kbps": 48, "codec_sr": 32000}
META_COLS = ["file_id", "path", "label", "class_name", "origin", "source", "supplementary", "group",
             "speaker", "target_voice", "generator", "official_partition", "partition", "sha256"]


def cache_key(audio: AudioConfig = AUDIO) -> str:
    return config_hash(audio, EXTRACT_CFG, extra={"variants": VARIANTS, "mp3cv": MP3CV})


def _process_file(path: str, sha: str, cache_dir: Path, audio: AudioConfig) -> dict:
    cache = cache_dir / f"{sha}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files} | {"status": "ok", "cached": True}
    try:
        la = audio_io.load(str(ROOT / path), audio, max_duration_s=None)
    except audio_io.AudioError as e:
        return {"status": e.code, "detail": e.detail}
    spans = audio_io.select_windows(la.samples, audio)
    X = {v: [] for v in VARIANTS}
    for s, e in spans:
        w = la.samples[s:e]
        X["orig"].append(extract(w, audio.sample_rate, EXTRACT_CFG))
        w_mp3 = transforms.mp3_roundtrip(w, audio.sample_rate, kbps=MP3CV["kbps"], codec_sr=MP3CV["codec_sr"])
        X["mp3cv"].append(extract(w_mp3, audio.sample_rate, EXTRACT_CFG))
    out = {"starts": np.array([s for s, _ in spans]), "ends": np.array([e for _, e in spans])}
    out |= {f"X_{v}": np.vstack(X[v]) for v in VARIANTS}
    np.savez_compressed(cache, **out)
    return out | {"status": "ok", "cached": False}


def build_table(manifest: pd.DataFrame, audio: AudioConfig = AUDIO, n_jobs: int = 8, verbose: bool = True) -> pd.DataFrame:
    key = cache_key(audio)
    cache_dir = FEATURE_CACHE_DIR / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_process_file)(r.path, r.sha256, cache_dir, audio) for r in manifest.itertuples(index=False))
    names = list(feature_names("extended", EXTRACT_CFG.n_mfcc))
    blocks, skips = [], []
    for r, res in zip(manifest.itertuples(index=False), results):
        if res["status"] != "ok":
            skips.append({"file_id": r.file_id, "path": r.path, "source": r.source, "class_name": r.class_name,
                          "reason": res["status"], "detail": res.get("detail", "")})
            continue
        n = len(res["starts"])
        meta = {c: getattr(r, c) for c in META_COLS}
        for v in VARIANTS:
            df = pd.DataFrame(res[f"X_{v}"], columns=names)
            df.insert(0, "variant", v)
            df.insert(0, "end", res["ends"])
            df.insert(0, "start", res["starts"])
            df.insert(0, "window_idx", np.arange(n))
            for c in reversed(META_COLS):
                df.insert(0, c, meta[c])
            blocks.append(df)
    table = pd.concat(blocks, ignore_index=True)
    skips_df = pd.DataFrame(skips, columns=["file_id", "path", "source", "class_name", "reason", "detail"])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    skips_df.to_csv(REPORTS_DIR / "feature_skips.csv", index=False)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(FEATURES_DIR / f"windows_{key}.parquet", index=False)
    summary = {
        "cache_key": key, "files": int(len(manifest)), "files_ok": int(len(manifest) - len(skips_df)),
        "files_skipped": int(len(skips_df)), "skipped_by_reason": skips_df.reason.value_counts().to_dict(),
        "windows_orig": int((table.variant == "orig").sum()),
        "windows_by_source_class": {f"{s}/{c}": int(n) for (s, c), n in
                                    table[table.variant == "orig"].groupby(["source", "class_name"]).size().items()},
        "seconds": round(time.time() - t0, 1),
    }
    (REPORTS_DIR / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(summary, indent=2))
    return table


# ---------------------------------------------------------------------------------------------
# Robustness variants (added after the test-set robustness analysis showed noise sensitivity; see
# docs/IMPLEMENTATION_STATUS.md). Kept in a SEPARATE table/cache so the original table and every
# result computed from it stay reproducible. Rows are appended after the base rows.
#   aug_noise / aug_mic       : randomised training augmentations (used in training folds only)
#   ev_white20 / ev_room20 / ev_mic : fixed perturbations used only to SCORE validation windows
AUG_VARIANTS = {
    "aug_noise": lambda w, sr, s: transforms.random_noise(w, sr, s),
    "aug_mic": lambda w, sr, s: transforms.random_mic(w, sr, s),
    "ev_white20": lambda w, sr, s: transforms.add_noise(w, 20, "white", s),
    "ev_room20": lambda w, sr, s: transforms.add_noise(w, 20, "pink", s),
    "ev_mic": lambda w, sr, s: transforms.mic_like(w, sr, s),
}
AUG_VERSION = "1"


def aug_cache_key(audio: AudioConfig = AUDIO) -> str:
    return config_hash(audio, EXTRACT_CFG, extra={"aug_variants": sorted(AUG_VARIANTS), "aug_version": AUG_VERSION})


def _process_file_aug(path: str, sha: str, starts: np.ndarray, ends: np.ndarray, cache_dir: Path,
                      audio: AudioConfig) -> dict:
    import zlib
    cache = cache_dir / f"{sha}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    la = audio_io.load(str(ROOT / path), audio, max_duration_s=None)
    out = {}
    for v, fn in AUG_VARIANTS.items():
        rows = []
        for i, (s, e) in enumerate(zip(starts, ends)):
            seed = zlib.crc32(f"{sha}:{i}:{v}".encode())  # stable, distinct per window and variant
            rows.append(extract(fn(la.samples[s:e], audio.sample_rate, seed), audio.sample_rate, EXTRACT_CFG))
        out[f"X_{v}"] = np.vstack(rows)
    np.savez_compressed(cache, **out)
    return out


def build_aug_table(base: pd.DataFrame, audio: AudioConfig = AUDIO, n_jobs: int = 8, verbose: bool = True) -> pd.DataFrame:
    """Robustness variants for every window of the base table (same files, windows and groups)."""
    key = aug_cache_key(audio)
    cache_dir = FEATURE_CACHE_DIR / f"aug_{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    orig = base[base.variant == "orig"]
    files = [(fid, d.sort_values("window_idx")) for fid, d in orig.groupby("file_id", sort=True)]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(_process_file_aug)(d.path.iloc[0], d.sha256.iloc[0], d.start.to_numpy(),
                                                             d.end.to_numpy(), cache_dir, audio) for _, d in files)
    names = list(feature_names("extended", EXTRACT_CFG.n_mfcc))
    blocks = []
    for (fid, d), r in zip(files, res):
        meta = d.drop(columns=names).reset_index(drop=True)
        for v in AUG_VARIANTS:
            block = pd.concat([meta.assign(variant=v), pd.DataFrame(r[f"X_{v}"], columns=names)], axis=1)
            blocks.append(block)
    table = pd.concat(blocks, ignore_index=True)[list(base.columns)]
    table.to_parquet(FEATURES_DIR / f"windows_aug_{key}.parquet", index=False)
    if verbose:
        print(f"augmented variants: {len(table)} rows ({len(AUG_VARIANTS)} x {len(orig)} windows) in {time.time() - t0:.0f}s")
    return table


def load_table(audio: AudioConfig = AUDIO, include_aug: bool = True) -> pd.DataFrame:
    """Base table (orig + mp3cv) with the robustness variants appended when they exist. Appending keeps
    every base row at the same position, so row indices saved by earlier steps stay valid."""
    p = FEATURES_DIR / f"windows_{cache_key(audio)}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run the 'features' step first")
    base = pd.read_parquet(p)
    pa = FEATURES_DIR / f"windows_aug_{aug_cache_key(audio)}.parquet"
    if include_aug and pa.exists():
        return pd.concat([base, pd.read_parquet(pa)], ignore_index=True)
    return base


def feature_columns(feature_set: str) -> list[str]:
    return list(feature_names(feature_set, EXTRACT_CFG.n_mfcc))


def window_audio(path: str, start: int, end: int, audio: AudioConfig = AUDIO) -> np.ndarray:
    """Re-create a window's samples exactly as the table was built (used by robustness/UI)."""
    la = audio_io.load(str(ROOT / path), audio, max_duration_s=None)
    return la.samples[start:end]
