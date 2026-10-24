"""Dataset audit -> reports/dataset_audit.json and reports/dataset_audit.csv."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed

from . import audio_io
from .config import AUDIO, REPORTS_DIR, ROOT, FeatureConfig
from .features import extract

SHORT_S = 2.0          # "very short" if less than this much sound after trimming
NEAR_SILENT_DBFS = -45  # whole-file RMS below this (but above the hard silence floor) is flagged


def _probe(path: str) -> dict:
    p = ROOT / path
    out = {"path": path, "readable": False, "error": ""}
    try:
        info = sf.info(str(p))
        out.update(sample_rate=info.samplerate, channels=info.channels, duration_s=info.duration,
                   format=info.format, subtype=info.subtype)
        y, sr, ch, dur = audio_io.decode(str(p), max_duration_s=None)
        rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
        out.update(readable=True, rms_dbfs=20 * np.log10(max(rms, 1e-12)), peak=float(np.max(np.abs(y))))
        try:
            yt, _ = audio_io.prepare(y, sr, AUDIO)
            out["trimmed_duration_s"] = len(yt) / AUDIO.sample_rate
            out["n_windows"] = len(audio_io.select_windows(yt, AUDIO))
            out["loader_status"] = "ok"
            first = yt[: int(AUDIO.window_seconds * AUDIO.sample_rate)]
            out["_mfcc_mean"] = extract(first, AUDIO.sample_rate, FeatureConfig("baseline"))[1:40].tolist()
        except audio_io.AudioError as e:
            out["loader_status"] = e.code
            out["trimmed_duration_s"] = np.nan
            out["n_windows"] = 0
    except Exception as e:  # unreadable / corrupt
        out["error"] = f"{type(e).__name__}: {e}"[:200]
        out["loader_status"] = getattr(e, "code", "undecodable")
    return out


def group_coherence(df: pd.DataFrame) -> dict:
    """Nearest-neighbour purity of inferred groups using cepstral-mean vectors (a speaker/channel
    signature). Purity well above chance supports the grouping; it cannot prove it."""
    d = df[df["_mfcc_mean"].notna()]
    if len(d) < 3 or d["group"].nunique() < 2:
        return {}
    X = np.vstack(d["_mfcc_mean"].values)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    g = d["group"].values
    purity = float(np.mean(g[nn] == g))
    sizes = d.groupby("group").size()
    chance = float(np.sum(sizes * (sizes - 1)) / (len(d) * (len(d) - 1)))
    return {"n_clips": int(len(d)), "n_groups": int(d["group"].nunique()),
            "nearest_neighbour_same_group": round(purity, 3), "chance_level": round(chance, 3),
            "clips_whose_nearest_neighbour_is_in_another_group": int(np.sum(g[nn] != g))}


def _desc(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {}
    return {k: round(float(v), 3) for k, v in
            {"min": s.min(), "p25": s.quantile(.25), "median": s.median(), "mean": s.mean(), "p75": s.quantile(.75), "max": s.max()}.items()}


def adequacy(df: pd.DataFrame, name: str) -> dict:
    ok = df[df.readable]
    n_h, n_a = int((ok.label == 0).sum()), int((ok.label == 1).sum())
    g_h = int(ok.loc[ok.label == 0, "group"].nunique())
    g_a = int(ok.loc[ok.label == 1, "group"].nunique())
    gens = sorted(set(ok.loc[ok.label == 1, "generator"]))
    checks = {
        "both_classes_present": n_h > 0 and n_a > 0,
        ">=100_independent_human_units": g_h >= 100 or n_h >= 100,
        ">=100_independent_ai_units": g_a >= 100,
        "more_than_one_ai_generator": len(gens) > 1,
        "more_than_one_human_speaker_or_condition": g_h > 1,
        "file_level_class_ratio_within_1:3": (min(n_h, n_a) / max(n_h, n_a) >= 1 / 3) if min(n_h, n_a) else False,
        "no_exact_duplicates": int(ok.duplicated("sha256").sum()) == 0,
    }
    return {"dataset": name, "files_human": n_h, "files_ai": n_a,
            "independent_groups_human": g_h, "independent_groups_ai": g_a, "ai_generators": gens,
            "checks": checks, "adequate": all(checks.values())}


def run_audit(manifest: pd.DataFrame, n_jobs: int = 6) -> dict:
    probes = Parallel(n_jobs=n_jobs)(delayed(_probe)(p) for p in manifest.path)
    df = manifest.merge(pd.DataFrame(probes), on="path", how="left")
    df["near_silent"] = df.readable & (df.rms_dbfs < NEAR_SILENT_DBFS)
    df["very_short"] = df.readable & (df.trimmed_duration_s.fillna(0) < SHORT_S)
    dup = df[df.duplicated("sha256", keep=False)].groupby("sha256").path.apply(list).tolist()

    per_source = {}
    for src, d in df.groupby("source"):
        per_source[src] = {
            "files": int(len(d)), "class": sorted(d.class_name.unique().tolist()),
            "origin": d.origin.iloc[0],
            "formats": d.ext.value_counts().to_dict(),
            "sample_rates": {str(k): int(v) for k, v in d.sample_rate.value_counts().items()},
            "channels": {str(k): int(v) for k, v in d.channels.value_counts().items()},
            "duration_s": _desc(d.duration_s), "trimmed_duration_s": _desc(d.trimmed_duration_s),
            "rms_dbfs": _desc(d.rms_dbfs),
            "unreadable": d.loc[~d.readable, "path"].tolist(),
            "rejected_by_loader": d.loc[d.loader_status != "ok", ["path", "loader_status"]].values.tolist(),
            "near_silent": d.loc[d.near_silent, "path"].tolist(),
            "very_short": d.loc[d.very_short, "path"].tolist(),
            "groups": int(d.group.nunique()), "group_basis": d.group_basis.iloc[0],
            "group_confidence": d.group_confidence.iloc[0],
            "group_size": _desc(d.groupby("group").size().astype(float)),
            "generators": d.generator.value_counts().to_dict(),
            "windows_total": int(d.n_windows.fillna(0).sum()),
            "partition_files": d.partition.value_counts().to_dict(),
        }
    cv = df[df.source == "user_commonvoice"]
    vc = df[df.source == "user_voice_conversion"]
    vc_dur = vc.groupby("group").duration_s.agg(lambda s: int(s.round(2).nunique())).to_dict() if len(vc) else {}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files_total": int(len(df)),
        "files_per_class": df.class_name.value_counts().to_dict(),
        "files_per_source": df.source.value_counts().to_dict(),
        "per_source": per_source,
        "exact_duplicates": dup,
        "unreadable_total": int((~df.readable).sum()),
        "rejected_by_loader_total": int((df.loader_status != "ok").sum()),
        "group_inference": {
            "user_voice_conversion": {
                "basis": "filename '<source>-to-<target>'; group = source speaker",
                "distinct_durations_per_source_group": vc_dur,
                "interpretation": "1 distinct duration per source => the 7 conversions of a source are re-voicings of the same recording",
                "source_speakers": sorted(vc.speaker.unique().tolist()) if len(vc) else [],
                "target_voices": sorted(vc.target_voice.unique().tolist()) if len(vc) else [],
                "confidence": "high (filename pattern + identical durations); generator tool unknown, assumed single",
            },
            "user_commonvoice": {
                "basis": "runs of near-consecutive clip IDs (gap <= 5); Common Voice records in batches of 5",
                "coherence_check": group_coherence(cv),
                "confidence": "low-medium (no client_id metadata; one speaker may span several runs)",
            },
            "asvspoof2019_la": {"basis": "official protocol speaker IDs and attack system IDs", "confidence": "high"},
        },
        "adequacy": {
            "user_only": adequacy(df[df.origin == "user"], "user data only"),
            "user_plus_supplementary": adequacy(df, "user + supplementary"),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = [c for c in df.columns if not c.startswith("_")]
    df[out_cols].to_csv(REPORTS_DIR / "dataset_audit.csv", index=False)
    with open(REPORTS_DIR / "dataset_audit.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return report
