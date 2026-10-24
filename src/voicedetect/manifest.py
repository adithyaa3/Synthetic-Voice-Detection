"""Build data/manifest.csv: one row per audio file with class, source, origin, group and partition.

Group inference (see docs/DATA_SOURCES.md for the evidence):
* user AI files ``<source>-to-<target>.wav``: group = source speaker (all 7 conversions of one source
  share an identical duration, i.e. the same source recording). Confidence: high.
* user human Common Voice clips: speaker IDs are not available. Group = run of near-consecutive clip
  IDs (gap <= 5); Common Voice records sentences in batches of 5, so a run approximates a recording
  session. Confidence: low-medium.
* ASVspoof 2019 LA: group = official speaker ID from the protocol (bona fide and spoofed speech
  of one speaker share a group). Confidence: high.
An optional ``data/metadata.csv`` (columns: path, and any of group, speaker, generator, target_voice)
overrides the inferred values.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ASVSPOOF_DIR, DATA_DIR, MANIFEST_PATH, ROOT, SEED, USER_AI_DIR, USER_HUMAN_DIR

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg")
CV_RUN_GAP = 5
TEST_AI_SOURCES = 2          # user AI source speakers held out as the final test set
TEST_HUMAN_FRACTION = 0.25   # fraction of user human clips (by run group) held out


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def parse_user_ai_name(name: str) -> tuple[str, str]:
    m = re.match(r"^(?P<src>[A-Za-z]+)-to-(?P<tgt>[A-Za-z]+)$", Path(name).stem)
    if not m:
        return "unknown", "unknown"
    return m.group("src").lower(), m.group("tgt").lower()


def common_voice_runs(names: list[str], gap: int = CV_RUN_GAP) -> dict[str, str]:
    ids = {}
    for n in names:
        m = re.search(r"(\d+)$", Path(n).stem)
        ids[n] = int(m.group(1)) if m else None
    known = sorted((v, k) for k, v in ids.items() if v is not None)
    groups, run, prev = {}, 0, None
    for v, k in known:
        if prev is not None and v - prev > gap:
            run += 1
        groups[k] = f"cv_run_{run:03d}"
        prev = v
    for k, v in ids.items():
        if v is None:
            groups[k] = f"cv_file_{Path(k).stem}"
    return groups


def generator_descriptions() -> dict[str, str]:
    """System descriptions parsed from ASVspoof's own README.LA.txt (e.g. 'A04' -> 'TTS, waveform
    concatenation'), plus the user's data. Nothing is filled in from memory."""
    out = {"user_vc": "voice conversion (user data; tool unknown)", "human": "bona fide speech"}
    readme = ASVSPOOF_DIR / "README.LA.txt"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^\s*(A\d\d)\s+(\S+)\s+(.+?)\s*$", line)
            if m:
                out[m.group(1)] = f"{m.group(2).replace('_', '+')}, {m.group(3)}"
    return out


def _rel(p: Path, root: Path | None = None) -> str:
    """Project-relative path WITHOUT resolving links, so a data/ folder that is a symlink or junction to
    another drive still works (found by the from-scratch reproduction run)."""
    root = ROOT if root is None else root
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return Path(os.path.relpath(p, root)).as_posix()


def _user_rows() -> list[dict]:
    rows = []
    human = sorted(p for p in USER_HUMAN_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS) if USER_HUMAN_DIR.exists() else []
    runs = common_voice_runs([p.name for p in human])
    for p in human:
        rows.append(dict(file_id=f"user_human/{p.stem}", path=_rel(p), label=0, class_name="human",
                         origin="user", source="user_commonvoice", supplementary=False,
                         group=runs[p.name], group_basis="Common Voice clip-ID run (gap<=5)",
                         group_confidence="low-medium", speaker="unknown", target_voice="",
                         generator="human", official_partition=""))
    ai = sorted(p for p in USER_AI_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS) if USER_AI_DIR.exists() else []
    for p in ai:
        src, tgt = parse_user_ai_name(p.name)
        rows.append(dict(file_id=f"user_ai/{p.stem}", path=_rel(p), label=1, class_name="ai",
                         origin="user", source="user_voice_conversion", supplementary=False,
                         group=f"vc_src_{src}", group_basis="source speaker in filename (identical durations confirm shared source recording)",
                         group_confidence="high", speaker=src, target_voice=tgt,
                         generator="user_vc", official_partition=""))
    return rows


def _asvspoof_rows() -> list[dict]:
    sel = ASVSPOOF_DIR / "selection.csv"
    if not sel.exists():
        return []
    df = pd.read_csv(sel, dtype=str)
    rows = []
    for r in df.itertuples(index=False):
        p = ASVSPOOF_DIR / "flac" / f"{r.utt}.flac"
        if not p.exists():
            continue
        is_ai = r.key == "spoof"
        rows.append(dict(file_id=f"asvspoof/{r.utt}", path=_rel(p), label=int(is_ai),
                         class_name="ai" if is_ai else "human", origin="public",
                         source="asvspoof2019_la", supplementary=True,
                         group=f"asv_{r.speaker}", group_basis="ASVspoof 2019 protocol speaker ID",
                         group_confidence="high", speaker=r.speaker, target_voice="",
                         generator=r.system if is_ai else "human", official_partition=r.partition))
    return rows


def _apply_metadata_overrides(df: pd.DataFrame) -> pd.DataFrame:
    meta_path = DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        return df
    meta = pd.read_csv(meta_path, dtype=str)
    if "path" not in meta.columns:
        raise ValueError("data/metadata.csv must have a 'path' column")
    meta["path"] = meta["path"].str.replace("\\", "/", regex=False)
    df = df.set_index("path")
    for col in ("group", "speaker", "generator", "target_voice"):
        if col in meta.columns:
            upd = meta.set_index("path")[col].dropna()
            hit = upd.index.intersection(df.index)
            df.loc[hit, col] = upd.loc[hit]
            if col == "group":
                df.loc[hit, "group_basis"] = "data/metadata.csv"
                df.loc[hit, "group_confidence"] = "metadata"
    return df.reset_index()


def assign_partitions(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Final untouched test set, chosen by GROUP before any modelling.
    user AI: TEST_AI_SOURCES whole source speakers; user human: whole run-groups (~25% of clips);
    ASVspoof: official eval partition (unseen speakers and unseen attack systems)."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    df["partition"] = "dev"
    ai_groups = sorted(df.loc[(df.origin == "user") & (df.label == 1), "group"].unique())
    if ai_groups:
        test_ai = set(rng.choice(ai_groups, size=min(TEST_AI_SOURCES, len(ai_groups)), replace=False))
        df.loc[(df.origin == "user") & df.group.isin(test_ai), "partition"] = "test"
    hum = df[(df.origin == "user") & (df.label == 0)]
    if len(hum):
        sizes = hum.groupby("group").size()
        order = list(rng.permutation(sorted(sizes.index)))
        chosen, total = [], 0
        for g in order:
            if total >= TEST_HUMAN_FRACTION * len(hum):
                break
            chosen.append(g)
            total += sizes[g]
        df.loc[(df.origin == "user") & df.group.isin(chosen), "partition"] = "test"
    df.loc[df.official_partition == "eval", "partition"] = "test"
    return df


def build_manifest(include_supplementary: bool = True, compute_hashes: bool = True) -> pd.DataFrame:
    rows = _user_rows() + (_asvspoof_rows() if include_supplementary else [])
    if not rows:
        raise FileNotFoundError(f"No audio found under {DATA_DIR}")
    df = pd.DataFrame(rows)
    df = _apply_metadata_overrides(df)
    df["ext"] = df.path.str.extract(r"(\.[^.]+)$", expand=False).str.lower()
    df["size_bytes"] = [(ROOT / p).stat().st_size for p in df.path]
    if compute_hashes:
        df["sha256"] = [sha256_file(ROOT / p) for p in df.path]
    df = assign_partitions(df)
    return df.sort_values("file_id").reset_index(drop=True)


def manifest_hash(df: pd.DataFrame) -> str:
    cols = ["file_id", "sha256", "label", "group", "partition", "source"]
    payload = "\n".join(",".join(str(v) for v in row) for row in df[cols].sort_values("file_id").itertuples(index=False))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def save_manifest(df: pd.DataFrame, path: Path = MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"official_partition": str, "target_voice": str})
    df["official_partition"] = df["official_partition"].fillna("")
    df["target_voice"] = df["target_voice"].fillna("")
    return df
