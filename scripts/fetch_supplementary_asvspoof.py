"""Fetch a small, seeded, stratified subset of ASVspoof 2019 LA as supplementary data.

Source : Edinburgh DataShare, https://doi.org/10.7488/ds/2555 (LA.zip, 7.12 GB)
Licence: Open Data Commons Attribution License (ODC-By) v1.0, verified from the dataset's
         README.txt / LICENSE_text.txt, which are saved next to the downloaded files.

The full archive is too large for this project (~1.6 h at the measured ~1.2 MB/s), so this script
reads the zip's central directory with HTTP range requests and downloads only the selected
members, checking each CRC32. Nothing in data/human or data/ai is touched.

Usage:  python scripts/fetch_supplementary_asvspoof.py [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import time
import urllib.request
import zipfile
import zlib
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "supplementary" / "asvspoof2019_la"
LA_ZIP_URL = "https://datashare.ed.ac.uk/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/download"
README_URL = "https://datashare.ed.ac.uk/bitstreams/765597d6-633f-418a-96b0-39274a8d56c9/download"
LICENSE_URL = "https://datashare.ed.ac.uk/bitstreams/60c7de6d-37d3-45c1-b52e-9f18b0338e47/download"
SEED = 42

# How many files to take from each official partition (see docs/DATA_SOURCES.md).
PLAN = {
    "train": {"bonafide_per_speaker": 12, "spoof_per_system": 40},
    "dev": {"bonafide_per_speaker": 12, "spoof_per_system": 20},
    "eval": {"bonafide_total": 300, "spoof_per_system": 25},
}
USER_AGENT = "voicedetect-research-fetch/1.0 (academic project)"


def _request(url: str, start: int | None = None, end: int | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
                if start is not None and resp.status != 206:
                    raise IOError(f"server ignored Range header (status {resp.status})")
                return data
        except Exception as exc:  # network hiccup: retry with backoff, then give up loudly
            if attempt == 4:
                raise
            print(f"  retry {attempt + 1} after error: {exc}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def resolve(url: str) -> tuple[str, int]:
    """Follow redirects once and return (final_url, total_size)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        final = resp.geturl()
        crange = resp.headers.get("Content-Range", "")
    total = int(crange.split("/")[-1])
    return final, total


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests, with a small block cache."""

    BLOCK = 1 << 20

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()
        self.bytes_fetched = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def _block(self, idx: int) -> bytes:
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        start = idx * self.BLOCK
        end = min(self.size, start + self.BLOCK) - 1
        data = _request(self.url, start, end)
        self.bytes_fetched += len(data)
        self.cache[idx] = data
        if len(self.cache) > 64:
            self.cache.popitem(last=False)
        return data

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        out, pos, remaining = bytearray(), self.pos, n
        while remaining:
            idx, off = divmod(pos, self.BLOCK)
            chunk = self._block(idx)[off: off + remaining]
            out += chunk
            pos += len(chunk)
            remaining -= len(chunk)
        b[:n] = out
        self.pos += n
        return n


def fetch_member(url: str, info: zipfile.ZipInfo) -> bytes:
    """Download one member via its local header and verify its CRC32."""
    name_len = len(info.filename.encode("utf-8"))
    span = 30 + name_len + 512 + info.compress_size
    raw = _request(url, info.header_offset, info.header_offset + span - 1)
    sig, *_ = struct.unpack("<I", raw[:4])
    if sig != 0x04034B50:
        raise IOError(f"bad local header for {info.filename}")
    n, m = struct.unpack("<HH", raw[26:30])
    data = raw[30 + n + m: 30 + n + m + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        payload = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(data, -15)
    else:
        raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return payload


def parse_protocol(text: str) -> list[dict]:
    rows = []
    for line in text.strip().splitlines():
        spk, utt, _, system, key = line.split()
        rows.append({"speaker": spk, "utt": utt, "system": system if system != "-" else "bonafide", "key": key})
    return rows


def select(rows: list[dict], plan: dict, rng: np.random.Generator) -> list[dict]:
    bona = [r for r in rows if r["key"] == "bonafide"]
    spoof = [r for r in rows if r["key"] == "spoof"]
    chosen = []
    by_spk = defaultdict(list)
    for r in bona:
        by_spk[r["speaker"]].append(r)
    if "bonafide_per_speaker" in plan:
        for spk in sorted(by_spk):
            pool = sorted(by_spk[spk], key=lambda r: r["utt"])
            k = min(plan["bonafide_per_speaker"], len(pool))
            chosen += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
    else:  # spread a total budget evenly over speakers
        spks = sorted(by_spk)
        per = int(np.ceil(plan["bonafide_total"] / len(spks)))
        picked = []
        for spk in spks:
            pool = sorted(by_spk[spk], key=lambda r: r["utt"])
            k = min(per, len(pool))
            picked += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
        idx = sorted(rng.choice(len(picked), min(plan["bonafide_total"], len(picked)), replace=False))
        chosen += [picked[i] for i in idx]
    by_sys = defaultdict(list)
    for r in spoof:
        by_sys[r["system"]].append(r)
    for system in sorted(by_sys):
        pool = sorted(by_sys[system], key=lambda r: r["utt"])
        k = min(plan["spoof_per_system"], len(pool))
        chosen += [pool[i] for i in sorted(rng.choice(len(pool), k, replace=False))]
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="select files but do not download audio")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "flac").mkdir(exist_ok=True)
    print("Saving licence and README for attribution ...")
    (OUT_DIR / "LICENSE_text.txt").write_bytes(_request(LICENSE_URL))
    (OUT_DIR / "README_ASVspoof2019.txt").write_bytes(_request(README_URL))

    url, size = resolve(LA_ZIP_URL)
    print(f"LA.zip: {size / 1e9:.2f} GB at {url}")
    rf = HttpRangeFile(url, size)
    t0 = time.time()
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
    infos = {i.filename: i for i in zf.infolist()}
    print(f"Central directory: {len(infos)} entries, fetched {rf.bytes_fetched / 1e6:.1f} MB in {time.time() - t0:.0f}s")

    proto_names = {
        "train": next(n for n in infos if n.endswith("ASVspoof2019.LA.cm.train.trn.txt")),
        "dev": next(n for n in infos if n.endswith("ASVspoof2019.LA.cm.dev.trl.txt")),
        "eval": next(n for n in infos if n.endswith("ASVspoof2019.LA.cm.eval.trl.txt")),
    }
    # the LA README describes each attack system (A01-A19: TTS/VC and method); the pipeline uses it to label
    # generators, so it is part of the reproducible download (found missing by the from-scratch reproduction)
    la_readme = next(n for n in infos if n.endswith("README.LA.txt"))
    (OUT_DIR / "README.LA.txt").write_bytes(fetch_member(url, infos[la_readme]))
    rng = np.random.default_rng(SEED)
    selection = []
    (OUT_DIR / "protocols").mkdir(exist_ok=True)
    for part, pname in proto_names.items():
        text = fetch_member(url, infos[pname]).decode("utf-8")
        (OUT_DIR / "protocols" / Path(pname).name).write_text(text, encoding="utf-8")
        rows = parse_protocol(text)
        chosen = select(rows, PLAN[part], rng)
        for r in chosen:
            member = next((n for n in (f"LA/ASVspoof2019_LA_{part}/flac/{r['utt']}.flac",) if n in infos), None)
            if member is None:
                raise KeyError(f"{r['utt']} not found in archive for partition {part}")
            r.update({"partition": part, "member": member})
        selection += chosen
        n_spk = len({r["speaker"] for r in rows})
        print(f"  {part}: protocol rows {len(rows)} ({n_spk} speakers) -> selected {len(chosen)} "
              f"(bonafide {sum(r['key'] == 'bonafide' for r in chosen)}, spoof {sum(r['key'] == 'spoof' for r in chosen)})")

    total_bytes = sum(infos[r["member"]].compress_size for r in selection)
    print(f"Selected {len(selection)} files, {total_bytes / 1e6:.1f} MB to download")
    if args.dry_run:
        return

    todo = [r for r in selection if not (OUT_DIR / "flac" / f"{r['utt']}.flac").exists()]
    print(f"Downloading {len(todo)} files (already present: {len(selection) - len(todo)}) ...")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_member, url, infos[r["member"]]): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            (OUT_DIR / "flac" / f"{r['utt']}.flac").write_bytes(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} ({time.time() - t0:.0f}s)")

    with open(OUT_DIR / "selection.csv", "w", encoding="utf-8") as fh:
        fh.write("utt,speaker,system,key,partition,member\n")
        for r in selection:
            fh.write(f"{r['utt']},{r['speaker']},{r['system']},{r['key']},{r['partition']},{r['member']}\n")
    prov = {
        "dataset": "ASVspoof 2019 Logical Access (LA)",
        "doi": "https://doi.org/10.7488/ds/2555",
        "landing_page": "https://datashare.ed.ac.uk/handle/10283/3336",
        "archive_url": LA_ZIP_URL,
        "archive_size_bytes": size,
        "licence": "Open Data Commons Attribution License (ODC-By) v1.0",
        "licence_url": "https://opendatacommons.org/licenses/by/1.0/",
        "citation": "Todisco et al., 'ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection', Interspeech 2019",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_seed": SEED,
        "selection_plan": PLAN,
        "n_files": len(selection),
        "method": "HTTP range requests on the remote zip; CRC32 verified for every member",
    }
    (OUT_DIR / "PROVENANCE.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(f"Done in {time.time() - t0:.0f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
