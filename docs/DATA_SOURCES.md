# Data sources

Every clip used anywhere in this project appears in `data/manifest.csv` with its `source`, `class_name`,
`group`, `origin` (`user` / `public` / `self_generated`), `generator`, `partition` and SHA-256 hash.
Nothing in `data/human` or `data/ai` was modified. Derived audio (MP3 re-encodings, perturbations) is
computed in memory and never written over the originals.

## 1. User data (primary)

| Folder | Files | Format | Content | Grouping used |
|---|---|---|---|---|
| `data/human/` | 605 | MP3, 32 kHz mono, exactly 48 kbps CBR, 1.3-15.2 s | Mozilla Common Voice (English) clips | runs of near-consecutive clip IDs (gap ≤ 5) → 56 groups, confidence **low-medium** |
| `data/ai/` | 56 | WAV PCM-16 stereo, 44.1 kHz (52) / 40 kHz (4), 79-600 s | voice conversions named `<source>-to-<target>` for 8 public figures | source speaker → 8 groups, confidence **high** |

Evidence and caveats:
* All 7 conversions of a source speaker have **identical durations** (600.00, 570.36, 93.69, 79.49 s),
  so they re-voice one source recording. The 56 files therefore contain only **8 independent source
  recordings**, all apparently from **one** conversion tool (unknown).
* Common Voice has no speaker IDs in the supplied files. It records sentences in batches of 5, and runs
  of exactly 5 consecutive IDs dominate. An acoustic check supports the grouping: 87 % of clips have
  their nearest cepstral neighbour in the same inferred group, against 8.5 % by chance (`reports/dataset_audit.json`).
  One speaker may still span several runs.
* Loader rejections: 3 Common Voice clips are silent and 2 have < 1 s of sound (`reports/feature_skips.csv`).
* Licence / terms: Mozilla describes Common Voice as a platform "to create public domain datasets"
  (README of github.com/common-voice/common-voice, checked 2026-10-24). The release version and the
  download terms you accepted are not recorded in the supplied files. The AI files' tool, source
  recordings and licence are unknown. They imitate real public figures, so keep them private and use
  them for detection research only.

## 2. Why the user data alone is not adequate

`reports/dataset_audit.json` → `adequacy.user_only` fails three checks:

1. **Independent AI units: 8** (< ~100). Windows or files from one source recording are not independent.
2. **AI generators: 1.** Generalisation to other generators cannot be measured.
3. **File-level class ratio 605 : 56.** Accuracy is dominated by the majority class (91.7 % for "always human").

In addition, the file format is perfectly confounded with the label (every human clip is a 48 kbps MP3,
every AI clip uncompressed WAV), as are the recording conditions (crowd-sourced microphones vs
broadcast-style source speech).

## 3. Search for supplementary data (what was reachable)

| Host | Reachable from this machine? | Outcome |
|---|---|---|
| Hugging Face (`huggingface.co`) | **No**: requests timed out (2026-10-24) | could not inspect or download any dataset or TTS model hosted there |
| Zenodo (`zenodo.org`) | **No**: timed out | could not check e.g. WaveFake |
| Edinburgh DataShare (`datashare.ed.ac.uk`) | Yes | **ASVspoof 2019** (doi:10.7488/ds/2555) found; licence verified; used |
| OpenSLR (`openslr.org`) | Yes | human-only corpora; not needed because ASVspoof provides bona fide speech recorded under the same conditions as its spoofs |
| GitHub | Yes | used only to verify licence statements |
| PyPI | Yes | packages only |

Self-generated TTS was considered. The only offline TTS on this machine is Windows SAPI (Microsoft
David / Zira), which is not open source and is far older than any modern generator. Because a
documented, licensed, multi-generator public corpus was reachable, no self-generated data was used
(`origin = self_generated` count: 0).

## 4. ASVspoof 2019 Logical Access subset (supplementary, `origin = public`)

* **What:** bona fide speech derived from the VCTK corpus (107 speakers), plus spoofed speech from 19
  TTS / voice-conversion systems (A01-A06 in train/dev, A07-A19 in eval). Official partitions are
  speaker-disjoint, and this was re-checked on the downloaded protocols: 20 / 20 / 67 speakers with zero overlap.
* **Licence:** Open Data Commons Attribution License (ODC-By) v1.0. Verified from the dataset's own
  `README.txt` ("You are free to use this database under Open Data Commons Attribution License (ODC-By)")
  and `LICENSE_text.txt`. Both are saved in `data/supplementary/asvspoof2019_la/` for attribution.
  **Citation:** M. Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection",
  Proc. Interspeech 2019. Data: doi:10.7488/ds/2555.
* **How it was obtained:** the full `LA.zip` is 7.64 GB (about 1.6 h at the measured ~1.2 MB/s), so
  `scripts/fetch_supplementary_asvspoof.py` reads the zip's central directory with HTTP range requests
  and downloads only the selected members, verifying each CRC32. 88 MB were downloaded (2026-10-24).
  It also saves the archive's `LA/README.LA.txt`, whose system table (A01-A19: TTS / VC and method) is
  parsed to label generators in the error and generalisation analyses. A from-scratch re-download
  produced byte-identical audio files and selection (see VERIFICATION_REPORT G14).
* **Selection (seed 42, `PROVENANCE.json`, `selection.csv`):**

| Official partition | Bona fide | Spoofed | Used in this project as |
|---|---|---|---|
| train (20 speakers, A01-A06) | 12 per speaker = 240 | 40 per system = 240 | development |
| dev (20 speakers, A01-A06) | 12 per speaker = 240 | 20 per system = 120 | development |
| eval (67 speakers, A07-A19) | 300 spread over speakers | 25 per system = 325 | **final test only** |

* **Format:** FLAC, 16 kHz mono, median 3.2 s. 89 clips (6 %) have < 1 s of sound after silence trimming
  and are skipped (human 61, AI 28). Spoofed clips have more speech than bona fide ones (mean 2.57 s vs
  2.03 s after trimming), consistent with the known silence/duration shortcut in ASVspoof 2019.
  Trimming leading and trailing silence before feature extraction is the guard against it.
* **Grouping:** official speaker ID; bona fide and spoofed speech of one speaker share a group.

## 5. How the sources are combined (never silently)

* Every result is reported for **user data only** and for **user + supplementary** (the
  `user` / `combined` settings in `reports/model_comparison.csv`, plus per-source rows in
  `reports/test_metrics.json`, `reports/robustness.csv` and `reports/error_summary_*.csv`).
* A third setting, **user data codec-matched**, re-encodes the user AI windows to the exact Common
  Voice format (32 kHz, 48 kbps CBR MP3) to test whether the file format alone explains the results.
* The final test partition was fixed by group before any modelling: 2 of the 8 user source speakers
  (seeded random choice: `biden`, `taylor`), about 25 % of Common Voice run-groups, and the ASVspoof
  eval partition.

## 6. Augmentation and perturbation variants

Every derived copy of an analysis window is computed in memory from the original audio and stored only as
features. It inherits its source window's file id and group.

| Variant | What it is | Used for |
|---|---|---|
| `mp3cv` | the window re-encoded as 32 kHz / 48 kbps CBR MP3 (the Common Voice format) | codec-matched control; training augmentation candidate; scoring validation windows |
| `aug_noise` | white or pink noise at a random SNR in 10-30 dB | training augmentation candidate |
| `aug_mic` | randomised microphone chain (low-pass 5-8 kHz, reverb RT60 0.2-0.6 s, pink noise 15-30 dB, 16 kHz, gain 0.3-1.0) | training augmentation candidate |
| `ev_white20`, `ev_room20`, `ev_mic` | fixed white / pink noise at 20 dB and the fixed microphone-like chain | **scoring** validation windows only (development data) |

Training augmentations are added to **training folds only** and never appear in a validation or test fold.
`assert_no_leakage` checks this on every split, and `tests/test_splits.py` covers it. Which augmentation
the deployed model uses was decided on development data (`reports/selection/augmentation.json` for the
first, MP3-only rule; `reports/selection/robustness_study.json` for the robustness-aware rule).
Augmented copies are never counted as independent data. The test-set robustness suite
(`src/voicedetect/robustness.py`) uses the same transform functions with different random seeds.
