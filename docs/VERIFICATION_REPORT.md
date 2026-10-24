# Verification report

Environment: Windows 11 Home 10.0.26200, Python 3.12.8, project venv `.venv` (versions in
`requirements.txt`), Google Chrome for browser checks. Date: 2026-10-24.

Status legend: **PASS** = verified by the command shown; **FAIL** = verified broken; **NOT VERIFIED** =
could not be checked here (reason given).

## G1 Environment ok, dependencies install from requirements.txt — PASS

* Objective: a fresh environment installs from `requirements.txt` alone and everything imports.
* Command: `python -m venv <scratch>/venv_g1 && <venv>/python -m pip install -r requirements.txt && pip install -e . --no-deps && pip check && python -c "import ...all modules..."`
* Expected: no errors, `pip check` clean, MP3 support available without ffmpeg.
* Actual: installed in 96 s; "No broken requirements found."; all imports OK; libsndfile 1.2.2 with MP3 = True;
  numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.0, librosa 1.0.0, soundfile 0.14.0, pandas 3.1.0, streamlit 1.66.1, plotly 7.1.0.
* Limitation: tested on Windows / Python 3.12 only. ffmpeg is not installed and not required.

## G2 Dataset audit produced and reviewed — PASS

* Command: `python -m voicedetect.pipeline manifest audit`
* Expected: `reports/dataset_audit.{json,csv}` with formats, rates, channels, durations, corrupt files,
  duplicates, near-silent / short clips, inferred groups and an adequacy verdict.
* Actual: 2,126 files audited in 75 s; 0 unreadable; 0 exact duplicates (SHA-256); 94 rejected by the
  loader (91 too short, 3 silent); groups: 8 voice-conversion sources (identical durations within each
  source), 56 Common Voice ID runs (nearest-neighbour coherence 0.87 vs chance 0.085), 107 ASVspoof
  speakers. User data alone fails 3 adequacy checks. Reviewed: findings in `docs/DATA_SOURCES.md`.
* Limitation: Common Voice speaker identity is inferred, not known.

## G3 Feature extraction ok, parity tests pass — PASS

* Command: `python -m voicedetect.pipeline features` and `python -m pytest tests/test_features.py tests/test_parity.py`
* Expected: fixed length (80 / 210), finite, deterministic; training path == prediction path; baseline
  identical to the original feature code.
* Actual: 2,032 files -> 2,564 windows (x2 variants) in 98 s; skipped files logged in
  `reports/feature_skips.csv`. Baseline features equal the original `extract_features` output
  **bit-for-bit** (max abs difference 0.0). Training-path features == `predict_file` features, and
  `predict_file` == `predict_bytes` == `predict_array` probabilities on WAV / stereo / FLAC / MP3
  fixtures. Re-running the build hits the cache with identical arrays.

## G4 No-leakage split tests pass — PASS

* Command: `python -m pytest tests/test_splits.py`
* Expected: no group, file hash or file id on both sides of any split; augmented copies never evaluated;
  real manifest partitions group-disjoint.
* Actual: all pass, including the checks on the real feature table for all 5 outer folds of all three
  settings plus dev-vs-test. `assert_no_leakage` also runs inside every outer fold of the pipeline
  (it would abort the run).
* Measured effect of the original leakage: 11/11 AI test files of the original split shared their
  source recording with training files (`reports/legacy_reproduction.json`).

## G5 Model comparison, tuning, metrics produced from real runs — PASS

* Command: `python -m voicedetect.pipeline legacy compare select`
* Expected: 3 settings x 2 feature sets x 7 models under identical group folds; nested tuning; mean ± std,
  paired comparisons, RBF heatmap.
* Actual: completed in 1,837 s; outputs `reports/model_comparison.csv`, `cv_fold_metrics.csv`,
  `oof_predictions.parquet`, `rbf_svm_grid.csv`, `pooled_auc_ci.csv`, `paired_comparisons.csv`,
  `selection/model_selection.json`. The "too good" user-data results (ROC-AUC ≈ 1.0) were investigated
  (`shortcut` step, `reports/shortcut_summary.json`) and are reported as a dataset-specific effect, not
  as the headline.

## G6 Calibration + threshold documented — PASS

* Command: `python -m voicedetect.pipeline calibrate`
* Expected: reliability analysis, ECE, a calibration method chosen on validation data, a cost-based
  threshold with explicit costs, and an inconclusive band, for each model.
* Actual: `reports/selection/calibration.json` (deployed) and `calibration_primary.json` (pre-registered):
  none / sigmoid / isotonic compared by outer-fold log loss; sigmoid chosen for both. Thresholds 0.64
  (deployed) and 0.66 (pre-registered) minimise validation cost with false alarm = 2, miss = 1; band
  ± 0.15. Reliability tables in `reliability_oof*.csv`, threshold trade-off tables in
  `threshold_tradeoff_oof*.csv`; cost rationale in the JSON and on the Evaluation page.
* Defect found and fixed during this gate: the first implementation averaged inner-fold calibrated
  models, which lowered validation AUC (0.966 -> 0.957). It was replaced by one monotone mapping on
  inner out-of-fold scores. Regression tests in `tests/test_tuning.py` show ranking is unchanged, with
  and without augmented training rows.

## G8 Prediction API passes validation tests (file and in-memory audio) — PASS

* Command: `python -m pytest tests/test_predict_api.py tests/test_parity.py`
* Expected: contract keys; long files averaged over windows; coded errors for missing / unsupported /
  empty / corrupt / silent / too short / too large / too long input; `predict_file` == `predict_bytes`
  == `predict_array`; model save/load round trip; refusal to load a model whose feature signature
  differs; uploads decoded in memory with no files written.
* Actual: all pass. Also checked by hand with the real model: `scripts/predict.py` on a held-out
  Common Voice clip -> HUMAN (P(AI) 1.8 %); on two held-out 10-minute AI files -> AI-GENERATED (96.0 %,
  84.8 %); `README.md` -> "Unsupported file type", exit 2; a missing file -> "The file does not exist",
  exit 2. (These hand checks ran on the pre-registered model; G13 below covers the final models.)

## G7 Robustness + error analysis produced — PASS

* Command: `python -m voicedetect.pipeline robustness errors`
* Expected: per-condition metrics for noise at several SNRs, MP3 at several bitrates, resampling,
  truncation to 1 / 2 / 4 s, volume, low-pass, reverb and a microphone-like chain; misclassified files
  with probabilities; per-generator error summary.
* Actual: `reports/robustness.csv` (deployed) and `robustness_primary.csv` (pre-registered): 22 conditions
  x 4 scopes on 884 test windows. **MP3 re-encoding is VERIFIED** (libsndfile's built-in encoder, 128 / 64 /
  48 / 32 kbps; no ffmpeg needed). `reports/errors_{test,oof}_{windows,files}.csv` and
  `error_summary_{test,oof}.csv`, with generator types parsed from ASVspoof's own README.
* Main findings (from those files): added noise or the microphone-like chain makes both models call
  nearly everything AI (human specificity 0.00-0.13). MP3 augmentation improved MP3 robustness. The
  neural-waveform and waveform-concatenation systems are missed most often.

## G9 Full pytest suite passes after a clean run — PASS

* Command: `python -m pytest -p no:cacheprovider` (no cache, full suite including real-data, model,
  app and README-consistency tests)
* Expected: all tests pass and none are skipped.
* Actual: **102 passed** in 62 s, 0 skipped, 0 failed. Coverage: audio edge cases, features, train /
  predict parity, file / bytes / array parity, leakage guards (synthetic and real splits), metrics vs
  hand-computed values, threshold and inconclusive logic, calibration keeps ranking (with and without
  augmentation), model round trip and signature refusal, manifest incl. a linked data folder,
  transforms, selection rules re-applied to the saved artifacts, headless app tests for all 9 pages,
  upload / microphone / corrupt recording / threshold slider / storage off, and README == artifacts.
* Defects caught by tests while building: an over-loose assertion in the app test (the CSS also contains
  the verdict class name), corrected to a stricter one; a floating-point edge in a hand-computed band test.

## G10 App starts; all pages load; controls work; screenshots inspected — PASS

* Command: `python scripts/e2e_check.py` (starts Streamlit, drives the installed Google Chrome through
  Playwright 1.64.1)
* Expected: 9 pages x 3 widths (1440 / 768 / 390 px) render without exceptions, error states or
  horizontal overflow; every control works; downloads work; no external requests.
* Actual: **88 / 88 checks passed** (`reports/e2e_results.json`, 40 screenshots in `reports/screenshots/`).
  Controls exercised: partition filter, example selectboxes with audio players, MFCC slider (keyboard),
  projection switch, setting / feature-set switches, expander, CSV / HTML downloads, threshold slider
  (value checked against an independent recomputation from the saved predictions), test / scope
  switches, PR tab, robustness model toggle, scope select, pills, 4 tabs, error-table row selection,
  which plays that window. The Overview tiles were checked against the artifacts.
* Screenshots inspected by eye. Defects found and fixed before the final run: light charts on the dark
  theme (theme detection unreliable on first load, so charts now always use the dark palette); a
  `[theme.light]` section that made the app follow the OS instead of defaulting to dark; the sidebar
  covering content at 390 px (now `auto`); cramped side-by-side model chart (now stacked); clipped chart
  labels at phone width (short labels, details in tooltips); overlapping labels in the trade-off chart;
  an invisible zero bar (value labels added); a truncated metric label; messy log-axis ticks on the
  spectrogram; screenshots and the overflow check that only covered the first viewport (now the whole
  inner scroll container).
* Console: 68 "404" messages are Streamlit's own base-path probe when a page is opened by deep link
  (`/<page>/_stcore/health`, `/<page>/_stcore/host-config`; recorded URLs in `e2e_results.json`), after
  which it uses `/_stcore/...`. 0 page errors. 4 aborted `blob:` requests are audio players replaced on
  rerun. **0 external requests**: the favicon was changed from a Material icon (fetched from
  fonts.gstatic.com) to a local PNG.

## G11 Upload classification works end to end in the browser — PASS

* Actual (Chrome, `reports/screenshots/live_upload_*`): held-out Common Voice MP3 -> "Human", P(AI) 2.0 %;
  15 s excerpt of a held-out 10-minute AI file (stereo WAV) -> "AI-generated", 94.7 %, 4 windows
  averaged. Rejections: renamed text file `.wav` -> "couldn't be decoded"; silent WAV -> "silent or almost
  silent"; 0.5 s clip -> "too short"; `.txt` -> the uploader's own "not allowed"; 107 MB WAV -> the app's
  "larger than 50 MB" message. (Before the fix, the widget's 50 MB limit only showed a red icon; the
  widget limit is now 200 MB and the app enforces 50 MB with text.)

## G12 Microphone recording works end to end — PASS (fake media device), with a manual-test note

* Method: Chrome launched with `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream
  --use-file-for-fake-audio-capture=<wav>`, microphone permission granted, then Record -> 7 s -> Stop
  -> Play control present -> **Classify recording** -> verdict -> Clear recording -> recorder ready again.
  Run twice: a held-out human clip (-> "Human", 9.0 %) and a held-out AI excerpt (-> "AI-generated", 98.5 %).
* Permission denied: a second browser without permission shows Streamlit's "This app would like to use
  your microphone. Learn how to allow access." plus the app's help expander.
* Limitation: the fake device feeds a clean digital file, not a real microphone in a room. A real laptop
  microphone was **not** tested here. Given the measured noise / microphone sensitivity, a real voice
  may well be flagged as AI. To test by hand: run `streamlit run app/streamlit_app.py`, open
  `http://localhost:8501` in Chrome or Edge, *Live prediction* -> *Record with microphone*, allow the
  microphone, speak 5-8 s, stop, play back, *Classify recording*. Repeat after denying permission in the
  site settings to see the denied state.

## G13 Original scripts/train.py and scripts/predict.py still run — PASS

* `python scripts/train.py --refit` -> exit 0 in 6.5 min; prints the model, version and test metrics;
  `reports/test_metrics.json` byte-identical before and after (determinism).
* `python scripts/predict.py <file>` with the final model -> held-out human MP3 "HUMAN" (2.0 %); held-out AI
  files "AI-GENERATED" (95.5 %) and "INCONCLUSIVE" (54.5 %, `taylor-to-linus.wav`); `README.md` ->
  "Unsupported file type", exit 2; the interactive prompt (path on stdin) works; a pasted path with a
  trailing CR now works (fixed during this gate).
* `python scripts/train.py` (full) runs the same steps as `python -m voicedetect.pipeline all` (see G14).
* The original scripts' Windows Unicode crash (defect D1) is fixed: the wrappers reconfigure stdout to UTF-8.

## G14 README commands reproduce results from scratch — PASS

* Method: a separate copy of the project (code only; the user's `data/human` and `data/ai` linked
  read-only through directory junctions; **no** manifest, features, models, reports or supplementary
  data), with the fresh G1 environment installed from `requirements.txt`. Then exactly the README commands:
  `python scripts/fetch_supplementary_asvspoof.py` and `python -m voicedetect.pipeline all`, followed by
  `python -m pytest`.
* Expected: identical results to the main run.
* Actual: **identical**. Fresh download: 1,465 audio files byte-identical, same selection, licence files and
  LA README. Manifest identical. All 17 JSON artifacts identical (selection, augmentation, robustness
  study, both calibrations, test metrics, legacy reproduction, audit, shortcut summary). All 42 CSV /
  parquet artifacts identical with max |numeric difference| = 0.0, including t-SNE coordinates and
  permutation importance. All three models' metadata identical (same version strings). Both generated
  README blocks identical. In the reproduced copy, pytest -> 102 passed.
* Defects found by this gate and fixed (with regression tests):
  1. `manifest._rel` resolved symlinks / junctions, so a `data/` folder linked from another drive crashed
     the first step -> now project-relative without resolving (`test_paths_through_a_linked_data_folder...`).
  2. The fetch script did not save `README.LA.txt` (generator descriptions); only that metadata column
     differed -> the script now saves it; re-fetched in both copies (byte-identical).
* Interruptions (environmental, not code): the Claude Code session restarted mid-run. Its orphaned
  processes could not start new worker pools (`BrokenProcessPool` / `WinError 6: The handle is invalid`)
  until relaunched with stdin from `/dev/null`. The machine was also suspended for about 2 h. The
  pipeline is step-wise (each step reads saved artifacts), so the run was resumed from the failed step
  in a fresh process. Permutation importance now runs without a process pool (results are unchanged:
  seeds are drawn before parallelisation).
* Run time (measured, uncontended): about 3.3 h for `pipeline all` plus about 4 min download, on an
  i9-13900H with 15.6 GB RAM.

## G15 Final accuracy audit: numbers in README / UI match saved artifacts — PASS

* README: the data table and the whole results section are generated from artifacts
  (`python -m voicedetect.pipeline readme`), and `tests/test_docs_numbers.py` fails if they differ. It
  passes. The narrative sections contain no hand-typed metrics.
* UI: every page reads `reports/` and `models/` at runtime. The e2e run compared the Overview tiles with
  `test_metrics.json` / `dataset_audit.json` / `feature_summary.json`, and the threshold-slider recall with
  an independent recomputation (all equal).
* HTML report: built only from artifacts, with no external references.
* Selection decisions re-derived from the saved artifacts by `tests/test_selection_rules.py` (same model).


## Summary

| Gate | Status | Gate | Status |
|---|---|---|---|
| G1 environment / requirements | PASS | G9 full pytest (102 tests) | PASS |
| G2 dataset audit | PASS | G10 app, pages, controls, screenshots | PASS |
| G3 features + parity | PASS | G11 upload end to end | PASS |
| G4 no-leakage splits | PASS | G12 microphone end to end | PASS with fake media device; **real microphone NOT VERIFIED** |
| G5 model comparison from real runs | PASS | G13 original scripts | PASS |
| G6 calibration + threshold | PASS | G14 reproduce from scratch | PASS (identical) |
| G7 robustness + errors (MP3 verified) | PASS | G15 numbers match artifacts | PASS |
| G8 prediction API | PASS | | |
