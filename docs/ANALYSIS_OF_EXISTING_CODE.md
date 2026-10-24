# Analysis of the existing code (Phase 1)

Written before any code was changed. Everything below was checked by running the original code
on 2026-10-24. The original `scripts/train.py` was run unmodified, from a scratch copy, so the
pickles in `models/` were not overwritten. Your original pickles are also backed up, byte for byte,
in `models/legacy_original/` (SHA-256 verified).

## 1. Environment found

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200, Intel i9-13900H (14 cores / 20 threads), 15.6 GB RAM, ~31 GB free disk |
| Python | 3.12.8 (global install; 3.13 also present) |
| Audio/ML packages (global) | numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.0, librosa 1.0.0, soundfile 0.14.0 (libsndfile 1.2.2), soxr 1.1.0, joblib 1.6.0, numba 0.67.0, pytest 7.4.3 |
| Missing | pandas, matplotlib, streamlit, playwright, **ffmpeg / ffprobe / lame / sox (none on PATH)** |
| MP3 decoding | Works without ffmpeg: libsndfile 1.2.2 reads (and writes) MP3 natively, and librosa 1.0 uses soundfile |
| Browsers | Microsoft Edge and Google Chrome installed (Playwright can drive them without downloading browsers) |
| Network | PyPI and GitHub reachable; Hugging Face and Zenodo **time out**; Edinburgh DataShare and OpenSLR reachable |
| Git | branch `main`, clean; `data/` and `models/*.pkl` are git-ignored |

## 2. What the existing code does

### `scripts/train.py`
1. Lists `data/human` (label 0) and `data/ai` (label 1). Any `.wav`/`.mp3` file is included.
2. **Feature function** `extract_features(path)`:
   `librosa.load(path, sr=22050, duration=4)` loads the first 4 s, mixed to mono and resampled to 22 050 Hz.
   `librosa.feature.mfcc(n_mfcc=40)` uses librosa defaults: n_fft=2048, hop=512, 128 mel bands.
   It then takes the mean and the standard deviation of each coefficient over time, giving **80 features**.
3. A single `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
4. `StandardScaler` is fit on the training split only. This part is correct.
5. `SVC(kernel="rbf", C=10, gamma="scale", probability=True)`.
6. Prints accuracy, a classification report and a confusion matrix.
7. Saves `models/voice_detector_svm.pkl` and `models/scaler.pkl` with joblib.

### `scripts/predict.py`
A *copy* of the same feature function. It loads both pickles, scales the features and calls
`predict()` and `predict_proba()`. It prints `HUMAN` / `AI-GENERATED` and a "confidence" equal to
`proba[predicted_class]`.

## 3. Reproduced result

`python scripts/train.py` (original code, with stdout in UTF-8, see defect D1):

```
Dataset shape: (661, 80)   Human: 605   AI: 56
Test Accuracy: 1.0000      (122 human + 11 AI test clips)
Confusion matrix [[122 0] [0 11]]
```

A classifier that always answers "human" would already score **122/133 = 91.7 %** on this test set.

## 4. Why 100 % is not trustworthy: measured evidence

| Check | Result |
|---|---|
| The 56 AI files are 8 source recordings × 7 target voices. All files that share a source (e.g. `biden-to-*`) have **identical durations** (600.00 s, 570.36 s, 79.49 s, 93.69 s). | They are the same source speech re-voiced into different targets. |
| The 11 AI test files in the original split, versus the source recordings in training | **11 / 11** test AI files have a sibling from the same source recording in training. This is direct leakage. |
| Same SVM, **leave-one-source-speaker-out** (8 folds, humans in 8 random groups) | accuracy 0.951 ± 0.029 (inflated by imbalance), **balanced accuracy 0.730 ± 0.166** |
| AI recall per held-out source speaker | biden 0.29, **linus 0.00**, margot 1.00, musk 0.71, obama 0.14, ryan 0.86, taylor 0.43, trump 0.29 |
| Single "energy above 8 kHz" feature (codec/bandwidth confound probe) | ROC-AUC 0.595, so bandwidth alone does not explain the separation |

Conclusion: the 100 % comes from the random split putting re-voiced copies of the same speech in both
train and test, plus a 91.7 % majority-class floor. Under a group-aware split, the original model
misses most AI clips from unseen source recordings. (The script for this diagnostic is reproduced
inside the new package as the "legacy reproduction" experiment.)

## 5. Defects and risks (concrete list)

| # | Defect / risk | Evidence / impact |
|---|---|---|
| D1 | **Crash on Windows when output is redirected.** `print("✓ …")` raises `UnicodeEncodeError` under cp1252. The `except` that catches it prints "✗ …", which crashes again. | Observed: `python scripts/train.py > log.txt` dies on the first file. |
| D2 | **Single random split with group leakage.** Re-voiced siblings of the same source recording land in train and test. | 11/11 test AI files leaked (section 4). |
| D3 | **No cross-validation.** One split, one number, no variance estimate. | With 11 AI test clips, one error moves recall by 9 points. |
| D4 | **Severe class imbalance** (605 human : 56 AI files), no class weighting, and **accuracy is the only headline metric**. | Majority-class floor is 91.7 %. Balanced accuracy, ROC-AUC, PR-AUC and recall are never computed. |
| D5 | **Hard-coded hyperparameters** (C=10, gamma="scale"). No search and no baselines (majority class, linear models). | There is no evidence that the RBF kernel is needed. |
| D6 | **Feature code duplicated** in train.py and predict.py. | A classic train/predict skew risk; any edit to one silently breaks the other. |
| D7 | **"First 4 seconds" policy.** Leading silence is not trimmed. For the 7-10 minute AI files only the first 4 s (0.7 % of the audio) is used. Clips shorter than 4 s get shorter feature windows, and nothing is padded or documented. | 40 human clips are < 4 s (2 are < 2 s). The 4.3 GB of AI audio contributes 56 × 4 s. |
| D8 | **No input validation.** Silent, corrupt, zero-length or very short files are not detected. Silence and degenerate input are passed straight to the model (behaviour checked in the test suite). | `predict.py` only checks `os.path.exists`. |
| D9 | **Stereo/mono and sample rate are handled implicitly** by `librosa.load` (mono mix plus resample). This is fine, but undocumented. | AI files are 44.1 kHz / 40 kHz stereo; human files are 32 kHz mono MP3. |
| D10 | **Format perfectly confounded with label.** Every human file is a 32 kHz mono MP3 (Common Voice, ~6 s); every AI file is a 40/44.1 kHz stereo PCM WAV (80-600 s). | A model can learn "MP3 vs WAV" instead of "human vs synthetic". This matters most for live microphone input, which arrives as uncompressed WAV. |
| D11 | **MP3 decoding dependency** is undocumented. It works here only because libsndfile ≥ 1.1 decodes MP3; there is no ffmpeg. | On older soundfile builds MP3 loading would fail or fall back to deprecated audioread. |
| D12 | **Fixed 0.5 decision threshold**, never chosen on validation data, and there is no "inconclusive" zone. | The costs of a false alarm (accusing a real person) and a miss are never discussed. |
| D13 | **`probability=True` cost, inconsistency and deprecation.** It runs an extra internal 5-fold CV at fit time (Platt scaling). Measured on the original 661 x 80 features: fit 6.2 ms without vs 30.1 ms with it (**4.8x slower**, median of 20). scikit-learn 1.9 also emits `FutureWarning: The probability parameter was deprecated in 1.9 and will be removed in version 1.11`, so the original model construction will stop working on a future scikit-learn; the rebuild calibrates through `CalibratedClassifierCV` instead. `predict()` and `argmax(predict_proba())` can disagree (documented sklearn behaviour), so the printed "confidence" can then be < 50 %. | Not observed on this dataset (0/661 disagreements), but a latent bug in predict.py's confidence logic. |
| D14 | **Pickles without metadata.** No feature config, library versions, seed, data hash or metrics are saved. | Your saved model was pickled with **scikit-learn 1.7.2**; loading it on 1.9.0 raises `InconsistentVersionWarning`. Nothing stops a feature-config mismatch from loading silently. |
| D15 | **No seeds recorded** apart from the split; no requirements file; no tests. | Not reproducible on another machine. |
| D16 | `predict.py` swallows all exceptions and returns `(None, None)`, with an `input()` fallback. | Errors are invisible to callers. |
| D17 | AI filenames use inconsistent case (`Obama-to-Biden`, `obama-to-linus`, `trump-to-Obama`). | Group inference must normalise case, or "Obama" and "obama" become different groups. |

## 6. What this means for the rebuild

* Validation must be **group-aware**. The unit of independence on the AI side is the *source
  recording* (8 groups), not the file. Target voice is a second grouping factor.
* The AI side is **inadequate as-is**: 8 independent source recordings, apparently one generator,
  and a format confound. See `docs/DATA_SOURCES.md` for the supplementary-data plan.
* The long AI files can legitimately yield **several analysis windows per file** (more training
  windows). They are *not* independent samples: every window inherits its file's group.
* The headline must be **balanced accuracy / ROC-AUC with mean ± std over group folds**, not accuracy
  on one split.
