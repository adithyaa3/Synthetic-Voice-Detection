# Implementation status

| Phase | Item | Status | Where |
|---|---|---|---|
| 1 | Environment inspected, original code analysed, original result reproduced and re-evaluated | done | `docs/ANALYSIS_OF_EXISTING_CODE.md`, `reports/legacy_reproduction.json` |
| 2 | Dataset audit (formats, durations, corrupt, duplicates, silence, short clips, groups, adequacy) | done | `src/voicedetect/data_audit.py`, `reports/dataset_audit.{json,csv}` |
| 2 | Supplementary data (ASVspoof 2019 LA subset, ODC-By) with provenance, licence files and selection list | done | `scripts/fetch_supplementary_asvspoof.py`, `data/supplementary/asvspoof2019_la/`, `docs/DATA_SOURCES.md` |
| 2 | Manifest: source / class / group / origin / generator / partition / SHA-256 for every clip | done | `src/voicedetect/manifest.py`, `data/manifest.csv` |
| 3 | Installable package, single config, thin wrappers for the original scripts | done | `src/voicedetect/`, `pyproject.toml`, `scripts/train.py`, `scripts/predict.py` |
| 4 | One shared feature function (baseline 80 = original, extended 210), cache keyed by file + config hash | done | `features.py`, `dataset.py` |
| 4 | Feature ablation (baseline vs extended in every setting; drop-one / only-one feature group) | done | `reports/model_comparison.csv`, `reports/feature_group_ablation.csv` |
| 5 | Group-aware nested CV, untouched test partition, leakage assertions in code and tests | done | `splits.py`, `evaluate.py`, `tests/test_splits.py` |
| 5 | Leave-one-source / target-voice / generator-out, cross-dataset | done (runs after selection) | `generalisation.py`, `reports/generalisation.csv`, `reports/cross_dataset.csv` |
| 6 | Model zoo (6 families + original RBF settings), inner-CV tuning, RBF C x gamma heatmap | done | `models.py`, `tuning.py`, `reports/rbf_svm_grid.csv` |
| 6 | Metrics incl. Brier, log loss, ECE, per-fold values, paired bootstrap CIs, Wilcoxon | done | `metrics.py`, `reports/cv_fold_metrics.csv`, `reports/paired_comparisons.csv` |
| 6 | Pre-declared selection rule with simplicity tie-break; user-domain check | done | `pipeline.py::select_in_setting`, `reports/selection/model_selection.json` |
| 6 | MP3 augmentation ablation (inside training folds only), decided on validation data | done (not adopted) | `reports/selection/augmentation.json` |
| 6 | Calibration (none / sigmoid / isotonic), cost-based threshold, inconclusive band | done | `calibration.py`, `reports/selection/calibration.json` |
| 6 | Permutation importance (features and groups), PCA / t-SNE, shortcut probe | done | `explain.py`, `reports/permutation_importance.csv`, `reports/shortcut_summary.json` |
| 6 | Error analysis, robustness suite (noise, MP3, resampling, truncation, volume, mic-like), learning curve | done | `errors.py`, `robustness.py`, `reports/` |
| 7 | Prediction API (`predict_file`, `predict_bytes`, `predict_array`), validation, metadata, mismatch refusal | done | `predict.py`, `tests/test_predict_api.py` |
| 8 | Streamlit app: 9 pages, upload + microphone (`st.audio_input`), dark theme, report download | done | `app/`, `.streamlit/config.toml` |
| 9 | pytest suite (synthetic fixtures + real-data and app tests) | done: 102 passed, 0 skipped | `tests/` |
| 9 | Browser verification with Playwright (all pages, 3 widths, controls, upload, microphone) | done: 88/88 checks (microphone via fake device; real microphone not tested) | `scripts/e2e_check.py`, `reports/e2e_results.json`, `reports/screenshots/` |
| 10 | README (results generated from artifacts), data sources, presentation guide, verification report | done | `README.md`, `docs/` |

## Decisions and deviations worth knowing

* **Analysis unit = 4 s window.** Long AI files contribute up to 10 windows, and every window keeps its
  file's group. File-level metrics (mean over windows, as in the live demo) are reported alongside.
* **Silence trimming** before windowing (the original used the first 4 s untrimmed). This also guards
  against ASVspoof 2019's known silence shortcut. The original pipeline is reproduced unchanged in the
  `legacy` step.
* **Calibration implementation:** the first calibration run averaged inner-fold models
  (`CalibratedClassifierCV(ensemble=True)`), which lowered validation AUC. It was replaced by a single
  monotone mapping on inner out-of-fold scores (`ensemble=False`, regression test
  `tests/test_tuning.py`), and calibration was re-selected. All of this used development data only.
  The SVM Platt scaling inside the model comparison keeps the ensemble form.
* **Speed-up without changing results:** calibration methods reuse the per-fold hyperparameters found by
  the identical, deterministic inner search (verified: identical fold AUCs).
* **Two models, in this order.** (1) The *pre-registered* model was chosen by rules fixed before any test
  evaluation: model comparison, MP3-only augmentation rule, calibration by log loss. It was evaluated on the
  test set once. (2) Its test-set robustness analysis showed severe noise / microphone sensitivity, so a
  **development-data-only robustness study** with a newly declared rule (`step_robust`) chose the training
  augmentation of the *deployed* model. The deployed model's test results are post-hoc and are labelled as
  such everywhere; the pre-registered model and its results are kept (`models/primary_model.joblib`,
  `reports/test_metrics.json["primary"]`, `reports/robustness_primary.csv`).
* **The robustness rule was applied as written** even though it excluded the most noise-robust variant
  (its clean-AUC cost exceeded the 0.02 tolerance). That trade-off is reported rather than re-tuned.
* **No self-generated data** was needed: a licensed multi-generator corpus was reachable.
