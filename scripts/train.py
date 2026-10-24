"""Train the synthetic voice detector (thin wrapper over the voicedetect package).

    python scripts/train.py            # full, reproducible pipeline (audit -> features -> nested CV model
                                       # comparison -> selection -> calibration -> final model -> test ->
                                       # robustness / generalisation / explainability -> report)
    python scripts/train.py --refit    # reuse the saved, validated model selection and calibration
                                       # decisions; only rebuild manifest/features and refit + test the model

The original single-split SVM script is preserved in git history and reproduced exactly by the
'legacy' step (reports/legacy_reproduction.json).
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # the original script crashed on Windows consoles without UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from voicedetect import pipeline
    from voicedetect.config import FINAL_METADATA_PATH, REPORTS_DIR
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from voicedetect import pipeline
        from voicedetect.config import FINAL_METADATA_PATH, REPORTS_DIR
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc}). Activate the project environment first:\n"
                 "  .venv\\Scripts\\activate   (Windows)   or   source .venv/bin/activate\n"
                 "  pip install -r requirements.txt && pip install -e .")

FULL = ["manifest", "audit", "features", "legacy", "compare", "select", "augment", "calibrate", "finalize",
        "test", "robustness", "generalisation", "explain", "errors", "learning", "figures", "report"]
REFIT = ["manifest", "features", "finalize", "test"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refit", action="store_true", help="refit with the saved selection instead of re-running it")
    args = ap.parse_args()
    print("=== SYNTHETIC VOICE DETECTOR - TRAINING ===")
    if args.refit and not (REPORTS_DIR / "selection" / "calibration.json").exists():
        print("No saved selection found; running the full pipeline instead.")
        args.refit = False
    pipeline.main(REFIT if args.refit else [s for s in FULL if s in pipeline.STEPS])
    meta = json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8"))
    test = json.loads((REPORTS_DIR / "test_metrics.json").read_text(encoding="utf-8"))["deployed"]["scopes"]
    allm = test["all test windows"]
    print("\n=== TRAINING COMPLETED ===")
    print(f"Model      : {meta['model_name']} on {meta['feature_set']} features ({meta['training_setting']} data)")
    print(f"Version    : {meta['model_version']}   threshold {meta['threshold']:.2f}")
    print(f"Test (held-out groups): ROC-AUC {allm['roc_auc']:.3f}, balanced accuracy {allm['balanced_accuracy']:.3f}, "
          f"AI recall {allm['recall']:.3f}, human specificity {allm['specificity']:.3f}")
    print("Saved      : models/final_model.joblib + models/metadata.json ; reports in reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
