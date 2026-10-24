"""Classify an audio file as human or AI-generated (thin wrapper over voicedetect.predict).

    python scripts/predict.py path/to/audio.wav
    python scripts/predict.py            # asks for a path, like the original script

Exit codes: 0 = classified, 2 = input rejected (with a reason), 3 = no trained model.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from voicedetect.audio_io import AudioError
    from voicedetect.predict import ModelConfigMismatch, ModelNotFound, predict_file
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from voicedetect.audio_io import AudioError
        from voicedetect.predict import ModelConfigMismatch, ModelNotFound, predict_file
    except ImportError as exc:
        sys.exit(f"Missing dependency ({exc}). Activate the project environment first:\n"
                 "  .venv\\Scripts\\activate   (Windows)   or   source .venv/bin/activate")

VERDICT = {"human": "HUMAN", "ai": "AI-GENERATED", "inconclusive": "INCONCLUSIVE"}


def _report(r: dict) -> tuple[str, float]:
    p = r["probability"]
    conf = p if r["label"] in ("ai", "inconclusive") else 1 - p
    print("\n----------------------------")
    print(f"RESULT: {VERDICT[r['label']]}")
    print(f"P(AI-generated): {p:.1%}   (decision threshold {r['threshold']:.2f}; "
          f"inconclusive band {r['inconclusive_band'][0]:.2f}-{r['inconclusive_band'][1]:.2f})")
    print(f"File: {r['file']}   windows analysed: {r['n_windows']}   model {r['model_version']}")
    for w in r["warnings"]:
        print(f"Note: {w}")
    print("This is a model probability, not proof.")
    print("----------------------------")
    return VERDICT[r["label"]], conf


def predict_audio(audio_path):
    """Backward-compatible helper: returns (result, confidence) like the original script, where
    confidence is the probability of the reported class (P(AI) when inconclusive); (None, None) on error."""
    try:
        return _report(predict_file(audio_path))
    except (AudioError, ModelNotFound, ModelConfigMismatch) as exc:
        print(f"Error processing audio: {exc}")
        return None, None


def main() -> int:
    print("=== SYNTHETIC VOICE DETECTOR - PREDICTION ===")
    raw = sys.argv[1] if len(sys.argv) > 1 else input("Enter audio file path (MP3/WAV): ")
    audio_file = raw.strip().strip('"')  # tolerate pasted quotes / stray whitespace (e.g. a trailing CR)
    try:
        r = predict_file(audio_file)
    except AudioError as exc:
        print(f"Input rejected: {exc}")
        return 2
    except (ModelNotFound, ModelConfigMismatch) as exc:
        print(str(exc))
        return 3
    _report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
