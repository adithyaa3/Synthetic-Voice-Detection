"""Single inference API used by the CLI (scripts/predict.py) and the Streamlit app.

predict_file(path)                 -> dict
predict_bytes(data, filename)      -> dict   (uploads; decoded in memory, nothing written to disk)
predict_array(samples, sample_rate)-> dict   (microphone / in-memory audio)

All three go through audio_io (same loading/trim/window policy) and features.extract (same
feature function as training). The returned probability is P(AI-generated) averaged over the
analysed windows; it is a calibrated model score, not proof.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from . import audio_io
from .config import AUDIO, FINAL_METADATA_PATH, FINAL_MODEL_PATH, PREDICT, FeatureConfig, config_hash, feature_signature
from .features import extract_many


class ModelConfigMismatch(RuntimeError):
    """The saved model was trained with a feature configuration that differs from the code."""


class ModelNotFound(FileNotFoundError):
    pass


def library_versions() -> dict:
    import librosa
    import sklearn
    import soundfile
    return {"python": platform.python_version(), "numpy": np.__version__, "scikit-learn": sklearn.__version__,
            "librosa": librosa.__version__, "soundfile": soundfile.__version__,
            "libsndfile": soundfile.__libsndfile_version__}


def save_bundle(pipeline, feature_set: str, decision: dict, extra_metadata: dict,
                model_path: Path = FINAL_MODEL_PATH, metadata_path: Path = FINAL_METADATA_PATH) -> dict:
    fcfg = FeatureConfig(feature_set)
    sig_hash = config_hash(AUDIO, fcfg)
    created = datetime.now(timezone.utc)
    defining = {k: extra_metadata.get(k) for k in ("training_setting", "model_name", "best_params", "augmentation", "calibration")}
    defining["threshold"] = float(decision["threshold"])
    model_hash = hashlib.sha256(json.dumps(defining, sort_keys=True, default=str).encode()).hexdigest()
    meta = {
        "model_version": f"{created:%Y.%m.%d}-{sig_hash[:6]}-{model_hash[:6]}",
        "created_utc": created.isoformat(timespec="seconds"),
        "feature_set": feature_set,
        "feature_signature": feature_signature(AUDIO, fcfg),
        "feature_signature_hash": sig_hash,
        "threshold": float(decision["threshold"]),
        "inconclusive_halfwidth": float(decision["inconclusive_halfwidth"]),
        "library_versions": library_versions(),
        **extra_metadata,
    }
    meta_json = json.dumps(meta, indent=2, default=str)  # serialise first: never leave a model without metadata
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": meta}, model_path)
    metadata_path.write_text(meta_json, encoding="utf-8")
    return meta


class ModelBundle:
    def __init__(self, pipeline, metadata: dict):
        self.pipeline = pipeline
        self.metadata = metadata
        self.feature_set = metadata["feature_set"]
        self.feature_config = FeatureConfig(**metadata["feature_signature"]["features"])
        self.threshold = float(metadata["threshold"])
        self.halfwidth = float(metadata["inconclusive_halfwidth"])
        self.version = metadata["model_version"]


def load_bundle(path: Path = FINAL_MODEL_PATH) -> ModelBundle:
    path = Path(path)
    if not path.exists():
        raise ModelNotFound(f"No trained model at {path}. Run: python -m voicedetect.pipeline all")
    obj = joblib.load(path)
    meta = obj["metadata"]
    expected = config_hash(AUDIO, FeatureConfig(meta["feature_set"]))
    if meta.get("feature_signature_hash") != expected:
        raise ModelConfigMismatch(
            f"Model {path.name} was trained with feature signature {meta.get('feature_signature_hash')} but the "
            f"code now produces {expected}. Retrain the model (python -m voicedetect.pipeline all).")
    return ModelBundle(obj["pipeline"], meta)


_CACHED: dict = {}


def default_bundle() -> ModelBundle:
    if "bundle" not in _CACHED:
        _CACHED["bundle"] = load_bundle()
    return _CACHED["bundle"]


def decide(p: float, threshold: float, halfwidth: float) -> tuple[str, str, bool]:
    raw = "ai" if p >= threshold else "human"
    inconclusive = abs(p - threshold) < halfwidth
    return ("inconclusive" if inconclusive else raw), raw, inconclusive


def predict_loaded(la: audio_io.LoadedAudio, bundle: ModelBundle | None = None) -> dict:
    bundle = bundle or default_bundle()
    spans = audio_io.select_windows(la.samples, AUDIO)
    X = extract_many([la.samples[s:e] for s, e in spans], la.sample_rate, bundle.feature_config)
    p_w = bundle.pipeline.predict_proba(X)[:, 1]
    p = float(np.mean(p_w))
    label, raw, inconclusive = decide(p, bundle.threshold, bundle.halfwidth)
    warnings = list(la.warnings)
    if len(spans) > 1:
        warnings.append(f"Long audio: {len(spans)} windows of {AUDIO.window_seconds:.0f} s were analysed and their "
                        f"probabilities averaged (window range {p_w.min():.2f}-{p_w.max():.2f}).")
    return {
        "label": label, "raw_label": raw, "probability": p, "threshold": bundle.threshold,
        "inconclusive": inconclusive,
        "inconclusive_band": [max(0.0, bundle.threshold - bundle.halfwidth), min(1.0, bundle.threshold + bundle.halfwidth)],
        "model_version": bundle.version, "warnings": warnings,
        "n_windows": len(spans), "window_probabilities": [float(v) for v in p_w],
        "window_spans_s": [(s / la.sample_rate, e / la.sample_rate) for s, e in spans],
        "duration_s": la.original_duration_s, "analysed_duration_s": la.trimmed_duration_s,
        "input_sample_rate": la.original_sample_rate, "input_channels": la.original_channels,
    }


def predict_file(path, bundle: ModelBundle | None = None) -> dict:
    p = audio_io.check_path(path)
    out = predict_loaded(audio_io.load(str(p), AUDIO, PREDICT.max_duration_s), bundle)
    out["file"] = p.name
    return out


def predict_bytes(data: bytes, filename: str, bundle: ModelBundle | None = None,
                  max_mb: float = PREDICT.max_file_mb) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in PREDICT.allowed_extensions:
        raise audio_io.AudioError("unsupported_format", f"'{suffix or filename}' (allowed: {', '.join(PREDICT.allowed_extensions)})")
    if len(data) / 1e6 > max_mb:
        raise audio_io.AudioError("too_large", f"({len(data) / 1e6:.1f} MB > {max_mb:.0f} MB)")
    out = predict_loaded(audio_io.load_bytes(data, AUDIO, PREDICT.max_duration_s), bundle)
    out["file"] = Path(filename).name
    return out


def predict_array(samples: np.ndarray, sample_rate: int, bundle: ModelBundle | None = None) -> dict:
    return predict_loaded(audio_io.load_array(samples, sample_rate, AUDIO), bundle)


def describe_config() -> dict:
    return {"audio": asdict(AUDIO), "predict": asdict(PREDICT)}
