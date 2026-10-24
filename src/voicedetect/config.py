"""All paths, seeds, audio and feature parameters live here (single source of truth)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
USER_HUMAN_DIR = DATA_DIR / "human"
USER_AI_DIR = DATA_DIR / "ai"
SUPP_DIR = DATA_DIR / "supplementary"
ASVSPOOF_DIR = SUPP_DIR / "asvspoof2019_la"
MANIFEST_PATH = DATA_DIR / "manifest.csv"

FEATURES_DIR = ROOT / "features"
FEATURE_CACHE_DIR = FEATURES_DIR / "cache"
MODELS_DIR = ROOT / "models"
FINAL_MODEL_PATH = MODELS_DIR / "final_model.joblib"
FINAL_METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"

SEED = 42
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 3

# Bump when the feature code changes in a way that alters numbers; part of the cache key and
# of the model's feature signature, so stale caches and incompatible models are rejected.
FEATURE_CODE_VERSION = "1"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050          # same analysis rate as the original script
    window_seconds: float = 4.0       # same analysis window length as the original script
    max_windows_per_file: int = 10    # long files contribute up to this many windows
    min_duration_s: float = 1.0       # after trimming; shorter clips are rejected
    trim_top_db: float = 35.0         # leading/trailing silence trim (dB below peak)
    active_rel_db: float = 30.0       # frame counts as active if within this many dB of the p95 frame level
    min_active_fraction: float = 0.5  # windows with less active speech are skipped (unless none qualify)
    silence_dbfs: float = -55.0       # whole-signal RMS below this = silent
    silence_peak: float = 3e-3        # ... or peak amplitude below this = silent
    frame_length: int = 2048
    hop_length: int = 512


@dataclass(frozen=True)
class FeatureConfig:
    feature_set: str = "baseline"     # "baseline" (original 80) or "extended"
    n_mfcc: int = 40
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128


@dataclass(frozen=True)
class PredictConfig:
    allowed_extensions: tuple = (".wav", ".mp3", ".flac", ".ogg")
    max_file_mb: float = 200.0        # CLI / API limit
    ui_max_upload_mb: int = 50        # Streamlit uploader limit
    max_duration_s: float = 900.0     # 15 minutes
    store_uploads: bool = False       # uploaded / recorded audio is never written to disk by default


AUDIO = AudioConfig()
PREDICT = PredictConfig()


def feature_signature(audio: AudioConfig, feat: FeatureConfig) -> dict:
    """Everything that determines the numbers a feature vector contains."""
    return {"audio": asdict(audio), "features": asdict(feat), "feature_code_version": FEATURE_CODE_VERSION}


def config_hash(audio: AudioConfig, feat: FeatureConfig, extra: dict | None = None) -> str:
    payload = feature_signature(audio, feat)
    if extra:
        payload = {**payload, "extra": extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DecisionConfig:
    """Defaults; the trained model's metadata.json overrides these with validated values."""
    threshold: float = 0.5
    inconclusive_halfwidth: float = 0.15
    cost_false_alarm: float = 2.0     # flagging a real person's voice as synthetic
    cost_miss: float = 1.0            # missing a synthetic voice


DECISION = DecisionConfig()

# Chart palette shared by report figures (light) and the app (dark). Categorical slots 1-3
# (blue, orange, aqua) were run through the colour-vision validator for BOTH surfaces with all
# pairs compared: CVD dE >= 9.2, normal-vision dE >= 20.9. Light aqua is < 3:1 on the light surface,
# so charts always carry a legend and a table. Never add a 4th categorical colour to a scatter.
PALETTES = {
    "light": {"human": "#2a78d6", "ai": "#eb6834", "third": "#1baf7a", "neutral": "#898781",
              "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e",
              "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"human": "#3987e5", "ai": "#d95926", "third": "#199e70", "neutral": "#898781",
             "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#f0efec", "ink2": "#c3c2b7",
             "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835"},
}
# Single-hue sequential ramp (blue, light -> dark) for heatmaps / confusion matrices.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PALETTE = PALETTES["dark"]
MODEL_ORDER = ["majority", "logreg", "linear_svm", "rbf_svm", "random_forest", "hist_gb"]
MODEL_LABELS = {
    "majority": "Majority class",
    "logreg": "Logistic regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF SVM (original)",
    "random_forest": "Random forest",
    "hist_gb": "Hist. gradient boosting",
}
