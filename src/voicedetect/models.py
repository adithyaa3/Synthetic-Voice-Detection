"""Model zoo. Every model is an sklearn Pipeline so scaling is fit on training folds only."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from .config import SEED

# Models whose scores are margins, not probabilities: wrapped in sigmoid (Platt) calibration
# fit on group-aware inner folds of the TRAINING data only.
NEEDS_PLATT = {"linear_svm", "rbf_svm", "rbf_svm_original"}

# Simplicity order used to break ties (simpler first).
SIMPLICITY = ["majority", "logreg", "linear_svm", "rbf_svm", "rbf_svm_original", "random_forest", "hist_gb"]

RBF_C = [0.1, 1, 10, 100, 1000]
RBF_GAMMA = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]

PARAM_GRIDS = {
    "majority": {},
    "logreg": {"clf__C": [0.001, 0.01, 0.1, 1, 10]},
    "linear_svm": {"clf__C": [1e-4, 1e-3, 1e-2, 1e-1, 1]},
    "rbf_svm": {"clf__C": RBF_C, "clf__gamma": RBF_GAMMA},
    "rbf_svm_original": {},  # C=10, gamma="scale", no class weights: the original script, untuned
    "random_forest": {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.3]},
    "hist_gb": {"clf__learning_rate": [0.03, 0.1], "clf__max_leaf_nodes": [15, 31], "clf__l2_regularization": [0.0, 1.0]},
}


def make_estimator(name: str, seed: int = SEED, n_jobs: int = 1) -> Pipeline:
    if name == "majority":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])
    if name == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    elif name == "linear_svm":
        clf = LinearSVC(class_weight="balanced", max_iter=20000, random_state=seed)
    elif name == "rbf_svm":
        clf = SVC(kernel="rbf", class_weight="balanced", cache_size=500)
    elif name == "rbf_svm_original":
        clf = SVC(kernel="rbf", C=10, gamma="scale", cache_size=500)
    elif name == "random_forest":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
    elif name == "hist_gb":
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=False, class_weight="balanced", random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
