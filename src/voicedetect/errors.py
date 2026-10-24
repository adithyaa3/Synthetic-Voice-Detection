"""Error analysis: which clips, sources and generators the model gets wrong."""
from __future__ import annotations

import numpy as np
import pandas as pd


def file_level(pred: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window predictions to files (mean probability), as predict_file does."""
    agg = pred.groupby(["file_id", "path", "label", "source", "generator", "group"], as_index=False).agg(
        p_ai=("p_ai", "mean"), n_windows=("p_ai", "size"), p_min=("p_ai", "min"), p_max=("p_ai", "max"))
    return agg


def misclassified(pred: pd.DataFrame, threshold: float, halfwidth: float) -> pd.DataFrame:
    d = pred.copy()
    d["predicted"] = np.where(d.p_ai >= threshold, "ai", "human")
    d["truth"] = np.where(d.label == 1, "ai", "human")
    d["inconclusive"] = (d.p_ai - threshold).abs() < halfwidth
    d["error_type"] = np.select([(d.label == 0) & (d.predicted == "ai"), (d.label == 1) & (d.predicted == "human")],
                                ["false alarm (human flagged as AI)", "miss (AI passed as human)"], "correct")
    d["confidence_distance"] = (d.p_ai - threshold).abs()
    return d[d.error_type != "correct"].sort_values("confidence_distance", ascending=False)


def summary(pred: pd.DataFrame, threshold: float) -> pd.DataFrame:
    from .manifest import generator_descriptions
    desc = generator_descriptions()
    rows = []
    for (src, gen), d in pred.groupby(["source", "generator"]):
        wrong = (d.p_ai >= threshold).astype(int) != d.label
        rows.append({"source": src, "generator": gen, "generator_type": desc.get(gen, ""),
                     "truth": "ai" if d.label.iloc[0] == 1 else "human",
                     "n": int(len(d)), "errors": int(wrong.sum()), "error_rate": float(wrong.mean()),
                     "mean_p_ai": float(d.p_ai.mean())})
    return pd.DataFrame(rows).sort_values(["truth", "error_rate"], ascending=[True, False])
