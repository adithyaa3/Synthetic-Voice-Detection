"""Plotly chart builders shared by the Streamlit app (dark, interactive) and the report figures
(light, exported to PNG). Every builder takes data loaded from saved artifacts.

Colour semantics are fixed across the project:
  blue   = human (or "humans correctly cleared")      orange = AI (or "AI detected")
  aqua   = emphasis (chosen model / overall series)  grey   = de-emphasised context
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import MODEL_LABELS, PALETTES, SEQ_BLUE

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
BAR_PX = 26  # vertical pixels per horizontal bar slot (keeps bar thickness <= 24 px)


def P(mode: str) -> dict:
    return PALETTES[mode]


def style(fig: go.Figure, mode: str = "dark", height: int | None = None, title: str | None = None,
          legend: bool = True) -> go.Figure:
    p = P(mode)
    fig.update_layout(
        template="none", height=height, paper_bgcolor=p["surface"], plot_bgcolor=p["surface"],
        font=dict(family=FONT, size=13, color=p["ink2"]),
        title=dict(text=title, font=dict(size=15, color=p["ink"]), x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=20, t=50 if title else 20, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=p["ink2"]),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT, size=12), bgcolor=p["surface"], bordercolor=p["axis"]),
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=p["grid"], gridwidth=1, zeroline=False, linecolor=p["axis"], showline=True,
                     tickfont=dict(color=p["muted"]), title_font=dict(color=p["ink2"]), automargin=True)
    fig.update_yaxes(gridcolor=p["grid"], gridwidth=1, zeroline=False, linecolor=p["axis"], showline=False,
                     tickfont=dict(color=p["muted"]), title_font=dict(color=p["ink2"]), automargin=True,
                     ticklabelstandoff=8)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def _marker(color: str, mode: str, size: int = 9) -> dict:
    return dict(color=color, size=size, line=dict(color=P(mode)["surface"], width=2))


# ------------------------------------------------------------------------------------------ data
def class_balance(fs_summary: dict, mode: str = "dark") -> go.Figure:
    p = P(mode)
    rows = []
    for k, n in fs_summary["windows_by_source_class"].items():
        src, cls = k.split("/")
        rows.append((src, cls, n))
    df = pd.DataFrame(rows, columns=["source", "class", "n"])
    names = {"user_commonvoice": "User: Common Voice", "user_voice_conversion": "User: voice conversion",
             "asvspoof2019_la": "Supplementary: ASVspoof 2019 LA"}
    df["source"] = df.source.map(names).fillna(df.source)
    df["row"] = df.source + " · " + df["class"].map({"human": "human", "ai": "AI"})
    df = df.sort_values(["source", "class"], ascending=[False, False])
    fig = go.Figure()
    for cls, col, lab in [("human", p["human"], "Human"), ("ai", p["ai"], "AI-generated")]:
        d = df[df["class"] == cls]
        fig.add_bar(y=d.row, x=d.n, name=lab, orientation="h", marker_color=col, width=0.6, text=d.n,
                    textposition="outside", textfont=dict(color=p["ink2"]),
                    hovertemplate="%{y}: %{x} windows<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(categoryorder="array", categoryarray=list(df.row))
    fig.update_xaxes(title="Analysis windows (4 s)", range=[0, df.n.max() * 1.15])
    return style(fig, mode, height=90 + BAR_PX * 1.5 * len(df))


def duration_box(audit: pd.DataFrame, mode: str = "dark") -> go.Figure:
    p = P(mode)
    names = {"user_commonvoice": "User: Common Voice", "user_voice_conversion": "User: voice conversion",
             "asvspoof2019_la": "ASVspoof 2019 LA"}
    fig = go.Figure()
    for cls, col, lab in [("human", p["human"], "Human"), ("ai", p["ai"], "AI-generated")]:
        d = audit[(audit.class_name == cls) & audit.trimmed_duration_s.notna()]
        fig.add_box(x=d.trimmed_duration_s, y=d.source.map(names), name=lab, orientation="h", marker_color=col,
                    line=dict(width=1.5), boxpoints=False,
                    hovertemplate="%{y}<br>%{x:.1f} s<extra>" + lab + "</extra>")
    fig.update_layout(boxmode="group")
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    fig.update_xaxes(title="Speech duration after trimming silence (s, log scale)", type="log",
                     tickvals=ticks, ticktext=[str(t) for t in ticks])
    return style(fig, mode, height=300)


def waveform(y: np.ndarray, sr: int, mode: str = "dark", color: str | None = None, height: int = 180,
             title: str | None = None) -> go.Figure:
    p = P(mode)
    step = max(1, len(y) // 4000)  # envelope-preserving decimation for plotting only
    n = len(y) // step * step
    seg = y[:n].reshape(-1, step)
    t = np.arange(seg.shape[0]) * step / sr
    fig = go.Figure()
    fig.add_scatter(x=np.r_[t, t[::-1]], y=np.r_[seg.max(1), seg.min(1)[::-1]], fill="toself", mode="lines",
                    line=dict(width=1, color=color or p["neutral"]), fillcolor=color or p["neutral"],
                    hoverinfo="skip", showlegend=False)
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude", range=[-1.05, 1.05])
    return style(fig, mode, height=height, title=title, legend=False)


def mel_spectrogram(y: np.ndarray, sr: int, mode: str = "dark", height: int = 240, title: str | None = None) -> go.Figure:
    import librosa
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96), ref=np.max)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr)
    f = librosa.mel_frequencies(n_mels=96, fmax=sr / 2)
    scale = SEQ_BLUE if mode == "light" else SEQ_BLUE[::-1]
    fig = go.Figure(go.Heatmap(z=S, x=t, y=f, colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
                               colorbar=dict(title="dB", thickness=10, tickfont=dict(color=P(mode)["muted"])),
                               hovertemplate="%{x:.2f} s, %{y:.0f} Hz: %{z:.0f} dB<extra></extra>"))
    ticks = [t for t in (100, 300, 1000, 3000, 10000) if t <= sr / 2]
    fig.update_yaxes(title="Mel frequency (Hz)", type="log", tickvals=ticks,
                     ticktext=[f"{t // 1000}k" if t >= 1000 else str(t) for t in ticks])
    fig.update_xaxes(title="Time (s)")
    return style(fig, mode, height=height, title=title, legend=False)


# ---------------------------------------------------------------------------------------- models
def legacy_vs_honest(legacy: dict, mode: str = "dark") -> go.Figure:
    p = P(mode)
    o = legacy["original_split"]
    # short labels: the chart often sits in a half-width column (details are in the caption / tooltip)
    items = [
        ("Original split: accuracy", o["accuracy"], 0, p["neutral"]),
        ("Always 'human': accuracy", o["majority_class_accuracy"], 0, p["neutral"]),
        ("Group CV: balanced acc.", legacy["group_cv_5fold_x3"]["balanced_accuracy"]["mean"],
         legacy["group_cv_5fold_x3"]["balanced_accuracy"]["std"], p["third"]),
        ("New speaker: balanced acc.", legacy["leave_one_source_out"]["balanced_accuracy"]["mean"],
         legacy["leave_one_source_out"]["balanced_accuracy"]["std"], p["third"]),
    ]
    fig = go.Figure()
    for lab, v, sd, col in items[::-1]:
        fig.add_bar(y=[lab], x=[v], orientation="h", marker_color=col, width=0.6,
                    error_x=dict(type="data", array=[sd], visible=sd > 0, color=p["ink2"], thickness=1.5, width=4),
                    hovertemplate=f"{lab}: %{{x:.3f}}" + (f" ± {sd:.3f}" if sd else "") + "<extra></extra>",
                    showlegend=False)
        # value label placed beyond the whisker so it never collides with it
        # value only (the whisker shows the spread; "± std" is in the tooltip) so the label never clips at 390 px
        fig.add_scatter(y=[lab], x=[v + sd + 0.015], mode="text", text=[f"{v:.3f}"],
                        textposition="middle right", textfont=dict(color=p["ink2"]), hoverinfo="skip", showlegend=False)
    fig.update_xaxes(range=[0, 1.5], title="Score (1.0 = perfect)", tickvals=[0, 0.5, 1.0])
    return style(fig, mode, height=100 + BAR_PX * 1.8 * len(items), legend=False)


def model_comparison(summary: pd.DataFrame, setting: str, chosen: tuple[str, str] | None, mode: str = "dark",
                     metric: str = "roc_auc") -> go.Figure:
    p = P(mode)
    s = summary[(summary.setting == setting) & (summary.augment == "none") & (summary.calibration == "default")]
    fsets = ["baseline", "extended"]
    # stacked panels (not side by side) so the chart stays legible at tablet and phone widths
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=["Baseline features (80)", "Extended features (210)"])
    order = [m for m in MODEL_LABELS if m in set(s.model)] + (["rbf_svm_original"] if "rbf_svm_original" in set(s.model) else [])
    order = list(dict.fromkeys(order))
    labels = {**MODEL_LABELS, "rbf_svm": "RBF SVM (tuned)", "rbf_svm_original": "RBF SVM (original settings)"}
    # Dot plot (not bars): AUC differences live in a narrow range, and dots need no zero baseline.
    for j, fs in enumerate(fsets):
        d = s[s.feature_set == fs].set_index("model").reindex(order[::-1]).dropna(how="all")
        is_chosen = [bool(chosen) and (m, fs) == tuple(chosen) for m in d.index]
        cols = [p["third"] if c else p["neutral"] for c in is_chosen]
        ylab = [labels.get(m, m) for m in d.index]
        mean, sd = d[f"{metric}_mean"].to_numpy(), d[f"{metric}_std"].to_numpy()
        fig.add_scatter(y=ylab, x=mean, mode="markers", marker=dict(color=cols, size=[13 if c else 10 for c in is_chosen],
                        line=dict(color=p["surface"], width=2)),
                        error_x=dict(type="data", array=sd, color=p["ink2"], thickness=1.5, width=4),
                        customdata=np.c_[sd, d["balanced_accuracy_mean"]],
                        hovertemplate="%{y}<br>mean %{x:.3f} ± %{customdata[0]:.3f}<br>balanced acc. %{customdata[1]:.3f}<extra></extra>",
                        showlegend=False, row=j + 1, col=1)
        fig.add_scatter(y=ylab, x=np.minimum(mean + sd + 0.012, 1.02), mode="text", text=[f"{v:.3f}" for v in mean],
                        textposition="middle right", textfont=dict(color=p["ink2"], size=12), hoverinfo="skip",
                        showlegend=False, row=j + 1, col=1)
        fig.update_xaxes(range=[0.45, 1.1], row=j + 1, col=1, tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         title=("ROC-AUC (mean ± std, 5 group folds)" if metric == "roc_auc" else metric) if j == 1 else None,
                         showticklabels=True)
    fig.add_vline(x=0.5, line=dict(color=p["axis"], width=1), row="all", col=1)
    fig = style(fig, mode, height=150 + 2 * BAR_PX * len(order), legend=False)
    for a in fig.layout.annotations:
        a.font = dict(color=p["ink2"], size=13)
    return fig


def rbf_heatmap(grid: pd.DataFrame, setting: str, feature_set: str, mode: str = "dark") -> go.Figure:
    g = grid[(grid.setting == setting) & (grid.feature_set == feature_set)]
    h = g.groupby(["C", "gamma"]).mean_auc.mean().unstack("gamma")
    scale = SEQ_BLUE if mode == "light" else SEQ_BLUE[::-1]
    z = h.values
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{c:g}" for c in h.columns], y=[f"{c:g}" for c in h.index],
        colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
        text=np.vectorize(lambda v: f"{v:.3f}")(z), texttemplate="%{text}",
        textfont=dict(size=11), colorbar=dict(title="AUC", thickness=10, tickfont=dict(color=P(mode)["muted"])),
        hovertemplate="C=%{y}, gamma=%{x}<br>inner-CV ROC-AUC %{z:.3f}<extra></extra>", xgap=2, ygap=2))
    fig.update_xaxes(title="gamma", type="category", showgrid=False)
    fig.update_yaxes(title="C", type="category", showgrid=False)
    return style(fig, mode, height=340, legend=False)


# ------------------------------------------------------------------------------------ evaluation
def curves(curves_df: pd.DataFrame, model: str, kind: str, mode: str = "dark", scopes=("all", "user", "asvspoof")) -> go.Figure:
    p = P(mode)
    names = {"all": "All test windows", "user": "User data", "asvspoof": "ASVspoof eval (unseen systems)"}
    colors = {"all": p["third"], "user": p["ink2"], "asvspoof": p["neutral"]}
    fig = go.Figure()
    if kind == "roc":
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=p["axis"], width=1), name="Chance",
                        hoverinfo="skip")
    for sc in scopes:
        d = curves_df[(curves_df.model == model) & (curves_df.scope == sc) & (curves_df.curve == kind)]
        if d.empty:
            continue
        fig.add_scatter(x=d.x, y=d.y, mode="lines", name=names[sc], line=dict(color=colors[sc], width=2.5 if sc == "all" else 2),
                        hovertemplate=("FPR %{x:.3f}<br>TPR %{y:.3f}" if kind == "roc" else "Recall %{x:.3f}<br>Precision %{y:.3f}") + f"<extra>{names[sc]}</extra>")
    if kind == "roc":
        fig.update_xaxes(title="False alarm rate (humans flagged as AI)", range=[-0.01, 1.01])
        fig.update_yaxes(title="Detection rate (AI flagged)", range=[-0.01, 1.01])
    else:
        fig.update_xaxes(title="Recall (AI flagged)", range=[-0.01, 1.01])
        fig.update_yaxes(title="Precision", range=[-0.01, 1.03])
    return style(fig, mode, height=360)


def confusion(m: dict, mode: str = "dark", title: str | None = None) -> go.Figure:
    z = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    pct = z / z.sum(axis=1, keepdims=True)
    scale = SEQ_BLUE if mode == "light" else SEQ_BLUE[::-1]
    text = [[f"{z[i, j]}<br>({pct[i, j]:.0%} of row)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=["Predicted human", "Predicted AI"], y=["Actually human", "Actually AI"],
                               colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)], zmin=0, zmax=1,
                               text=text, texttemplate="%{text}", textfont=dict(size=14), showscale=False,
                               hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>", xgap=3, ygap=3))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return style(fig, mode, height=300, title=title, legend=False)


def reliability(rel: pd.DataFrame, chosen: str, mode: str = "dark") -> go.Figure:
    p = P(mode)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=p["axis"], width=1), name="Perfect calibration",
                    hoverinfo="skip")
    for meth, d in rel.groupby("method"):
        d = d[d["count"] > 0]
        col = p["third"] if meth == chosen else p["neutral"]
        fig.add_scatter(x=d.mean_predicted, y=d.fraction_ai, mode="lines+markers", name=f"{meth}{' (chosen)' if meth == chosen else ''}",
                        line=dict(color=col, width=2), marker=_marker(col, mode, 8), customdata=d["count"],
                        hovertemplate="predicted %{x:.2f}<br>observed AI fraction %{y:.2f}<br>n=%{customdata}<extra>" + meth + "</extra>")
    fig.update_xaxes(title="Mean predicted P(AI) in bin", range=[0, 1])
    fig.update_yaxes(title="Observed fraction AI", range=[0, 1.02])
    return style(fig, mode, height=360)


def threshold_tradeoff(tt: pd.DataFrame, threshold: float, band: tuple[float, float] | None, mode: str = "dark") -> go.Figure:
    p = P(mode)
    fig = go.Figure()
    if band:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor=p["neutral"], opacity=0.12, line_width=0,
                      annotation_text="inconclusive band", annotation_position="top left",
                      annotation_font=dict(color=p["muted"], size=11))
    for col, name, color in [("recall", "AI detected (recall)", p["ai"]), ("specificity", "Humans cleared (specificity)", p["human"]),
                             ("precision", "Precision of AI flags", p["third"])]:
        fig.add_scatter(x=tt.threshold, y=tt[col], mode="lines", name=name, line=dict(color=color, width=2),
                        hovertemplate="threshold %{x:.2f}<br>" + name + " %{y:.3f}<extra></extra>")
    fig.add_vline(x=threshold, line=dict(color=p["ink2"], width=1.5))
    fig.add_annotation(x=threshold, y=0.02, text=f"chosen {threshold:.2f}", showarrow=False, xanchor="left", xshift=4,
                       font=dict(color=p["ink2"], size=11))
    fig.update_xaxes(title="Decision threshold on P(AI)", range=[0, 1])
    fig.update_yaxes(title="Rate", range=[0, 1.03])
    return style(fig, mode, height=380)


# ---------------------------------------------------------------------------------- robustness etc.
SHORT_CONDITION = {
    "clean": "Clean", "white_noise_30dB": "White noise 30 dB", "white_noise_20dB": "White noise 20 dB",
    "white_noise_10dB": "White noise 10 dB", "white_noise_5dB": "White noise 5 dB", "room_noise_20dB": "Room noise 20 dB",
    "room_noise_10dB": "Room noise 10 dB", "mp3_128k": "MP3 128 kbps", "mp3_64k": "MP3 64 kbps",
    "mp3_48k": "MP3 48 kbps (CV)", "mp3_32k": "MP3 32 kbps", "resample_16k": "Resample 16 kHz",
    "resample_8k": "Resample 8 kHz", "truncate_1s": "First 1 s", "truncate_2s": "First 2 s", "truncate_4s": "First 4 s",
    "volume_0.1x": "Volume x0.1", "volume_0.5x": "Volume x0.5", "volume_2x_clip": "Volume x2 (clip)",
    "lowpass_4k": "Low-pass 4 kHz", "reverb_room": "Room reverb", "mic_like": "Mic-like chain",
}


def robustness_dots(rob: pd.DataFrame, mode: str = "dark", scope: str = "all test windows") -> go.Figure:
    p = P(mode)
    d = rob[rob.scope == scope].iloc[::-1]
    ylab = [SHORT_CONDITION.get(c, c) for c in d.condition]  # short axis labels; full text in the tooltip
    fig = go.Figure()
    for col, name, color in [("specificity_human", "Humans cleared (specificity)", p["human"]),
                             ("recall_ai", "AI detected (recall)", p["ai"])]:
        fig.add_scatter(x=d[col], y=ylab, mode="markers", name=name, marker=_marker(color, mode, 10),
                        customdata=np.c_[d.balanced_accuracy, d.roc_auc, d.description],
                        hovertemplate="%{customdata[2]}<br>" + name + " %{x:.3f}<br>balanced acc. %{customdata[0]:.3f}, AUC %{customdata[1]:.3f}<extra></extra>")
    clean = rob[(rob.scope == scope) & (rob.condition == "clean")]
    if len(clean) and pd.notna(clean.balanced_accuracy.iloc[0]):
        fig.add_vline(x=float(clean.balanced_accuracy.iloc[0]), line=dict(color=p["axis"], width=1),
                      annotation_text="clean balanced acc.", annotation_position="top",
                      annotation_font=dict(color=p["muted"], size=11))
    fig.update_xaxes(title="Rate on test windows", range=[-0.02, 1.02])
    return style(fig, mode, height=110 + 24 * len(d))


def generalisation_bars(gen: pd.DataFrame, experiment_prefix: str, mode: str = "dark") -> go.Figure:
    p = P(mode)
    d = gen[gen.experiment.str.startswith(experiment_prefix)].iloc[::-1].copy()
    desc = d.held_out_description.fillna("") if "held_out_description" in d else pd.Series([""] * len(d), index=d.index)
    # short axis label ("A13 · TTS+VC"); the full system description is in the tooltip and the table
    d["label"] = [f"{h} · {x.split(',')[0]}" if x and h.startswith("A") else h for h, x in zip(d.held_out, desc)]
    d["full"] = [f"{h}: {x}" if x else h for h, x in zip(d.held_out, desc)]
    fig = go.Figure()
    # recall is the quantity of interest and can be exactly 0 (an invisible bar), so it is labelled
    fig.add_bar(y=d.label, x=d.recall_ai, orientation="h", name="AI detected (recall) on the held-out value",
                marker_color=p["ai"], width=0.42, offset=-0.42, text=[f"{v:.2f}" for v in d.recall_ai],
                textposition="outside", textfont=dict(color=p["ink2"], size=11), cliponaxis=False,
                customdata=d.full, hovertemplate="held out %{customdata}<br>AI recall %{x:.3f}<extra></extra>")
    fig.add_bar(y=d.label, x=d.specificity_human, orientation="h", name="Humans cleared (specificity)",
                marker_color=p["human"], width=0.42, offset=0.0, customdata=d.full,
                hovertemplate="held out %{customdata}<br>human specificity %{x:.3f}<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 1.08], title="Rate at the deployed threshold")
    return style(fig, mode, height=110 + BAR_PX * 1.3 * len(d))


def robust_tradeoff(study: dict, mode: str = "dark") -> go.Figure:
    """Clean vs robust (mean over perturbed conditions) validation AUC per augmentation candidate."""
    p = P(mode)
    rows = study["candidates"]
    base_clean = rows[0]["auc_by_condition"]["clean"]
    tol = float(study["rule"].split("within ")[1].split(" ")[0]) if "within " in study["rule"] else 0.02
    xs = [r["auc_by_condition"]["clean"] for r in rows]
    ys = [r["robust_auc"] for r in rows]
    lo_x = min(xs + [base_clean - tol]) - 0.01
    fig = go.Figure()
    fig.add_vrect(x0=base_clean - tol, x1=1.0, fillcolor=p["third"], opacity=0.08, line_width=0,
                  annotation_text="eligible: clean AUC within tolerance", annotation_position="top left",
                  annotation_font=dict(color=p["muted"], size=11))
    for i, (r, x, y) in enumerate(zip(rows, xs, ys)):
        chosen = r["candidate"] == study["chosen"]
        col = p["third"] if chosen else p["neutral"]
        # candidates can coincide; alternate label sides so labels never overprint
        near = sum(1 for xx, yy in zip(xs[:i], ys[:i]) if abs(xx - x) < 0.004 and abs(yy - y) < 0.01)
        pos = ["middle right", "bottom right", "top right"][near % 3]
        fig.add_scatter(x=[x], y=[y], mode="markers+text", marker=_marker(col, mode, 14 if chosen else 11),
                        text=[r["candidate"] + (" (chosen)" if chosen else "")], textposition=pos,
                        textfont=dict(color=p["ink2"]), showlegend=False, cliponaxis=False,
                        hovertemplate=(f"{r['candidate']}<br>clean AUC %{{x:.3f}}<br>robust AUC %{{y:.3f}}<br>" +
                                       "<br>".join(f"{c}: {v:.3f}" for c, v in r["auc_by_condition"].items()) +
                                       "<extra></extra>"))
    fig.update_xaxes(title="Validation ROC-AUC on clean windows", range=[lo_x, 1.0])
    fig.update_yaxes(title="Robust AUC (mean of 5 conditions)", range=[min(ys) - 0.02, max(ys) + 0.03])
    return style(fig, mode, height=380, legend=False)


def importance_bars(df: pd.DataFrame, label_col: str, value_col: str, err_col: str | None, mode: str = "dark",
                    xtitle: str = "Drop in ROC-AUC when permuted") -> go.Figure:
    p = P(mode)
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(y=d[label_col], x=d[value_col], orientation="h", marker_color=p["third"], width=0.6,
                           error_x=dict(type="data", array=d[err_col], color=p["ink2"], thickness=1.2, width=3) if err_col else None,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title=xtitle)
    return style(fig, mode, height=90 + BAR_PX * len(d), legend=False)


def projection(coords: pd.DataFrame, method: str, mode: str = "dark") -> go.Figure:
    p = P(mode)
    fig = go.Figure()
    symbols = {"user": "circle", "public": "diamond"}
    for (cls, dataset), d in coords.assign(dataset=np.where(coords.source == "asvspoof2019_la", "public", "user")).groupby(["class_name", "dataset"]):
        col = p["human"] if cls == "human" else p["ai"]
        name = f"{'Human' if cls == 'human' else 'AI'} · {'ASVspoof' if dataset == 'public' else 'user data'}"
        fig.add_scatter(x=d[f"{method}1"], y=d[f"{method}2"], mode="markers", name=name,
                        marker=dict(color=col, size=7, symbol=symbols[dataset], opacity=0.75,
                                    line=dict(color=p["surface"], width=1)),
                        customdata=np.c_[d.file_id, d.generator],
                        hovertemplate="%{customdata[0]}<br>generator %{customdata[1]}<extra>" + name + "</extra>")
    fig.update_xaxes(title=f"{method.upper()} 1", showticklabels=False)
    fig.update_yaxes(title=f"{method.upper()} 2", showticklabels=False)
    return style(fig, mode, height=440)


def learning_curve(lc: pd.DataFrame, mode: str = "dark") -> go.Figure:
    p = P(mode)
    g = lc.groupby("fraction_of_training_groups").agg(auc=("roc_auc", "mean"), sd=("roc_auc", "std"),
                                                      n=("n_train_windows", "mean")).reset_index()
    g["sd"] = g.sd.fillna(0.0)
    fig = go.Figure()
    x = g.fraction_of_training_groups
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[g.auc + g.sd, (g.auc - g.sd)[::-1]], fill="toself", mode="lines",
                    line=dict(width=0), fillcolor=_rgba(p["third"], 0.12), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=g.auc, mode="lines+markers", name="ROC-AUC (mean ± std over folds)",
                    line=dict(color=p["third"], width=2), marker=_marker(p["third"], mode, 8), customdata=g.n,
                    hovertemplate="%{x:.0%} of training groups (~%{customdata:.0f} windows)<br>AUC %{y:.3f}<extra></extra>")
    fig.update_xaxes(title="Fraction of training groups used", tickformat=".0%")
    fig.update_yaxes(title="ROC-AUC on validation folds")
    return style(fig, mode, height=340, legend=False)


def probability_meter(p_ai: float, threshold: float, band: tuple[float, float], mode: str = "dark") -> go.Figure:
    p = P(mode)
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, x1=1, y0=0.3, y1=0.7, fillcolor=p["grid"], line_width=0)
    fig.add_shape(type="rect", x0=band[0], x1=band[1], y0=0.3, y1=0.7, fillcolor=p["neutral"], opacity=0.35, line_width=0)
    col = p["ai"] if p_ai >= band[1] else p["human"] if p_ai < band[0] else p["neutral"]
    fig.add_shape(type="rect", x0=0, x1=p_ai, y0=0.38, y1=0.62, fillcolor=col, line_width=0)
    fig.add_shape(type="line", x0=threshold, x1=threshold, y0=0.15, y1=0.85, line=dict(color=p["ink"], width=2))
    fig.add_annotation(x=threshold, y=0.92, text=f"threshold {threshold:.2f}", showarrow=False, font=dict(color=p["ink2"], size=11))
    fig.add_annotation(x=0, y=0.05, text="0 · human-like", showarrow=False, xanchor="left", font=dict(color=p["muted"], size=11))
    fig.add_annotation(x=1, y=0.05, text="AI-like · 1", showarrow=False, xanchor="right", font=dict(color=p["muted"], size=11))
    fig.update_xaxes(range=[-0.01, 1.01], visible=False)
    fig.update_yaxes(range=[0, 1], visible=False)
    return style(fig, mode, height=110, legend=False)


def window_probs(probs: list[float], spans: list, threshold: float, mode: str = "dark") -> go.Figure:
    p = P(mode)
    labels = [f"{s:.0f}-{e:.0f} s" for s, e in spans]
    cols = [p["ai"] if v >= threshold else p["human"] for v in probs]
    fig = go.Figure(go.Bar(x=labels, y=probs, marker_color=cols, width=0.5,
                           hovertemplate="window %{x}<br>P(AI) %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line=dict(color=p["ink2"], width=1.5), annotation_text=f"threshold {threshold:.2f}",
                  annotation_position="top right", annotation_font=dict(color=p["ink2"], size=11))
    fig.update_yaxes(range=[0, 1.05], title="P(AI) per window")
    fig.update_xaxes(title="Analysed window", type="category")
    return style(fig, mode, height=240, legend=False)
