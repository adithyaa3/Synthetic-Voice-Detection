"""Self-contained HTML report built ONLY from saved artifacts (figures embedded, no external requests)."""
from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone

import pandas as pd

from .config import FIGURES_DIR, MODEL_LABELS, REPORTS_DIR, ROOT
from .evaluate import SETTING_LABELS

LABELS = {**MODEL_LABELS, "rbf_svm": "RBF SVM (tuned)", "rbf_svm_original": "RBF SVM (original settings)"}


def _j(name):
    return json.loads((REPORTS_DIR / name).read_text(encoding="utf-8"))


def _img(name: str, alt: str) -> str:
    p = FIGURES_DIR / f"{name}.png"
    if not p.exists():
        return f"<p class='muted'>[figure {html.escape(name)} not generated]</p>"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"<figure><img src='data:image/png;base64,{b64}' alt='{html.escape(alt)}'><figcaption>{html.escape(alt)}</figcaption></figure>"


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<div class='tw'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _pm(m, s):
    return f"{m:.3f} ± {s:.3f}"


CSS = """
:root{--bg:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6b6a65;--line:#e1e0d9;--accent:#127a55}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}
main{max-width:1000px;margin:0 auto;padding:32px 16px 64px}h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 6px}.muted{color:var(--muted)}p{color:var(--ink2);max-width:75ch}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px}.tile .l{font-size:13px;color:var(--muted)}.tile .v{font-size:26px;font-weight:600}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface)}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;font-variant-numeric:tabular-nums}th{color:var(--ink2);font-weight:600}
figure{margin:16px 0}img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:8px}figcaption{font-size:13px;color:var(--muted)}
.callout{background:var(--surface);border-left:3px solid var(--accent);padding:10px 14px;border-radius:4px;margin:14px 0}
code{font-size:13px}
"""


def build() -> str:
    meta = json.loads((ROOT / "models" / "metadata.json").read_text(encoding="utf-8"))
    audit = _j("dataset_audit.json")
    legacy = _j("legacy_reproduction.json")
    sel = _j("selection/model_selection.json")
    aug = _j("selection/augmentation.json")
    cal = _j("selection/calibration.json")
    test = _j("test_metrics.json")
    summ = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    rob = pd.read_csv(REPORTS_DIR / "robustness.csv")
    gen = pd.read_csv(REPORTS_DIR / "generalisation.csv")
    xd = pd.read_csv(REPORTS_DIR / "cross_dataset.csv")
    gimp = pd.read_csv(REPORTS_DIR / "group_permutation_importance.csv")
    abl = pd.read_csv(REPORTS_DIR / "feature_group_ablation.csv")
    errs = pd.read_csv(REPORTS_DIR / "error_summary_test.csv")
    fsum = _j("feature_summary.json")

    dep = test["deployed"]["scopes"]
    allm = dep["all test windows"]
    tiles = [("Test ROC-AUC (all held-out windows)", f"{allm['roc_auc']:.3f}"),
             ("Test balanced accuracy", f"{allm['balanced_accuracy']:.3f}"),
             ("AI recall / human specificity", f"{allm['recall']:.2f} / {allm['specificity']:.2f}"),
             ("Dev CV ROC-AUC, deployed configuration", f"{cal['methods'][cal['chosen_method']]['roc_auc_mean']:.3f}")]
    tiles_html = "".join(f"<div class='tile'><div class='l'>{html.escape(l)}</div><div class='v'>{html.escape(v)}</div></div>" for l, v in tiles)

    comp = summ[(summ.augment == "none") & (summ.calibration == "default")].copy()
    comp["setting"] = comp.setting.map(SETTING_LABELS)
    comp["model"] = comp.model.map(LABELS).fillna(comp.model)
    comp_t = pd.DataFrame({"Setting": comp.setting, "Features": comp.feature_set, "Model": comp.model,
                           "ROC-AUC": [_pm(m, s) for m, s in zip(comp.roc_auc_mean, comp.roc_auc_std)],
                           "Balanced acc.": [_pm(m, s) for m, s in zip(comp.balanced_accuracy_mean, comp.balanced_accuracy_std)],
                           "PR-AUC": [_pm(m, s) for m, s in zip(comp.pr_auc_mean, comp.pr_auc_std)],
                           "Brier": [_pm(m, s) for m, s in zip(comp.brier_mean, comp.brier_std)],
                           "ROC-AUC on MP3 copies": [f"{m:.3f}" if pd.notna(m) else "n/a" for m in comp.get("roc_auc_mp3_mean", pd.Series(dtype=float))]})

    scope_rows = []
    tag_labels = {"deployed": "Deployed (post-hoc)" if "primary" in test else "Deployed",
                  "primary": "Pre-registered (primary result)", "user_only": "User-data-only comparison"}
    for tag in [t for t in ("deployed", "primary", "user_only") if t in test]:
        for scope, m in test[tag]["scopes"].items():
            ci = m.get("roc_auc_ci95")
            scope_rows.append({"Model": tag_labels[tag],
                               "Scope": scope, "Windows": m["n"], "ROC-AUC": m["roc_auc"],
                               "AUC 95% CI (group bootstrap)": f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n/a",
                               "Balanced acc.": m["balanced_accuracy"], "AI recall": m["recall"],
                               "Human specificity": m["specificity"], "Precision": m["precision"],
                               "F1": m["f1"], "Brier": m["brier"], "File-level bal. acc.": m["file_level"]["balanced_accuracy"]})
    gsum = gen.groupby("experiment").agg(folds=("held_out", "size"), recall_ai=("recall_ai", "mean"),
                                         specificity_human=("specificity_human", "mean"),
                                         balanced_accuracy=("balanced_accuracy", "mean"), roc_auc=("roc_auc", "mean")).reset_index()
    rob_t = rob[rob.scope == "all test windows"][["description", "n", "balanced_accuracy", "roc_auc", "recall_ai", "specificity_human"]]
    uv = gen[gen.experiment.str.startswith("LOGO") & (gen.held_out == "user_vc")]
    logo_uv_html = ""
    if len(uv):
        r0 = uv.iloc[0]
        logo_uv_html = (f"<div class='callout'>Trained on all 19 ASVspoof systems but not on the user's conversion tool, the "
                        f"model flags {r0.recall_ai:.1%} of the user's {int(r0.n_test_ai)} AI windows (ROC-AUC "
                        f"{r0.roc_auc:.3f}); trained on the user's data only, it reaches ROC-AUC "
                        f"{float(xd.loc[xd.experiment.str.startswith('train user'), 'roc_auc'].iloc[0]):.3f} on ASVspoof. "
                        "Detection does not transfer to unseen generators or datasets.</div>")
    rp = REPORTS_DIR / "robustness_primary.csv"
    rob_p_html = ""
    if rp.exists():
        rpd = pd.read_csv(rp)
        rob_p_html = ("<h3>Pre-registered model (not deployed)</h3>" + _table(
            rpd[rpd.scope == "all test windows"][["description", "n", "balanced_accuracy", "roc_auc", "recall_ai",
                                                   "specificity_human"]]) + _img("robustness_primary", "Robustness of the pre-registered model"))
    rs_html = ""
    rsp = REPORTS_DIR / "selection" / "robustness_study.json"
    if rsp.exists():
        rs = json.loads(rsp.read_text(encoding="utf-8"))
        rs_tab = pd.DataFrame([{"Training augmentation": r["candidate"], **{f"AUC {c}": v for c, v in r["auc_by_condition"].items()},
                                "Mean": r["robust_auc"]} for r in rs["candidates"]])
        rs_html = (f"<h3>Robustness-aware augmentation study (development data only)</h3><div class='callout'>{html.escape(rs['motivation'])}</div>"
                   f"<p><b>Rule (fixed before the study ran):</b> {html.escape(rs['rule'])} <b>Chosen:</b> {html.escape(rs['chosen'])}.</p>"
                   + _table(rs_tab) + _img("robust_tradeoff", "Clean vs robust validation AUC per augmentation candidate"))
    posthoc_html = ("<div class='callout'>The <b>pre-registered</b> model was chosen before any test evaluation; its rows are the "
                    "primary result. The <b>deployed</b> model was chosen afterwards by the development-data robustness study "
                    "(section 3), so its test rows are post-hoc.</div>") if "primary" in test else ""
    per_src = audit["per_source"]
    src_rows = [{"Source": s, "Origin": v["origin"], "Files": v["files"], "Classes": ", ".join(v["class"]),
                 "Sample rates": ", ".join(f"{k} Hz×{n}" for k, n in v["sample_rates"].items()),
                 "Groups": v["groups"], "Group confidence": v["group_confidence"],
                 "Rejected by loader": len(v["rejected_by_loader"]), "Analysis windows": v["windows_total"]}
                for s, v in per_src.items()]

    from .manifest import load_manifest
    man = load_manifest()
    t_ai = sorted(man[(man.partition == "test") & (man.source == "user_voice_conversion")].speaker.unique())
    uh = man[man.source == "user_commonvoice"]
    t_h = uh[uh.partition == "test"]
    asv_t = man[(man.partition == "test") & (man.source == "asvspoof2019_la")]
    test_desc = (f"user source speakers {', '.join(t_ai)} (all {len(man[(man.partition == 'test') & (man.source == 'user_voice_conversion')])} "
                 f"of their conversions); {t_h.group.nunique()} of {uh.group.nunique()} Common Voice run-groups "
                 f"({len(t_h)} of {len(uh)} clips); and the ASVspoof evaluation partition ({len(asv_t)} files, "
                 f"{asv_t.speaker.nunique()} speakers, systems {', '.join(sorted(set(asv_t.generator) - {'human'}))})")
    leg = legacy
    ch = sel["deploy"]
    body = f"""
<main>
<h1>Synthetic voice detection: evaluation report</h1>
<p class="muted">Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from saved artifacts · model {html.escape(meta['model_version'])}</p>
<div class="callout">Every number in this report was produced by the pipeline (<code>python -m voicedetect.pipeline all</code>) and read from files in <code>reports/</code> and <code>models/</code>. The output of the detector is a probability, not proof; do not use it as the sole evidence against a person.</div>
<div class="tiles">{tiles_html}</div>

<h2>1. Why the original 100 % was not a valid estimate</h2>
<p>The original script reached accuracy {leg['original_split']['accuracy']:.3f} on a single random split with only {leg['original_split']['n_test_ai']} AI test clips, where always answering "human" already scores {leg['original_split']['majority_class_accuracy']:.3f}. {len(leg['original_split']['test_ai_files_with_same_source_recording_in_train'])} of {leg['original_split']['n_test_ai']} AI test files were re-voicings of a source recording that was also in training. With the same features and model, group-aware CV gives balanced accuracy {_pm(leg['group_cv_5fold_x3']['balanced_accuracy']['mean'], leg['group_cv_5fold_x3']['balanced_accuracy']['std'])} and leave-one-source-speaker-out gives {_pm(leg['leave_one_source_out']['balanced_accuracy']['mean'], leg['leave_one_source_out']['balanced_accuracy']['std'])}.</p>
{_img('legacy_vs_honest', 'Original split vs leakage-safe evaluation of the original model')}

<h2>2. Data</h2>
<p>{audit['files_total']} files; {fsum['windows_orig']} analysis windows after {fsum['files_skipped']} files were skipped ({', '.join(f'{k}: {v}' for k, v in fsum['skipped_by_reason'].items())}). User data alone fails the adequacy checks (independent AI groups: {audit['adequacy']['user_only']['independent_groups_ai']}; AI generators: {len(audit['adequacy']['user_only']['ai_generators'])}). Supplementary data: ASVspoof 2019 LA subset (ODC-By licence), adding {len(audit['adequacy']['user_plus_supplementary']['ai_generators']) - 1} spoofing systems.</p>
{_table(pd.DataFrame(src_rows))}
{_img('class_balance', 'Analysis windows per source and class')}

<h2>3. Model comparison (nested, group-aware 5-fold CV on the development partition)</h2>
<p>Hyperparameters were tuned by inner group-aware CV on each outer training fold only. The selection rule was fixed before results were inspected: {html.escape(sel['rule'])}</p>
<div class="callout">Selected: <b>{html.escape(LABELS.get(ch['model'], ch['model']))}</b> on <b>{html.escape(ch['feature_set'])}</b> features, trained on <b>{html.escape(SETTING_LABELS[ch['setting']])}</b>. Best mean AUC in that setting: {html.escape(LABELS.get(sel['selection_by_setting'][ch['setting']]['best_by_mean_auc']['model'], ''))} / {html.escape(sel['selection_by_setting'][ch['setting']]['best_by_mean_auc']['feature_set'])} ({sel['selection_by_setting'][ch['setting']]['best_by_mean_auc']['roc_auc_mean']:.3f}). First augmentation rule (MP3 only): adopted {aug['adopted']}. Deployed training augmentation: {html.escape(', '.join(meta['augmentation']) or 'none')}. Calibration: {html.escape(cal['chosen_method'])}.</div>
{_img('model_comparison_combined', 'ROC-AUC per model, user + supplementary data')}
{_img('model_comparison_user', 'ROC-AUC per model, user data only')}
{_img('model_comparison_codec_matched', 'ROC-AUC per model, user data with AI audio re-encoded to the Common Voice MP3 format')}
{_table(comp_t)}
{_img('rbf_heatmap', 'RBF SVM inner-CV ROC-AUC over C and gamma')}
{rs_html}

<h2>4. Calibration and decision threshold</h2>
<p>{html.escape(cal['cost_rationale'])} Chosen threshold: <b>{meta['threshold']:.2f}</b>; inconclusive band {meta['threshold'] - meta['inconclusive_halfwidth']:.2f}-{meta['threshold'] + meta['inconclusive_halfwidth']:.2f} (on validation predictions {cal['decision']['inconclusive']['fraction_inconclusive']:.1%} of windows fall inside it; accuracy outside it {cal['decision']['inconclusive'].get('accuracy_outside_band', float('nan')):.3f}).</p>
{_img('calibration', 'Reliability of out-of-fold probabilities')}
{_img('threshold_tradeoff', 'Precision / recall / specificity versus threshold (validation)')}

<h2>5. Held-out test set</h2>
<p>Test groups never appeared in development: {test_desc}.</p>
{posthoc_html}
{_table(pd.DataFrame(scope_rows))}
{_img('roc_test', 'ROC curves on the test set')}
{_img('pr_test', 'Precision-recall curves on the test set')}
{_img('confusion_test', 'Confusion matrix on all test windows at the chosen threshold')}

<h2>6. Generalisation</h2>
{logo_uv_html}
{_table(gsum)}
{_table(xd)}
{_img('logo', 'Leave-one-generator-out: detection of each unseen generator')}
{_img('loso', 'Leave-one-source-speaker-out on user data')}

<h2>7. Robustness (test windows)</h2>
<h3>Deployed model</h3>
{_table(rob_t)}
{_img('robustness', 'Robustness under perturbations')}
{rob_p_html}

<h2>8. What the model uses</h2>
{_table(gimp)}
{_table(abl)}
{_img('group_importance', 'Group permutation importance')}
{_img('projection_tsne', 't-SNE of development windows')}
{_img('learning_curve', 'Learning curve over training groups')}

<h2>9. Errors</h2>
{_table(errs)}

<h2>10. Limitations and ethics</h2>
<ul>
<li>The user's AI audio comes from a single (unknown) voice-conversion tool and 8 source recordings; the human audio is crowd-sourced Common Voice MP3. Recording channel differs systematically between the classes, so part of what any model learns may be channel rather than synthesis.</li>
<li>Common Voice speaker identities are unavailable; human groups are inferred from clip-ID runs (low-medium confidence).</li>
<li>ASVspoof 2019 systems date from 2019; modern generators are not covered. Detectors can be fooled by unseen generators, post-processing and adversarial edits.</li>
<li>Microphone recordings differ from all training data (room acoustics, browser processing); treat live results as indicative only.</li>
<li>Accent, language and recording-device bias are not measured. Never use this output as the sole evidence against a person.</li>
</ul>
</main>"""
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Voice detection report</title><style>{CSS}</style></head><body>{body}</body></html>"
    out = REPORTS_DIR / "report.html"
    out.write_text(doc, encoding="utf-8")
    return str(out)
