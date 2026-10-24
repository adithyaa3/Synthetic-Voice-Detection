"""Browser verification of the Streamlit app with Playwright (drives the installed Chrome).

    python scripts/e2e_check.py [--port 8599] [--headed]

What it does
* starts `streamlit run app/streamlit_app.py` and waits for /_stcore/health
* opens every page at 1440, 768 and 390 px; full-page screenshots -> reports/screenshots/
* records console errors, page errors, failed requests, external (non-localhost) requests and
  horizontal overflow
* exercises controls on every page (segmented controls, sliders, selects, tabs, table row selection,
  downloads), uploads valid WAV/MP3 files and invalid ones (wrong type, fake WAV, silent, too short,
  too large)
* microphone: Chrome fake media stream fed from a WAV file (--use-file-for-fake-audio-capture):
  records, stops, classifies; then a second browser without permission checks the denied path
Results -> reports/e2e_results.json. Test media are generated in a temp folder and deleted.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports" / "screenshots"
RESULTS = ROOT / "reports" / "e2e_results.json"
PAGES = [("overview", ""), ("dataset", "dataset"), ("features", "features"), ("models", "models"),
         ("evaluation", "evaluation"), ("robustness", "robustness"), ("errors", "errors"), ("live", "live"),
         ("about", "about")]
WIDTHS = {"desktop": 1440, "tablet": 768, "narrow": 390}


# ------------------------------------------------------------------------------------- test media
def make_media(tmp: Path) -> dict:
    import librosa
    from voicedetect.manifest import load_manifest
    m = load_manifest()
    test = m[(m.partition == "test") & (m.origin == "user")]
    human = test[test.label == 0].sort_values("file_id").iloc[0]
    ai = test[test.label == 1].sort_values("file_id").iloc[0]
    media = {"human_mp3": tmp / "human_test_clip.mp3", "ai_wav": tmp / "ai_test_excerpt.wav"}
    shutil.copy(ROOT / human.path, media["human_mp3"])
    y, sr = sf.read(ROOT / ai.path, start=44100 * 60, frames=44100 * 15, dtype="float32", always_2d=True)
    sf.write(media["ai_wav"], y, sr)
    for tag, src in (("human", media["human_mp3"]), ("ai", media["ai_wav"])):
        z, _ = librosa.load(str(src), sr=48000, mono=True)
        z = z[: 48000 * 8]
        media[f"mic_{tag}"] = tmp / f"mic_{tag}.wav"
        sf.write(media[f"mic_{tag}"], z, 48000, subtype="PCM_16")
    media["notes_txt"] = tmp / "notes.txt"
    media["notes_txt"].write_text("this is not audio")
    media["fake_wav"] = tmp / "renamed_document.wav"
    media["fake_wav"].write_bytes(b"Definitely not RIFF audio data. " * 200)
    media["silent_wav"] = tmp / "silence.wav"
    sf.write(media["silent_wav"], np.zeros(44100 * 4, dtype=np.float32), 44100)
    media["short_wav"] = tmp / "half_second.wav"
    sf.write(media["short_wav"], (0.3 * np.sin(2 * np.pi * 220 * np.arange(22050) / 44100)).astype(np.float32), 44100)
    media["big_wav"] = tmp / "too_big.wav"
    sf.write(media["big_wav"], (0.01 * np.random.default_rng(0).standard_normal(48000 * 2 * 280)).astype(np.float32)
             .reshape(-1, 2), 48000, subtype="PCM_16")  # ~107 MB
    media["_sources"] = {"human": human.file_id, "ai": ai.file_id}
    return media


# ------------------------------------------------------------------------------------- server
def start_server(port: int) -> subprocess.Popen:
    log = open(ROOT / "reports" / "run_e2e_server.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port", str(port),
                             "--server.headless", "true"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2).read() == b"ok":
                return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise RuntimeError("streamlit did not become healthy")


# ------------------------------------------------------------------------------------- helpers
class Recorder:
    def __init__(self):
        self.console, self.page_errors, self.failed, self.external, self.http_errors = [], [], [], [], []

    def attach(self, page, port):
        page.on("console", lambda m: self.console.append({"type": m.type, "text": m.text[:300], "url": page.url})
                if m.type == "error" else None)
        page.on("pageerror", lambda e: self.page_errors.append({"error": str(e)[:300], "url": page.url}))
        page.on("requestfailed", lambda r: self.failed.append({"url": r.url[:200], "failure": str(r.failure)[:120]}))
        page.on("response", lambda r: self.http_errors.append({"url": r.url[:200], "status": r.status, "page": page.url})
                if r.status >= 400 else None)
        page.on("request", lambda r: self.external.append(r.url[:200])
                if not r.url.startswith((f"http://localhost:{port}", f"ws://localhost:{port}", "data:", "blob:")) else None)


def settle(page, timeout=60):
    """Wait until Streamlit has finished running the script and spinners are gone."""
    page.wait_for_selector('[data-testid="stMain"]', timeout=timeout * 1000)
    t0 = time.time()
    quiet = 0
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count() + page.locator('[data-testid="stSpinner"]').count()
        quiet = quiet + 1 if running == 0 else 0
        if quiet >= 4:
            return
        time.sleep(0.25)


def click_choice(page, name: str) -> None:
    """Click a segmented-control / pills option; single-select groups expose radios, multi-select buttons."""
    for role in ("radio", "button"):
        loc = page.get_by_role(role, name=name, exact=True)
        if loc.count():
            loc.first.click()
            settle(page)
            return
    raise LookupError(f"no radio/button named {name!r}")


def choose_option(page, selectbox, index: int, attempts: int = 3) -> str:
    """Open a Streamlit (react-aria) selectbox and pick option ``index``; retries if a rerun closes it."""
    for _ in range(attempts):
        selectbox.scroll_into_view_if_needed()
        selectbox.get_by_role("button", name="Open").click()
        try:
            page.get_by_role("listbox").first.wait_for(state="visible", timeout=5000)
            opt = page.get_by_role("option").nth(index)
            text = opt.inner_text()
            opt.click()
            settle(page)
            return text
        except Exception:
            settle(page)
    raise TimeoutError("selectbox options did not appear")


MAIN = '[data-testid="stMain"]'  # Streamlit scrolls inside this container, not the document


def page_problems(page) -> dict:
    return {
        "exceptions": page.locator('[data-testid="stException"]').count(),
        "friendly_page_error": page.get_by_text("Something went wrong while rendering").count(),
        "empty_states": page.get_by_text("is not available yet").count(),
        # overflow of the document AND of the inner scroll container (where overflow would actually occur)
        "horizontal_overflow_px": page.evaluate(
            f"""() => {{ const d = document.documentElement, m = document.querySelector('{MAIN}');
                 return Math.max(0, d.scrollWidth - d.clientWidth, m ? m.scrollWidth - m.clientWidth : 0); }}"""),
    }


def shot(page, name):
    """Screenshot of the WHOLE page: grow the viewport to the inner container's scroll height first."""
    OUT.mkdir(parents=True, exist_ok=True)
    vp = page.viewport_size
    h = page.evaluate(f"() => {{ const m = document.querySelector('{MAIN}'); return m ? m.scrollHeight : 0; }}") or vp["height"]
    page.set_viewport_size({"width": vp["width"], "height": int(min(max(h, vp["height"]), 16000))})
    time.sleep(0.6)  # let charts re-layout for the new size
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
    page.set_viewport_size(vp)


def check(results, name, ok, detail=""):
    results.append({"check": name, "pass": bool(ok), "detail": str(detail)[:400]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")


def text_of(page, selector):
    loc = page.locator(selector)
    return loc.first.inner_text() if loc.count() else ""


# ------------------------------------------------------------------------------------- scenarios
def run_pages(browser, base, port, rec, results):
    for wname, width in WIDTHS.items():
        ctx = browser.new_context(viewport={"width": width, "height": 900}, device_scale_factor=1)
        page = ctx.new_page()
        rec.attach(page, port)
        for pname, path in PAGES:
            page.goto(f"{base}/{path}")
            settle(page)
            prob = page_problems(page)
            shot(page, f"{pname}_{wname}_{width}")
            check(results, f"page {pname} @ {width}px renders without errors",
                  prob["exceptions"] == 0 and prob["friendly_page_error"] == 0 and prob["empty_states"] == 0, prob)
            check(results, f"page {pname} @ {width}px has no horizontal overflow", prob["horizontal_overflow_px"] <= 2, prob)
        ctx.close()


def run_controls(browser, base, port, rec, results, tmp):
    ctx = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
    page = ctx.new_page()
    rec.attach(page, port)

    # Overview: headline tiles must equal the saved artifacts (G15), then downloads
    page.goto(f"{base}/")
    settle(page)
    tm = json.loads((ROOT / "reports" / "test_metrics.json").read_text(encoding="utf-8"))["deployed"]["scopes"]["all test windows"]
    audit = json.loads((ROOT / "reports" / "dataset_audit.json").read_text(encoding="utf-8"))
    fsum = json.loads((ROOT / "reports" / "feature_summary.json").read_text(encoding="utf-8"))
    expected = {"Test ROC-AUC": f"{tm['roc_auc']:.3f}",
                "Test balanced accuracy": f"{tm['balanced_accuracy']:.3f}",
                "Audio files analysed": f"{audit['files_total']:,}", "Analysis windows (4 s)": f"{fsum['windows_orig']:,}"}
    for label, value in expected.items():
        tile = page.locator('[data-testid="stMetric"]').filter(has_text=label).first
        shown = tile.locator('[data-testid="stMetricValue"]').inner_text().strip()
        check(results, f"overview tile '{label}' equals artifact", shown == value, f"shown {shown!r} vs artifact {value!r}")
    for label in ("Full report (HTML)", "Model comparison (CSV)", "Test predictions (CSV)"):
        with page.expect_download() as dl:
            page.get_by_role("button", name=label).click()
        f = tmp / dl.value.suggested_filename
        dl.value.save_as(f)
        check(results, f"overview download '{label}'", f.stat().st_size > 1000, f"{f.stat().st_size} bytes")

    # Dataset: partition filter + example selection plays audio
    page.goto(f"{base}/dataset")
    settle(page)
    page.get_by_role("button", name="test", exact=True).first.click()
    settle(page)
    cap = page.get_by_text("rows · 'dev' = used for model development").first.inner_text()
    check(results, "dataset partition filter updates the table", "rows" in cap, cap)
    check(results, "dataset example audio players present", page.locator('[data-testid="stAudio"]').count() >= 2,
          page.locator('[data-testid="stAudio"]').count())
    sel = page.locator('[data-testid="stSelectbox"]').nth(1)
    picked = choose_option(page, sel, 1)
    shown = sel.get_by_role("combobox").input_value()
    check(results, "dataset example select changes clip",
          page.locator('[data-testid="stException"]').count() == 0 and shown == picked.strip(), f"{picked!r} -> {shown!r}")

    # Features: slider + projection switch
    page.goto(f"{base}/features")
    settle(page)
    # Streamlit 1.66 sliders are a visually hidden <input type="range"> (implicit slider role)
    slider = page.locator('[data-testid="stSlider"] input[type="range"]').first
    before = slider.input_value()
    slider.focus()
    page.keyboard.press("ArrowRight")
    settle(page)
    after = page.locator('[data-testid="stSlider"] input[type="range"]').first.input_value()
    check(results, "features coefficient slider responds to keyboard", before != after, f"{before} -> {after}")
    click_choice(page, "PCA")
    settle(page)
    check(results, "features projection switch (PCA)", page.locator('[data-testid="stException"]').count() == 0)

    # Models: setting + feature-set switches, expander, CSV download
    page.goto(f"{base}/models")
    settle(page)
    for label in ("User data only", "User data, codec-matched control", "User + supplementary (ASVspoof 2019 LA)"):
        click_choice(page, label)
        settle(page)
        check(results, f"models setting switch '{label}'", page.locator('[data-testid="stException"]').count() == 0)
    click_choice(page, "extended")
    settle(page)
    page.get_by_text("Paired comparisons (same folds)").click()
    with page.expect_download() as dl:
        page.get_by_role("button", name="Download this table (CSV)").click()
    check(results, "models table CSV download", dl.value.suggested_filename.endswith(".csv"), dl.value.suggested_filename)

    # Evaluation: threshold slider recomputes metrics; source + scope switches; PR tab
    page.goto(f"{base}/evaluation")
    settle(page)
    def recall_text():
        return page.locator('[data-testid="stMetric"]').filter(has_text="AI recall").first.inner_text()
    r0 = recall_text()
    s = page.locator('[data-testid="stSlider"] input[type="range"]').first
    s.focus()
    for _ in range(25):
        page.keyboard.press("ArrowRight")
    settle(page)
    r1 = recall_text()
    check(results, "evaluation threshold slider recomputes recall", r0 != r1, f"{r0!r} -> {r1!r}")
    # the recomputed value must equal an independent recomputation from the saved predictions
    import pandas as pd
    from voicedetect.metrics import binary_metrics
    thr = float(page.locator('[data-testid="stSlider"] input[type="range"]').first.input_value())
    oof = pd.read_parquet(ROOT / "reports" / "oof_final_config.parquet")
    exp = f"{binary_metrics(oof.label.values, oof.p_ai.values, thr)['recall']:.3f}"
    check(results, "evaluation slider value matches recomputation from saved predictions", exp in r1, f"thr {thr}: {r1!r} vs {exp}")
    click_choice(page, "Test")
    settle(page)
    click_choice(page, "ASVspoof")
    settle(page)
    page.get_by_role("tab", name="Precision-recall curve").click()
    check(results, "evaluation switches (test / ASVspoof / PR tab)", page.locator('[data-testid="stException"]').count() == 0)

    # Robustness: model toggle (if two models exist), scope select, pills, tabs
    page.goto(f"{base}/robustness")
    settle(page)
    if (page.get_by_role("radio", name="Pre-registered model", exact=True).count() or page.get_by_role("button", name="Pre-registered model", exact=True).count()):
        click_choice(page, "Pre-registered model")
        settle(page)
        check(results, "robustness model toggle (pre-registered)", page.locator('[data-testid="stException"]').count() == 0)
        click_choice(page, "Deployed model")
        settle(page)
    choose_option(page, page.locator('[data-testid="stSelectbox"]').first, 1)
    click_choice(page, "noise")
    settle(page)
    for tab in ("Leave one source speaker out", "Leave one target voice out", "Cross-dataset", "Leave one generator out"):
        page.get_by_role("tab", name=tab).click()
    check(results, "robustness scope/pills/tabs", page.locator('[data-testid="stException"]').count() == 0)

    # Errors: filter + select first row -> audio
    page.goto(f"{base}/errors")
    settle(page)
    click_choice(page, "Validation (out-of-fold)")
    settle(page)
    grid = page.locator('[data-testid="stDataFrame"]').nth(1)
    grid.scroll_into_view_if_needed()  # the table is below the fold; click coordinates must be on screen
    time.sleep(0.5)
    box = grid.bounding_box()
    page.mouse.click(box["x"] + 14, box["y"] + 35 + 17)  # selection cell of the first data row
    settle(page)
    has_audio = page.locator('[data-testid="stAudio"]').count() > 0
    check(results, "errors: selecting a misclassified window plays its audio", has_audio)
    shot(page, "errors_row_selected_desktop")
    ctx.close()


def run_upload(browser, base, port, rec, results, media):
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = ctx.new_page()
    rec.attach(page, port)
    page.goto(f"{base}/live")
    settle(page)
    inp = page.locator('[data-testid="stFileUploaderDropzoneInput"]')
    outcomes = {}
    for key, expect in (("human_mp3", ".vd-verdict"), ("ai_wav", ".vd-verdict")):
        inp.set_input_files(str(media[key]))
        # wait for THIS file to be listed, then for the rerun to finish (avoids reading a stale verdict)
        page.get_by_text(media[key].name).first.wait_for(timeout=60000)
        time.sleep(1.0)
        settle(page, timeout=180)
        page.wait_for_selector(expect, timeout=120000)
        outcomes[key] = text_of(page, ".vd-verdict")
        shot(page, f"live_upload_{key}_desktop")
        check(results, f"upload {key} ({media['_sources']['human' if 'human' in key else 'ai']}) classified end to end",
              "verdict" in outcomes[key].lower(), outcomes[key].replace("\n", " ")[:200])
    for key, needle in (("fake_wav", "couldn't be decoded"), ("silent_wav", "silent or almost silent"),
                        ("short_wav", "is too short")):
        inp.set_input_files(str(media[key]))
        page.get_by_text(media[key].name).first.wait_for(timeout=60000)
        page.get_by_text(needle).first.wait_for(timeout=60000)
        shot(page, f"live_upload_invalid_{key}_desktop")
        check(results, f"upload {key} shows friendly message", True, needle)
    for key, needle in (("notes_txt", "not allowed"), ("big_wav", "larger than")):
        inp.set_input_files(str(media[key]))
        try:
            page.get_by_text(needle, exact=False).first.wait_for(timeout=30000)
            ok = True
        except Exception:
            ok = False
        shot(page, f"live_upload_rejected_{key}_desktop")
        check(results, f"uploader rejects {key} with a message", ok, needle)
    ctx.close()
    return outcomes


def run_mic(pw, base, port, rec, results, wav: Path, tag: str, headed: bool):
    args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={wav}"]
    browser = pw.chromium.launch(channel="chrome", headless=not headed, args=args)
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000}, permissions=["microphone"])
    page = ctx.new_page()
    rec.attach(page, port)
    page.goto(f"{base}/live")
    settle(page)
    page.get_by_role("tab", name="Record with microphone").click()
    page.get_by_role("button", name="Record", exact=True).click()
    time.sleep(7)
    page.get_by_role("button", name="Stop recording").click()
    page.get_by_role("button", name="Classify recording").wait_for(timeout=30000)
    shot(page, f"live_mic_recorded_{tag}_desktop")
    play_ok = page.locator('[data-testid="stAudioInput"]').get_by_role("button", name="Play").count() > 0
    page.get_by_role("button", name="Classify recording").click()
    page.wait_for_selector(".vd-verdict", timeout=120000)
    settle(page)
    verdict = text_of(page, ".vd-verdict")
    shot(page, f"live_mic_classified_{tag}_desktop")
    check(results, f"microphone ({tag} audio via fake device): record -> play-back control -> classify", "verdict" in verdict.lower() and play_ok,
          verdict.replace("\n", " ")[:200])
    # re-record: reset clears the recording and the classify button disappears
    mic = page.locator('[data-testid="stAudioInput"]')
    mic.get_by_role("button", name="Clear recording: Record a voice sample").click()
    settle(page)
    check(results, f"microphone ({tag}): clearing the recording allows re-recording",
          mic.get_by_role("button", name="Record", exact=True).count() > 0 and page.get_by_role("button", name="Classify recording").count() == 0)
    browser.close()
    return verdict


def run_mic_denied(pw, base, port, rec, results, headed: bool):
    browser = pw.chromium.launch(channel="chrome", headless=not headed, args=["--use-fake-device-for-media-stream"])
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})  # microphone permission NOT granted
    page = ctx.new_page()
    rec.attach(page, port)
    page.goto(f"{base}/live")
    settle(page)
    page.get_by_role("tab", name="Record with microphone").click()
    page.get_by_role("button", name="Record", exact=True).click()
    try:
        page.get_by_text("This app would like to use your microphone").first.wait_for(timeout=20000)
        ok = True
    except Exception:
        ok = False
    shot(page, "live_mic_permission_denied_desktop")
    page.get_by_text("Microphone not working?").click()
    help_ok = page.get_by_text("secure context").count() > 0
    check(results, "microphone permission denied: widget explains how to allow access + help expander", ok and help_ok)
    browser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright
    tmp = Path(tempfile.mkdtemp(prefix="vd_e2e_"))
    results, extra = [], {}
    proc = start_server(args.port)
    base = f"http://localhost:{args.port}"
    rec = Recorder()
    try:
        media = make_media(tmp)
        with sync_playwright() as pw:
            browser = pw.chromium.launch(channel="chrome", headless=not args.headed)
            for name, fn in (("pages", lambda: run_pages(browser, base, args.port, rec, results)),
                             ("controls", lambda: run_controls(browser, base, args.port, rec, results, tmp)),
                             ("upload", lambda: extra.__setitem__("upload_verdicts", run_upload(browser, base, args.port, rec, results, media)))):
                try:
                    fn()
                except Exception as exc:
                    check(results, f"scenario '{name}' completed", False, f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
            browser.close()
            for tag in ("human", "ai"):
                try:
                    extra[f"mic_verdict_{tag}"] = run_mic(pw, base, args.port, rec, results, media[f"mic_{tag}"], tag, args.headed)
                except Exception as exc:
                    check(results, f"scenario 'microphone {tag}' completed", False, f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
            try:
                run_mic_denied(pw, base, args.port, rec, results, args.headed)
            except Exception as exc:
                check(results, "scenario 'microphone denied' completed", False, f"{type(exc).__name__}: {exc}")
                traceback.print_exc()
        extra["test_media_sources"] = media["_sources"]
    finally:
        proc.terminate()
        shutil.rmtree(tmp, ignore_errors=True)
    passed = sum(r["pass"] for r in results)
    summary = {"passed": passed, "failed": len(results) - passed, "checks": results,
               "console_errors": rec.console, "page_errors": rec.page_errors, "failed_requests": rec.failed,
               "external_requests": sorted(set(rec.external)), "http_errors": rec.http_errors, **extra,
               "browser": "Google Chrome (Playwright channel='chrome')", "screenshots_dir": str(OUT.relative_to(ROOT))}
    RESULTS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{passed}/{len(results)} checks passed; console errors {len(rec.console)}, page errors {len(rec.page_errors)}, "
          f"failed requests {len(rec.failed)}, external requests {len(set(rec.external))}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
