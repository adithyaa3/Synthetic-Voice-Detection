# Presentation guide

A 10-15 minute walkthrough of the app (`streamlit run app/streamlit_app.py`), a live-demo script, and
plain-language explanations for the audience. Numbers are deliberately not repeated here: read them
off the app (every page loads them from `reports/`), so the talk always matches the current run.

## Before the talk (checklist)

1. `python -m pytest` passes and `python scripts/e2e_check.py` reports all checks passed.
2. Open the app at `http://localhost:8501` (localhost is needed for the microphone).
3. Click *Record* once on the Live prediction page and allow the microphone, so the permission prompt
   doesn't appear mid-demo.
4. Have two files ready: a human clip and a synthetic clip that the model has **not** seen. The test
   partition is listed in `data/manifest.csv`, column `partition = test`. A short WAV excerpt of a
   test AI file works well because the original AI files are larger than the 50 MB upload limit.
5. Quiet room, laptop microphone not muted.

## Page by page

| Page | What to show | One-sentence message |
|---|---|---|
| **Overview** | The four tiles, then the "original 100 % was leakage" chart | "The first model looked perfect; it wasn't. Here is what an honest evaluation says." |
| **Dataset explorer** | Class balance, the adequacy table, then play one human and one AI example and compare the mel-spectrograms | "Our own AI data is 8 recordings from one tool, so we added a licensed public corpus and report both." |
| **Feature insights** | Move the MFCC slider; switch t-SNE / PCA; the group-importance bars | "The features that matter most and how much each group contributes under honest validation." |
| **Model comparison** | Switch between the three data settings; point at the error bars; open *Why this model* | "The choice was made by a rule fixed in advance; simpler models win ties." |
| **Evaluation** | The test table, ROC/PR tabs, then drag the threshold slider and watch false alarms vs misses | "A threshold is a policy decision: we weight false accusations twice as heavily as misses." |
| **Robustness & generalisation** | Noise / MP3 / microphone-like rows; the *Leave one generator out* tab; cross-dataset | "It works well on generators it has seen and much less well on ones it hasn't. That is the main limitation." |
| **Error analysis** | Filter to confident mistakes, click a row and listen | "Here is what fools it." |
| **Live prediction** | Upload demo, then microphone demo (script below) | "Same pipeline as training; the output is a probability, not proof." |
| **About & limitations** | Licences, limitations, ethics | "What this must never be used for." |

## Live demo script

**Upload (about 1 minute)**
1. *Live prediction* -> *Upload a file* -> drop the human test clip. Read out the verdict, the
   calibrated probability and where it sits relative to the threshold and the grey inconclusive band.
2. Drop the synthetic test excerpt. Point at the per-window probability bars: long audio is scored in
   several 4 s windows and averaged.
3. Drop a non-audio file renamed to `.wav`: a friendly "couldn't be decoded" message appears instead
   of a crash. (A `.txt` file is rejected by the uploader itself.)

**Microphone (about 1-2 minutes): a deliberate demonstration of domain shift**
1. Say upfront: "The model was never trained on laptop-microphone audio, and our robustness tests predict
   it will often call a real voice AI-generated. Let's see." The warning on the tab shows the measured figure.
2. *Record with microphone* -> press the microphone icon, speak naturally for 5-8 seconds, press stop.
3. Press play to listen back; record again (or use the clear icon above the recorder) if needed.
4. Press **Classify recording**. Whatever the verdict, connect it to the evidence: the *Robustness &
   generalisation* page (mic-like and noise rows, switch between deployed and pre-registered models) and the
   robustness study on the *Model comparison* page, where noise + microphone augmentation fixes much of this
   at a cost in clean accuracy.
5. Optional: play a synthetic clip from a phone speaker into the microphone. Re-recording through a speaker
   and room is a real-world attack and a known weakness.

If the microphone is blocked: open *Microphone not working?* on the same tab (secure context,
browser site permissions, Windows privacy settings), or fall back to the upload demo.

## Explaining the key ideas simply

* **ROC-AUC:** pick one random AI clip and one random human clip; the AUC is the chance that the model
  gives the AI clip the higher "synthetic" score. 0.5 = coin flip, 1.0 = always right. It doesn't
  depend on the threshold.
* **PR-AUC (average precision):** of the clips the model flags as AI, how many really are, averaged
  over all thresholds. More informative than ROC-AUC when one class is rare.
* **F1:** a single number that balances precision ("when it says AI, is it right?") and recall ("of all
  AI clips, how many did it catch?"). It is the harmonic mean, so it is only high if both are.
* **Balanced accuracy:** the average of the hit rate on AI clips and on human clips, so a model can't
  look good by always guessing the bigger class (on the original, imbalanced split that trick alone
  already looked impressive; see the Overview chart).
* **Cross-validation:** split the development data into 5 parts; train on 4 and test on the 5th, five
  times; report the average and the spread. "Nested" means the hyperparameters are chosen inside the
  training part each time, so the test part never influences any choice.
* **Leakage:** when information about the test data sneaks into training. Here every AI file had 6
  siblings, the *same* recording re-voiced into other voices, so a random split put near-copies in
  training and test, and the model only had to recognise recordings it had already heard. Group-aware
  splitting keeps all siblings on one side.
* **Calibration:** a calibrated "0.8" should mean "about 80 % of clips with this score are synthetic".
  The reliability curve on the Evaluation page shows how close we get.
* **Why the RBF kernel was (or wasn't) justified:** an RBF SVM can draw curved boundaries; a linear model
  can't. The question is whether that flexibility helps on unseen groups. On the *Model comparison*
  page, compare the tuned RBF SVM with logistic regression / linear SVM under the same folds, and look
  at the tie analysis in *Why this model*. The RBF heatmap shows how sensitive it is to C and gamma.
  The original settings (C=10, gamma='scale') are included as their own row.
* **Inconclusive:** scores close to the threshold are shown as inconclusive rather than forced into a
  label, which avoids confident-sounding verdicts on borderline audio.

## Questions to expect

* *"Why not just report the 100 %?"* Because it measured memory of near-duplicate recordings, not
  detection. The leave-one-source-out number is what you'd expect on a new recording.
* *"Would it catch the newest voice-cloning tools?"* Unknown and probably not reliably: see the
  leave-one-generator-out results. More diverse synthetic data is the top item in future work.
* *"Can I use this as evidence?"* No. It is a screening aid whose output is a probability.
