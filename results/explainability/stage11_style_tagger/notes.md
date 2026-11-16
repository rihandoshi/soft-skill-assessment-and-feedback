# Stage 11 notes: vocal confidence and clarity (Phase 3, 14 Nov 2026)

**Outcome: neither skill survives.**
- **Confidence:** the audio branch reports no confidence level. The README fallback applies.
- **Clarity:** dropped. The tagger rates accent, not clarity.
- **The tagger's "monotone" tag:** a weak second opinion for expressiveness.

Everything was pre-registered before it was run (`preregistration.md`, including the Part B addendum written after Part A failed). The generated numbers are in `report.md` (Part A), `partb_report.md` (B1, B2) and `partb_b3_report.md` (B3).

## Part A: Vox-Profile speaking-style tagger

### How it was run

- **Model:** `tiantiaf/wavlm-large-voice-quality`, revision `5c0df771`, OpenRAIL, no commercial use. It gives 25 independent tags trained on ParaSpeechCaps.
- **Loading:** weights only, through `voxprofile_models.py`. That module is shared with the speech-flow model, and its refactor reproduces the cached speech-flow outputs exactly (max difference 0.0).
- **Windows:** 10 s with a 5-s hop, keeping windows with at least 50% speech. Tag means are taken per answer.
- **Coverage:** 2,685 answers in 29 min, peak 2.05 GB on the 6-GB laptop GPU. 68 answers have no usable window (under 3 s or too little speech).

### Results (MIT, interview level)

| Test | Result | Verdict |
|---|---|---|
| C1 `style_confidence` vs `Overall` | +0.10, partial +0.06 [−0.14, +0.24] | no support |
| C2 vs `RecommendHiring` | +0.07, partial +0.02 | no support |
| C3 vs `NotAwkward` | +0.12, partial +0.08 | no support |
| C4 "hesitant" tag vs `NoFillers` (expected −) | +0.12 | no support (wrong sign) |
| E1 "monotone" tag vs `EngagingTone` (expected −) | −0.25, partial −0.24 [−0.42, −0.05] | supported |

### What the values track

From `tables/tracks.csv`:
- `style_confidence` follows recording quality (+0.33 on MIT, +0.50 on RecruitView). On RecruitView it also follows activation (+0.58).
- On MIT it follows *lower* voices: −0.39 with the speaker's pitch.
- It is 80% a property of the speaker.
- Between RecruitView people it runs *against* the experts' `confidence_score` (−0.42, exploratory).

### Clarity: accent effect

| | RecruitView median | MIT median |
|---|---|---|
| "slurred" probability | **0.92** | **0.06** |

The typical RecruitView answer (mostly Indian English) sits at MIT's 0th percentile of clarity. The result is the same in the best-quality third of each dataset, so it is not the microphones.

The tagger was trained mostly on American and British speech, and it treats an Indian-English accent as slurred. Clarity is dropped under the pre-registered rule. The tag must never be used in feedback.

### "Monotone" tag

- It supports the expressiveness skill, but at −0.24 it is far weaker than activation (+0.65).
- It is near zero for most answers, and only 13–18% of its variation lies between speakers.
- It is not added to the index. It stays available as a cached second opinion.

## Part B: retraining on RecruitView with the confounds removed

### Target

`confidence_score` minus what speaking time, its log and recording quality explain, fitted on train people. Those three explain R² = 0.12.

### Results

| Variant | Features | RecruitView grouped-CV (target = residual) | Test | MIT `Overall`, partial | Outcome |
|---|---|---|---|---|---|
| B1 Ridge | WavLM-base-plus L7 (legacy) | 0.327 | 0.284 | −0.43 [−0.57, −0.25] | contradicted |
| B2 Logistic on extremes | same | 0.321 | 0.281 | −0.34 | contradicted |
| B3 Ridge | WavLM-large L18 (layer chosen on train) | 0.339 | 0.314 | −0.42 | contradicted |
| B3 Logistic on extremes | same | 0.330 | 0.320 | −0.46 | contradicted |
| B3 Ridge | audeering emotion model, pooled | 0.318 | 0.301 | −0.45 | contradicted |
| B3 Logistic on extremes | same | 0.318 | 0.315 | −0.48 | contradicted |

- Every variant clears the RecruitView bar of 0.30.
- Every variant is contradicted on all three MIT labels (`Overall`, `RecommendHiring`, `NotAwkward`), even with length, quality and vocal activation controlled.
- Stronger features add little: 0.33 becomes 0.34.
- Fine-tuning was not attempted. The plan ranks it last and expects it to overfit, and it needs the bigger GPU.

### Why it fails (exploratory, after the tests)

- **The retrained score is mostly low vocal energy.** On MIT answers it correlates −0.72 with activation and −0.30 with voice pitch. It even goes *with* more hesitation (+0.47).
- **RecruitView's raters behave the same way.** Between people, `confidence_score` follows lower activation (−0.49) and lower voices (−0.24).
- **So the two sets of raters disagree.** RecruitView's psychologists rated calm, low-energy and lower-pitched delivery as confident. MIT's raters rated the same delivery lower on overall performance and hireability.

Removing length and quality does not change this. The disagreement is about the voice itself.

## Consequences

- **No audio confidence level.** `build_answer_evidence.py` keeps confidence under `not_assessed`, as before.
  - Perceived confidence becomes a fusion-level judgement with video and text.
  - The LLM may link validated behaviours to how they can come across, for example "you hesitated a lot at the start, which can come across as unsure". That is a wording rule (Phase 7), not a model.
- **Clarity is out of the audio branch.**
- **The audio branch's validated skills are fluency and expressiveness.**
- **The result is worth reporting:** two expert-rated datasets disagree about what a confident voice sounds like, and a model trained on one is reliably wrong on the other.

## Not done

- **Listening check** of the tagger extremes: `tables/listening_style.csv`. It no longer changes a decision; it is only useful to hear what "slurred" picks up.
- **Fine-tuning** (Part B step 3, last resort): not run. It would need the 16-GB GPU and is expected to overfit.

## Re-run

- **GPU:**
  - `run_style_tagger.py`: about 29 min, peak 2.05 GB.
  - `extract_partb_embeddings.py --model wavlm_large`: about 28 min, peak 2.05 GB.
  - `--model audeering`: about 14 min, peak 1.5 GB.
  - All are cached and resume from the cache.
- **CPU:**
  - `analyze_style_tagger.py`: about 2 min.
  - `run_confidence_partb.py`: about 3 min.
  - `run_confidence_partb.py --features b3`: about 10 min.
