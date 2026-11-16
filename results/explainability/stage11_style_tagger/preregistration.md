# Stage 11 pre-registration: speaking-style tagger (Phase 3, Part A)

Written on 14 Nov 2026, **before the tagger was run on any answer**. The analysis script (`analyze_style_tagger.py`) implements exactly this. Anything added after seeing results is marked as exploratory in the report.

## Model and measurement

- **Model:** `tiantiaf/wavlm-large-voice-quality` (Vox-Profile, revision `5c0df771`, OpenRAIL licence, no commercial use). It gives 25 independent sigmoid tag probabilities per window.
- **Windows:** 10 s with a 5-s hop over the whole answer. A window is kept when at least 50% of it is speech (Silero VAD). An answer shorter than 10 s with at least 3 s of audio is scored as one window. This stays inside the model's 3–15-s training range.
- **Per answer:** the mean probability of each tag over the kept windows.

## Values (fixed now)

| Value | Formula (per-answer tag means) |
|---|---|
| `style_confidence` | mean(authoritative, flowing, enunciated) − mean(hesitant, soft, hushed, stammering) |
| `style_clarity` | mean(crisp, enunciated) − slurred |
| `tag_hesitant` | hesitant |
| `tag_monotone` | monotone |

## Step 4: what do the values track? (descriptive, run before the tests)

1. **Spearman across answers** with:
   - speaking time (`speech_span_s`)
   - the recording-quality composite
   - vocal activation
   - the speaker's typical pitch (F0)
   - the fluency index
2. **Speaker share:** the share of variance that lies between speakers rather than within them.
3. **Tag rates by dataset:** RecruitView (mostly Indian English) against MIT (US students).

## Tests on MIT (interview level, 138 interviews, 69 students)

How every test is run:
- Measures are averaged over the interview's answers and correlated with the mean of the 9 raters.
- Spearman, with 95% CIs from 2,000 bootstrap resamples over students.
- "Partial" means controlling for the listed variables, with ranks residualised (`corr_pair` in `validate_categories_mit.py`).
- **Pass** means both the raw and the partial CI exclude 0 in the expected direction.

| # | Measure | Label (ICC(2,k)) | Expected | Controls |
|---|---|---|---|---|
| C1 | `style_confidence` | `Overall` (0.74) | + | length, quality, activation |
| C2 | `style_confidence` | `RecommendHiring` (0.76) | + | length, quality, activation |
| C3 | `style_confidence` | `NotAwkward` (0.68) | + | length, quality, activation |
| C4 | `tag_hesitant` | `NoFillers` (0.78) | − | length, quality |
| E1 | `tag_monotone` | `EngagingTone` (0.86) | − | length, quality, speaker F0 |

The controls follow the README:
- "Length" is the answer duration and "quality" is the quality composite.
- "Activation" is the audeering vocal activation, (arousal + dominance) / 2. It is a control so that a confidence value which only re-measures expressiveness does not pass.

**The confidence experiment passes** when C1, C2 and C3 all pass. C4 and E1 are supporting checks: they say whether the tags mean what their names say. They do not decide anything on their own.

**Halo check (reported, not a pass condition):** the partial correlation of `style_confidence` with every reliable MIT label (ICC ≥ 0.5). If it is as strong on unrelated labels (`Smiled`, `EyeContact`, `Friendly`) as on C1–C3, the report says so.

## Clarity (no MIT label exists)

**Accent check.** Compute where the median RecruitView answer's `style_clarity` sits within the MIT distribution.
- If it is below MIT's 25th percentile, the tagger rates the Indian-English answers as less clear across the board. That is treated as an accent effect, and clarity is dropped.
- The two datasets also differ in microphone and room. The check is therefore repeated on the best-quality third of each dataset (quality composite). An effect that holds there counts as accent.

**RecruitView relation.** Between people (person means over unique recordings, all splits), take the Spearman between `style_clarity` and `speaking_skills`. It must not be negative: point estimate ≥ 0.

**Listening.** The 10 highest and 10 lowest answers go into a listening list.

**Clarity passes** when the accent check finds no effect, the RecruitView relation is not negative, and the listening check agrees. Until someone has listened, the best possible state is "candidate".

## What happens next

- **Confidence passes:** it becomes a candidate skill and gets levels in both settings (Phase 2 machinery). It reaches the feedback only after a listening check.
- **Confidence fails:** Part B (retrain on RecruitView with confounds removed) starts with step B1 (CPU).
- **Clarity fails:** clarity is dropped from the audio branch.

---

# Addendum: Part B pre-registration

Written on 14 Nov 2026, after Part A failed and **before any Part B model was trained**. Script: `run_confidence_partb.py`.

## Target

RecruitView `confidence_score` with its known confounds removed.

1. On train people only (unique recordings, duplicate copies averaged), fit OLS: `confidence_score ~ speech_span_s + log(speech_span_s) + quality_composite`.
2. The target is the residual.
3. The same coefficients give the residual target for the val and test answers.

## Variants (cheapest first)

| # | Features | Model |
|---|---|---|
| B1 | WavLM-base-plus layer 7, mean + std pooled. This is the legacy cache that the failed Stage 8 detector used | Ridge, alpha chosen by grouped CV |
| B2 | Same features | Logistic regression on the top and bottom 30% of the residual (middle dropped), C chosen by grouped CV. Its decision value is the score |
| B3 | GPU, only if B1 and B2 fail: pooled embeddings from WavLM-large and from the audeering emotion model, same linear heads | Same as B1 and B2 |

Fine-tuning is not attempted on the current 6-GB GPU.

## How each variant is scored

**RecruitView:**
- 5-fold GroupKFold over participants on train.
- The Spearman between the out-of-fold score and the residual target, over all held-out train answers. For B2 that includes the middle 40% that was never trained on.
- Validation and test Spearman are also reported, from one model fitted on all train people.

**MIT:**
- One model fitted on all train people scores every MIT answer.
- Scores are averaged per interview and compared with the 9-rater mean.
- Tests B-C1 to B-C3 are `Overall`, `RecommendHiring` and `NotAwkward`, expected positive.
- Partial correlations control for answer length, recording quality and vocal activation, exactly as Part A's C1 to C3.
- 2,000 bootstrap resamples over students.

## Pass (from the README, made exact)

A variant passes when **both** hold:
1. **RecruitView:** grouped-CV Spearman ≥ 0.30.
2. **MIT:** none of B-C1 to B-C3 is "contradicted". That means for each label the raw and the partial 95% CI upper bounds are above 0.

A variant that passes **and** has B-C1 to B-C3 all "supported" (both CIs above 0) is promoted to a feedback candidate.

## If every variant fails

The audio branch reports no confidence level (README section 6.3, fallback).
