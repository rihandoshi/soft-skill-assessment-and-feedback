# Stage 8 notes: per-answer detectors (closed in Phase 1, 13 Nov 2026)

What Stage 8 delivers, what changed in Phase 1, and what is still open. The numbers come from `report.md` (generated) and the tables named below.

## What a level is

Each answer gets one level per audio dimension: `low`, `normal` or `high`. To get it:

1. Each component is turned into a percentile among RecruitView train answers (unique recordings).
2. The component percentiles are averaged.
3. The average is turned into a percentile again.
4. That percentile is cut at p20 and p80.

Only `low` and `high` reach the feedback. Everything needed to score a new answer is in `level_cuts.json`.

## Phase 1 changes

| Dimension | Before | After |
|---|---|---|
| Hesitation (the "fluency" skill) | 5 components, 2 of them CrisperWhisper word counts (fillers, restarts) | Audio only. Weighted mean of percentiles: vocametrix interjection probability (1/6), vocametrix repetition + block probability (1/6), Vox-Profile speech-flow "disfluent" probability (1/3), long mid-sentence pauses per minute (1/3). The two classifiers are averaged and keep the 2/3 share the single classifier had. The weights were fixed before the combination was tested on MIT. The CrisperWhisper columns stay in `answer_levels.csv` for the text branch |
| Expressiveness | Pitch variability and monotone share (Praat) | Vocal activation from the audeering emotion model, (arousal + dominance) / 2, adjusted for the speaker's typical pitch. Pitch variability is now only the reason in words |
| Length | A level, compared within question type | Removed. The text branch judges whether an answer ran long. `speech_span_s` stays in the table and in `internal` of the evidence JSON |
| Evidence JSON | Verbatim excerpt with "uh / um" for hesitation | No words at all. Each observation has a level, reasons in words, and a `where` time span from the model windows. Adds `setting`, `modality` and `internal` |

## Voice-pitch adjustment (expressiveness)

Activation rises with how high a person's voice is (Spearman with the speaker's median F0: +0.49 on RecruitView, +0.42 on MIT). Left alone, low voices would be called "flat" far more often:

| | Low voices | Middle | High voices |
|---|---|---|---|
| "Flat" share before the adjustment, RecruitView | 39% | 16% | 9% |
| "Flat" share before the adjustment, MIT | 37% | 10% | 8% |

The adjustment works like this:
- Activation is regressed on the speaker's median F0 in Hz, on the reference answers only.
- The fitted line is `0.2304 + 0.001023 × F0`.
- The residual is what gets percentiled.

After the adjustment (`tables/flat_by_voice_pitch.csv`, quality-gated answers excluded):

| | Low voices | Middle | High voices |
|---|---|---|---|
| "Flat" share, RecruitView | 23% | 10% | 18% |
| "Flat" share, MIT | 23% | 21% | 31% |

Other shapes were tried, all fitted on RecruitView train:

| Adjustment | RecruitView "flat" by tercile | MIT "flat" by tercile | MIT `EngagingTone`, partial (quality, length, F0) |
|---|---|---|---|
| None | 39 / 16 / 9% | 37 / 10 / 8% | +0.65 |
| **Linear in Hz (used)** | 27 / 14 / 22% | 23 / 21 / 31% | **+0.62** |
| Linear in log Hz | 25 / 15 / 23% | 18 / 24 / 31% | +0.60 |
| Quadratic or cubic in log Hz | 22 / 18–21 / 20–22% | 14–16 / 25–28 / 29–30% | +0.53 to +0.55 |
| Percentile within 3 or 5 pitch bands | 21–22 / 19–22 / 20–22% | 11–16 / 31–34 / 27–29% | +0.53 to +0.58 |

The tercile shares in this table include all answers, so they differ slightly from the gated table above.

The more flexible shapes look perfect on RecruitView but over-correct on MIT and weaken the relation with the ratings. The reason is that MIT's own slope is about two-thirds of RecruitView's (0.00067 against 0.00102 per Hz). With the slope fitted on MIT itself, MIT's "flat" shares become 26% / 15% / 19% and the `EngagingTone` relation stays at +0.69.

So the slope belongs to the reference population. Phase 2 (one reference per setting) fits it per setting. Until then, high voices in live interviews are called "flat" somewhat more often than low ones.

## MIT validation (interview level, 138 interviews, 69 students)

All in `report.md`. Partial means controlling for answer length and recording quality, plus the speaker's F0 for expressiveness.

| Test | ρ | Partial [95% CI] | Verdict |
|---|---|---|---|
| Hesitation index vs `NoFillers` | −0.63 | −0.58 [−0.70, −0.43] | supported |
| Hesitation index vs Turker filler rate | +0.49 | +0.58 [+0.42, +0.69] | supported |
| Hesitation index vs `Overall` | −0.30 | −0.44 [−0.60, −0.27] | supported |
| Speech-flow model alone vs `NoFillers` | −0.60 | −0.56 [−0.69, −0.39] | supported |
| Speech-flow model alone vs Turker filler rate | +0.57 | +0.60 [+0.44, +0.72] | supported |
| Odyssey activation vs `EngagingTone` | +0.72 | +0.65 [+0.50, +0.77] | supported |
| Odyssey activation vs `Excited` | +0.73 | +0.67 [+0.53, +0.79] | supported |
| Expressiveness index vs `EngagingTone` | +0.63 | +0.63 [+0.50, +0.73] | supported |
| Expressiveness index vs `Excited` | +0.63 | +0.66 [+0.52, +0.76] | supported |
| Share of answers called "flat" vs `EngagingTone` | −0.55 | −0.49 [−0.62, −0.34] | supported |
| Confidence model vs `Overall` | −0.60 | −0.58 [−0.72, −0.43] | contradicted (unchanged) |

The audio-only index from the three components (before speech-flow) was as strong as the old mixed one: −0.59 against `NoFillers`. Adding the speech-flow model takes it to −0.63, so the audio branch no longer needs word counts for fluency.

## Level rates (RecruitView reference)

| | RecruitView | MIT |
|---|---|---|
| Hesitation `high` | 21% | 45% (56% before speech-flow joined, 62% in the old mixed index) |
| Expressiveness `low` | 20% | 25% |

MIT hesitation is still shifted: people hesitate more in a live conversation. Phase 2 fixes this.

## Evidence

`stage9_feedback/evidence_v2.jsonl`:
- Answers with no observation: 42%.
- Answers with one observation: 42%.
- Answers with two observations: 16%.
- Every observation has a time span.

The two `where` spans are:
- **Hesitation:** the 3 consecutive classifier windows (about 6 s) with the highest mean non-fluent probability, averaged over the two classifiers.
- **Expressiveness:** the 3 consecutive emotion-model windows (about 8 s) with the lowest mean activation, or the highest for "lively".

## Phase 1 GPU runs

Both runs were on an RTX A3000 laptop GPU (6 GB), with PyTorch capped at 80% of its memory.

- **Odyssey emotion model on MIT** (`run_ser_windows.py --model odyssey --dataset mit --batch 8`):
  - 674 answers in 11 min, peak 1.9 GB.
  - Activation agrees with audeering at +0.86 on MIT answers (0.90–0.92 on RecruitView).
  - It passes the same expressiveness tests as audeering.
  - audeering stays the detector; Odyssey is the confirmation that the result is not one model's quirk.
- **Vox-Profile speech-flow model** (`run_speech_flow.py`):
  - What it is: `tiantiaf/wavlm-large-speech-flow`, revision `aea5a480`, OpenRAIL licence, no commercial use.
  - How it was loaded: re-implemented from the repo's `wavlm_fluency.py`, which was read, not run. The LoRA updates are folded into transformers' `WavLMModel`. Every weight in the file is used and none is missing.
  - The run: all 2,685 answers on the same 3-s windows as the vocametrix classifier, in 27 min, peak 1.7 GB.
  - Result: it passed its pre-registered tests (above), so it joined the fluency index.
  - Median "disfluent" probability: 0.32 on RecruitView, 0.38 on MIT.

## Open

- Listening check: `listening_check.csv` lists 60 answers (10 RecruitView + 5 MIT per skill and end). Fill in `heard_ok` and `problem` while listening.
- Phase 2: one reference per setting, including the voice-pitch slope.
