# Audio branch outputs for fusion

Everything the audio branch gives the fusion model, one row per answer. Built by `scripts/explainability/export_for_fusion.py` on 16 Nov 2026.

## Files

| File | What |
|---|---|
| `audio_answers.csv` | One row per answer: keys, audio score predictions, feedback dimensions, context. 2,011 RecruitView + 674 MIT answers |
| `audio_embeddings_wavlm_l7.npz` | `answer_id` (2,685) and `embedding` (2,685 × 1,536, float32). WavLM-base-plus layer 7, mean and std pooled over the answer. Same row order as `audio_answers.csv` |
| `evidence.jsonl` | One JSON per answer: what the LLM may say about the voice. The format is in the main README, section 11 |
| `score_model_metrics.csv` | Per target: Ridge alpha, out-of-fold Spearman on train, Spearman on val and test |

Load it like this:

```python
import numpy as np, pandas as pd
a = pd.read_csv("audio_answers.csv", dtype={"answer_id": str, "person_id": str, "duplicate_of": str})
z = np.load("audio_embeddings_wavlm_l7.npz")
emb = pd.DataFrame(z["embedding"], index=z["answer_id"])     # emb.loc[a["answer_id"]]
```

## Keys

| Column | Meaning |
|---|---|
| `answer_id` | RecruitView: the dataset `id` (string, keep the leading zeros, e.g. `0008`). MIT: `MIT_<interview>_<Q1..Q5>` |
| `dataset` | `recruitview` or `mit` |
| `split` | RecruitView: `train` / `val` / `test`, exactly as in `Datasets/Split Dataset/`. MIT: `mit` |
| `person_id` | RecruitView: `user_no`. MIT: the student (`MIT<n>`), the same for their pre and post interviews |
| `question_id` | RecruitView `question_id`; MIT `Q1`–`Q5` |
| `is_duplicate`, `duplicate_of` | RecruitView has 106 byte-identical recordings under different ids, rated separately. Duplicates get identical audio outputs. `duplicate_of` is the id of the copy that is kept |
| `interview_id`, `session` | MIT only: the interview (`P1`, `PP1`, ...) and `pre` / `post` coaching |

## Audio score predictions: `pred_<target>`

The 12 RecruitView targets: `openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`, `overall_personality`, `interview_score`, `answer_score`, `speaking_skills`, `confidence_score`, `facial_expression`, `overall_performance`. They are on the same scale as the labels.

**Model:** WavLM-base-plus layer 7 embedding (the file above) and a Ridge regression per target. It was fitted on train people only, using unique recordings, with alpha chosen by 5-fold cross-validation grouped by person. This was the best audio model in the earlier repeated cross-validation.

**Leakage rule:**
- **Train rows are out-of-fold.** Each train answer is predicted by a model that never saw that person (5-fold GroupKFold over `person_id`). You can train a late-fusion model on the train rows directly.
- **Val, test and MIT rows** come from one model fitted on all train people.
- Don't refit anything on val or test.

**Quality** (Spearman with the label, from `score_model_metrics.csv`):

| | Range over the 12 targets |
|---|---|
| Train, out-of-fold | 0.35–0.55 (slightly optimistic: alpha was picked on these folds) |
| Val | 0.24–0.43 |
| Test | 0.25–0.55 |

**Read this before using them:**
1. **The 12 predictions are essentially one signal.**
   - The first principal component holds 94% of their variance.
   - The median correlation between two predictions is 0.97 (`pred_neuroticism` is the same signal inverted).
   - Treat them as one audio input in fusion, for example `pred_overall_performance`, not as 12 independent ones.
2. **That signal follows answer length.**
   - It correlates about −0.6 with `speech_span_s`: shorter answers are predicted higher.
   - If the text branch also carries length, the two branches overlap on it.
3. **`pred_confidence_score` must not be used as a confidence measure outside RecruitView.**
   - On MIT, confidence models of this kind run *against* the human ratings of overall performance (Phase 3).
   - Use it as a fusion input on RecruitView only, never in feedback.
4. **On MIT the predictions are out of domain** (live interviews, far-field microphone, US students). They are included for completeness. MIT has no RecruitView labels to check them against.

## Feedback dimensions (validated on MIT)

These are what the feedback is based on. They do not use any RecruitView label, so they need no cross-validation and are the same for every split.

| Column | Meaning |
|---|---|
| `setting` | `recorded_answer` (RecruitView, compared with RecruitView train answers) or `live_interview` (MIT, each student compared with the other students) |
| `hesitation_index` | 0–100, higher = more hesitant: filled pauses, stumbles and long mid-sentence pauses, compared with other people's answers. Related to MIT `NoFillers` at −0.58 after controls |
| `hesitation_level` | `low` (index ≤ 20), `normal`, `high` (≥ 80), `missing` |
| `expressiveness_index` | 0–100, higher = more lively vocal activation, adjusted for the speaker's voice pitch. Related to MIT `EngagingTone` at +0.65 after controls |
| `expressiveness_level` | as for hesitation (`low` = flat, `high` = lively) |
| `*_withheld` | True when the recording quality is too low (RecruitView only). The level is then not used in feedback |
| `*_reasons` | Plain-word reasons behind a `high` hesitation or a `low` / `high` expressiveness level (may be empty) |
| `p_interjection`, `p_repetition_block`, `sf_p_disfluent`, `long_mid_pause_rate_per_min` | The four hesitation components (classifier probabilities, pauses per minute) |
| `activation`, `activation_adj` | Vocal activation (mean of arousal and dominance, audeering model) before and after the voice-pitch adjustment |
| `pitch_std_st` | Pitch variability in semitones |

**Missing values:** 46 answers (42 RecruitView, 4 MIT) have no index, because they are too short or have too little speech for the classifiers.

**How they relate to the score predictions** (RecruitView):
- Hesitation is nearly independent of them (ρ 0.02 with `pred_overall_performance`).
- Expressiveness runs against them (−0.39).

So they add information the score signal does not have. They are not a noisier copy of it.

## Interview level (MIT only, Phase 5)

For interview-level fusion on MIT, `../stage14_interview_level/` has:

| File | What |
|---|---|
| `tables/interview_aggregates.csv` | One row per interview (138): mean index, share low and high, trend per dimension, median response latency, speaking time, and the MIT ratings |
| `tables/ridge_baseline_oof.csv` | Audio-only Ridge predictions of every reliable MIT rating. Out-of-fold, grouped by student, so pre and post stay together. A baseline to beat: 0.60 Spearman with `Overall`, 0.68 with `EngagingTone` |
| `interview_summaries.jsonl` | The `interview_summary` block per interview (what the LLM may say about the interview as a whole) |

The keys are `interview` (`P1`, `PP1`, ...) and `person` (`MIT<n>`), the same as `interview_id` and `person_id` above.

## Context (internal: never shown to a user)

| Column | Meaning |
|---|---|
| `speech_span_s` | Seconds from first to last speech. The text branch judges whether a long answer is a problem |
| `quality_composite` | Recording quality, 0–100 (percentile among RecruitView train recordings). Missing for 22 RecruitView answers with too little speech to measure |
| `acoustic_ok` | Passes the quality gate |
| `speaker_f0_hz` | The speaker's typical pitch. Used only for the expressiveness adjustment; not a feature to judge people on |

## Not included

- **Perceived confidence.** No audio model of it holds up on MIT; it is a fusion-level judgement.
- **Clarity.** It was dropped because the only available model rates Indian-English accents as unclear.
- **Words, fillers and transcripts.** These belong to the text branch (`stage8_detectors/tables/crisper_words.csv` has the verbatim CrisperWhisper words if needed).
