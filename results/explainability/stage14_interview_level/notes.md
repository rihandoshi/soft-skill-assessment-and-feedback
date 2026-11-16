# Stage 14 notes: the interview-level layer (Phase 5, 16 Nov 2026)

**Outcome: done.**
- The `interview_summary` block exists for all 138 MIT interviews (`interview_summaries.jsonl`).
- `analyze_interview.py` builds the same block for a new recording, against `interview_cuts.json`.
- Every trend and latency statement the LLM may see passed its test.

All tests were written down before they ran (`preregistration.md`). The generated numbers are in `report.md`. Everything runs on the CPU, about 1 minute in total.

## What the summary contains

| Field | Rule |
|---|---|
| `level` per dimension | The interview's mean index as a percentile among the other students' interviews (cuts 20 / 80). New recordings: against the 138 MIT interview means (`live_interview`) or RecruitView train speakers' means (`recorded_answer`) |
| `say` | One sentence for the clear ends only: "hesitated a lot across the interview", "the voice sounded quite flat throughout the interview", and so on |
| `point_at` | Hesitation: the most and the least hesitant answer. Expressiveness: the flattest and the liveliest |
| `feedback_at` | `interview` for expressiveness (it is mostly a property of the person: section 6.2), `answer` for hesitation |
| `trend`, `trend_use` | See the trends section below |
| `hesitation.slow_to_start` | Level of the interview's median response latency. `say` only for `high`: "often took a while before starting to answer". It sits under hesitation because latency is not a skill of its own: it is hesitation before the answer starts (changed 16 Nov) |
| `internal` | Mean index, interview percentile, the trend size and its permutation p, median latency. Never shown to the LLM |

**Consistency check.** The interview level separates the ratings as expected. Mean `NoFillers` is 5.06 / 4.14 / 3.26 for low / normal / high hesitation, and mean `EngagingTone` is 3.82 / 4.90 / 5.59 for low / normal / high expressiveness. 28–29 interviews sit at each end.

## Response latency (test L): supported, used

**Definition:** the time from the interviewer's last word to the candidate's first answer word.
- Measured with the Stage 7 Whisper words and the Turker speaker alignment.
- Values outside 0–10 s are dropped.
- The interview value is the median over its answers.

**Coverage:** 663 of 674 answers, 137 interviews. Median 1.2 s per answer.

| Rating | Raw | After length and quality | Also controlling hesitation |
|---|---|---|---|
| `Overall` | −0.36 [−0.50, −0.20] | −0.38 [−0.52, −0.23] | −0.35 [−0.49, −0.18] |
| `NotAwkward` | −0.25 [−0.42, −0.08] | −0.27 [−0.44, −0.09] | −0.23 [−0.38, −0.05] |

- Both pass, and the relation survives controlling hesitation, so latency is its own observation, not hesitation again.
- Its relation with hesitation is only +0.17.
- **Only the slow end is mentioned.** A short latency is not praised: no test supports that.
- **Stage 7's `response_latency_s` was not usable.** It is the silence at the start of the clip, which was cut 0.2 s before the first word.

## Trends (tests T1 and T2)

The trend is the last two answers minus the first two, on an index adjusted for the question. Without the adjustment, Q1, which runs about 5 points lower, would look like a trend.

| Dimension | T1: larger than reordering chance? | T2: related to ratings beyond the mean? | Use |
|---|---|---|---|
| Hesitation | **Fails.** 7.5% of interviews have a trend against a null p95 of 8.2% | Contradicted (see below) | `not_used`. No trend word |
| Expressiveness | **Passes.** 8.3% against a null p95 of 6.8% | No support (`EngagingTone` +0.04, `Excited` −0.02, CIs include 0) | `description`. 11 of 133 interviews get a trend word (6 livelier, 5 flatter), stated as a change between the first and last answers, never as how the interview came across |

**What this means:**
- The plan's example statement, "hesitation fell across the interview", cannot be made: within one interview, hesitation changes no more than random reordering of the answers would produce.
- Expressiveness changes slightly more than chance, but the change does not affect the ratings once the average is known.

**The hesitation T2 result points the other way from what was expected.**
- Given the mean, interviews that grew *more* hesitant towards the end were rated *better* on `NoFillers` (+0.21) and `Overall` (+0.24).
- One reading is that raters weigh the first answers more: fluent first answers make a better impression.
- This was not pre-registered as a hypothesis. It is not used, and is noted only for the team.

## Ridge baseline for fusion (descriptive)

**Inputs:** the per-interview aggregates. Out-of-fold, grouped by student.

| Rating | Spearman |
|---|---|
| `Overall` | 0.60 [0.44, 0.72] |
| `RecommendHiring` | 0.61 |
| `EngagingTone` | 0.68 |
| `Excited` | 0.69 |
| `NoFillers` | 0.58 |
| `NotAwkward` | 0.47 |

- For `Overall`, this is above the best single feature (expressiveness mean, 0.52, chosen in-sample).
- Near zero for `Calm` and `Authentic`.
- The out-of-fold predictions are in `tables/ridge_baseline_oof.csv`: an audio-only interview baseline that a fusion model has to beat.

## New recordings

`analyze_interview.py` now writes this block instead of the provisional counts.
- **Trends and latency** are switched off in the `recorded_answer` setting: they were tested on live interviews only.
- **MIT P1 run end to end from the raw file** (89 s, peak 1.86 GB GPU):
  - Levels agree with the dataset summary: hesitation normal, expressiveness high.
  - Latency comes out as 1.4 s (normal) against 2.3 s (high) from the dataset version.
  - The cause is that the unvalidated segmentation starts Q1 about 2 s earlier (Phase 4, test 2). **Latency on new recordings is only as good as the speaker turns.**

## Open

- **Hand-over to Phase 7.** The LLM should use `say`, `trend_say` and `point_at`, and nothing under `internal`. Expressiveness should be spoken about once per interview, using the interview `say` rather than repeating each answer's.
- **Latency on new recordings** needs the segmentation fix (pyannote or the shared upstream) before it can be trusted.

## Re-run

- `python scripts/explainability/interview_summary.py`: about 1 min on the CPU. Run it after `fit_answer_levels.py`.
- It rewrites `interview_cuts.json`, which `analyze_interview.py` reads.
