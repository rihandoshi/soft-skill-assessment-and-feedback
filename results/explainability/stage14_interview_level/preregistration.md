# Stage 14 pre-registration: the interview-level layer (Phase 5)

Written on 16 Nov 2026, before any of the tests below was run. Only the distributions of the measures themselves were looked at beforehand (latency range, question means of the indices); nothing was compared with a rating.

## Data

- **MIT:** 674 answers, 138 interviews, 69 students. These are the `live_interview` levels from `answer_levels.csv`: each student is compared with the other students.
- **Ratings:** the mean of the 9 Turkers. Only labels with ICC(2,k) ≥ 0.5 count.
- **Confidence intervals:** 95% bootstrap over students, 2,000 resamples, seed 0. A student's pre and post interviews are resampled together.
- **Controls:** the same as Stage 8.
  - Mean answer clip duration and recording quality.
  - The speaker's typical pitch as well, for expressiveness measures.

## Measures

1. **Answer order.** Answers are ordered by time in the interview. The order is always Q1 to Q5; this was checked.
2. **Aggregates per interview and dimension** (hesitation, expressiveness):
   - the mean index;
   - the share of answers at `low` and at `high`;
   - the first half against the second half;
   - the strongest and weakest answer.
3. **Question-adjusted index, used only for trends.** Each answer's index minus the mean of its question over all MIT answers. Q1, "about yourself", runs about 5 points lower than the others on both dimensions; without the adjustment that would show up as a trend.
4. **Trend.** The mean question-adjusted index of the last two answers minus that of the first two.
   - Requires at least 4 answers with an index.
   - With 5 answers, the middle one is left out.
5. **Response latency per answer.**
   - The start of the candidate's first answer word minus the end of the last interviewer word before it. The Whisper word times come from Stage 7, and the speakers from the alignment to the Turker transcripts.
   - Values outside 0–10 s are set to missing: above 10 s, the alignment or the turn structure has most likely gone wrong.
   - The interview value is the median over its answers, given only when at least 3 answers have one.
   - Stage 7's `response_latency_s` is **not** used. It is measured inside the clip, which starts 0.2 s before the first word.

## Test L: response latency (plan step 2)

Interview median latency against two ratings:

| Rating | Expected direction |
|---|---|
| `Overall` | negative |
| `NotAwkward` | negative |

- **Supported:** the raw and the partial CI both exclude 0 in the expected direction. This is the Stage 8 `verdict` rule.
- **If supported for at least one of the two:** a second check also controls the interview's mean hesitation index.
  - If that CI still excludes 0, latency is a separate observation and enters the interview summary.
  - If it does not, latency is counted as part of hesitation and stays internal.
- **If not supported:** latency stays an internal number and the LLM never sees it.

## Test T: do trends mean anything? (plan step 3)

**T1, is the trend larger than chance reordering?**
- For each interview, an exact permutation p-value of its trend over all orderings of its own answers (two-sided).
- An interview "has a trend" when p ≤ 0.10 **and** |trend| ≥ 15 index points.
- The observed share of interviews with a trend is compared with 1,000 null runs, in which each interview's answer order is shuffled and the same rule applied.
- **Passes** when the observed share is above the 95th percentile of the null shares.

**T2, does the trend carry rating information beyond the average?**
- The partial Spearman of the trend with the rating, controlling the interview's mean index, length and quality (plus pitch for expressiveness). Expected directions:

| Trend | Rating | Expected direction |
|---|---|---|
| Hesitation | `NoFillers` | negative |
| Hesitation | `Overall` | negative |
| Expressiveness | `EngagingTone` | positive |
| Expressiveness | `Excited` | positive |

- **Passes** when that partial CI excludes 0 in the expected direction for at least one rating of the dimension. Here the raw relation is not required: the question is what the trend adds to the mean.

**Use of the trend words** (`steady`, `improved`, `got worse`):

| T1 | T2 | Use |
|---|---|---|
| fails | — | Trend words are never given to the LLM. The trend stays internal |
| passes | fails | **Description only.** The trend word is given only for interviews that have a trend (T1 rule), worded as a change between the first and last answers. It is never presented as affecting how the interview came across |
| passes | passes | As above, and it may also be used as feedback |

## Overall interview level (plan step 5)

- The interview's mean index becomes a percentile among the interviews of the **other** students: leave-one-student-out, the same as the answer levels.
  - Cuts at 20 and 80.
  - The interview mean is the quantity Stage 8 validated (−0.58 / +0.65 after controls).
- **New recordings** are scored against the quantiles of all 138 MIT interview means (`interview_cuts.json`) for `live_interview`.
  - For `recorded_answer`, against the per-person means of the RecruitView train speakers (about 6 answers each).
- **Answers to point at:**
  - Hesitation: the most and the least hesitant answer.
  - Expressiveness: the flattest and the liveliest.
- Expressiveness feedback is marked `feedback_at: interview`: it is given once for the interview, not per answer (section 6.2).

## Ridge baseline (plan step 4, descriptive, no pass bar)

- **Features:** both dimensions' mean, share low, share high and trend, median latency and mean speaking time. Missing values are filled with the training-fold median.
- **Targets:** every reliable label.
- **Model:** standardise, then Ridge with alpha chosen by inner cross-validation. Outer 5-fold GroupKFold over students, so pre and post stay together.
- **Reported:** the out-of-fold Spearman with a bootstrap CI, next to the best single feature, which needs no fitting. The out-of-fold predictions are written for the fusion team.

## Done when

- The `interview_summary` exists for all 138 interviews.
- Every trend and latency statement the LLM may use has passed its test above.
