# Stage 10 notes: one reference per recording setting (Phase 2, 14 Nov 2026)

## Why

A level says "compared with other candidates". Up to Phase 1, the comparison was always with RecruitView train answers: short clips people recorded alone. Live interviews sound different. People hesitate more in conversation, and the room and microphone differ. Scored against recorded clips, 45% of MIT answers were "hesitated a lot". A new full-interview video would look like MIT, so nearly half of candidates would get that feedback.

## What was done

| | `recorded_answer` | `live_interview` |
|---|---|---|
| Who it is for | Self-recorded answers to one question (RecruitView-like) | Answers inside a real interview (MIT-like) |
| Reference | RecruitView train participants, unique recordings (1,325 answers) | MIT, 69 students (669 answers) |
| How the dataset's own answers are scored | Against the reference, as before | Each student against the **other 68 students**. Both of a student's interviews (before and after coaching) are left out together, so nobody is compared with themselves |
| Voice-pitch slope for expressiveness | Fitted on RecruitView train: 0.00102 per Hz | Fitted on the reference students: 0.00067 per Hz |
| Cuts for new recordings (`level_cuts.json`) | `settings.recorded_answer` | `settings.live_interview` (all 69 students). This is the default, because the product analyses interviews |

Everything that is fitted (percentiles, the voice-pitch slope, the index quantiles) is fitted on the reference only.

The scoring code is one function, `score()` in `fit_answer_levels.py`. It is used for both settings and for the leave-one-student-out loop.

MIT answers are also scored against the recorded-answer reference (`*_recorded_ref` columns in `answer_levels.csv`). That is only to measure the shift.

## Results

All numbers are in `report.md`.

**1. Every component shifts** (`tables/component_shift.csv`). Where the median live answer sits among recorded answers, and the tail shares:

| Component | Live median as recorded percentile | `high` tail, recorded → live | `low` tail, recorded → live |
|---|---|---|---|
| Filled-pause score (vocametrix) | 82 | 20% → 53% | 20% → 2% |
| Stumble score (vocametrix) | 65 | 20% → 26% | 20% → 3% |
| Speech-flow disfluent probability | 62 | 20% → 24% | 20% → 7% |
| Long mid-sentence pauses | 40 | 20% → 45% | 0% → 0% |
| Vocal activation | 47 | 20% → 9% | 20% → 18% |

- **Fluency.** Live interviews are clearly more hesitant, most of all in filled pauses. This matches the human transcripts: 4.3 against 0.6 fillers per 100 words. The speech-flow model shifts least of the three classifier scores.
- **Activation.** The median barely moves, but live answers are less spread out: few reach the recorded "lively" end. Far-field audio and a narrower group of speakers are plausible reasons; this was not tested.
- **Long pauses.** 80% of recorded answers have no long mid-sentence pause at all, so a pause-free answer sits at the 40th percentile. The component can push an answer towards "hesitant" but never towards "smooth" in the recorded setting. That is acceptable, because a missing pause is weak evidence of fluency, but it is a property of the index to know about.

By the criterion fixed in advance (median percentile within 40–60 and tails within 5 points of the reference's own), no component keeps a single reference. Two references it is.

**2. Level rates meet the target.** `low` and `high` each fall within 18–21% for every dimension in both settings. Before this phase, live answers got 45% "hesitated a lot" and only 7% "lively".

| MIT, hesitation | Against recorded answers | Against live interviews |
|---|---|---|
| `high` | 45% | 20% |
| `low` | 2% | 20% |

Of the 303 MIT answers that were `high` against recorded answers, 136 stay `high` and 167 become `normal`. None flips to the opposite end. The same holds for expressiveness, except that 9 answers move from `normal` to `low`. That is the new voice-pitch slope: some high-voiced students were lifted too much by the recorded-answer slope.

**3. What the indices track did not change.** Interview level, partial correlations with the 95% CI:

| | Recorded-answer reference | Live-interview reference |
|---|---|---|
| Fluency vs `NoFillers` | −0.58 [−0.70, −0.43] | −0.58 [−0.70, −0.42] |
| Fluency vs `Overall` | −0.44 | −0.45 |
| Expressiveness vs `EngagingTone` | +0.63 | +0.65 [+0.52, +0.75] |
| Expressiveness vs `Excited` | +0.66 | +0.67 |

This was expected, since percentiles within MIT keep the ranks.

Is this circular? Not in the way the rules forbid. The live cut-offs are fixed at p20 / p80 in advance, not tuned on the MIT ratings. Each student's levels come from the other students only.

The one thing that does change is the share of "hesitant" answers per interview as a measure: its relation with `NoFillers` drops from −0.61 to −0.48, partial −0.43. It is still supported. Fewer answers are flagged, so the share is coarser.

**4. Voice pitch.** With MIT's own slope, activation no longer follows voice pitch within MIT: ρ +0.04 after the adjustment, +0.42 before. "Flat" rates by voice-pitch tercile in live interviews:

| | Low voices | Middle | High voices |
|---|---|---|---|
| Before the adjustment | 38% | 13% | 8% |
| Phase 1 (recorded-answer slope) | 23% | 21% | 31% |
| Phase 2 (live slope) | 27% | 16% | 19% |

The low-voice tercile is still flagged a little more often. A straight line cannot remove all of it: the middle tercile is where the two voice-pitch groups meet. With 69 students, this is close to what the data can support.

## Limits

- **The live-interview reference is 69 US students in one room**, all answering the same five questions. It is the best available. The team's own recordings should replace or join it as soon as there are a few dozen interviews. Re-run `fit_answer_levels.py` with them added to the pool.
- **The two settings are two datasets.** Each "setting" difference also contains everything else that differs between RecruitView and MIT: microphones, the people, the questions, the rating culture. It is not a pure effect of "live versus recorded".
- **A new recording has to say which setting it is.** Phase 4's `analyze_interview.py` uses `live_interview` and `analyze_answer.py` uses `recorded_answer`.
- **The confidence level is still computed but unused.** It was contradicted on MIT in Stage 8.

## Re-run

`fit_answer_levels.py` (both settings, about 10 s) → `validate_detectors_mit.py` → `build_answer_evidence.py` → `make_listening_list.py` → `compare_settings.py` (about 1 min). CPU only.
