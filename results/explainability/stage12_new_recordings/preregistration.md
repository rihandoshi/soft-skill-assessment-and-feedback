# Stage 12 pre-registration: running on a new recording (Phase 4)

Written on 14 Nov 2026, after a single smoke-test clip (RecruitView 1579) and **before either test below was run**. The test script (`test_new_recording.py`) implements exactly this.

## Test 1: one answer through `analyze_answer.py` reproduces its dataset row

**Sample:** 30 RecruitView validation answers (unique recordings, `random_state=0`). Validation people are not in the recorded-answer reference, so these are the closest thing to a new user that has a dataset row.

**How:** each clip goes through `analyze_answer.py` from the audio file alone, in the `recorded_answer` setting, and is compared with its row in `stage8_detectors/tables/answer_levels.csv`.

There are two runs:

| Run | Speaker context |
|---|---|
| A, "with context" (the test) | The pitch range, reference F0 and typical pitch of the speaker are taken from the dataset tables, as `analyze_interview.py` would compute them from the speaker's other answers |
| B, "clip alone" (reported) | Everything from this one clip |

**Pass (run A):**
1. Every component (the four hesitation components, activation, pitch variability, speaking time) agrees with the table with Spearman ≥ 0.98 across the 30 answers.
2. The hesitation and expressiveness indices are within 2 points of the table for at least 90% of answers. The levels are compared from the cuts in `level_cuts.json`, which store the reference as 201 quantiles, so tiny differences are expected.
3. The levels match the table for every answer whose table index is more than 2 points away from a cut (20 or 80).

Run B reports how often the level changes when the speaker's other answers are not available. It is not a pass condition.

## Test 2: MIT interviews segmented without the Turker transcripts

**What:** `segment_interview.py` gets only the interview audio. It runs Whisper over the whole interview and uses an open speaker-verification model (`microsoft/wavlm-base-plus-sv`, no token) to split the two speakers. It finds the five questions in the interviewer's speech, then cuts the answers with the same rules as Stage 7.

**Reference:** the Stage 7 answer clips (`mit_answers.csv`), cut with the Turker transcripts.

**Pass:**
1. At least 90% of the Stage 7 answers are found (same interview, same question).
2. Of the answers found, at least 85% have both their start and their end within 3 s of Stage 7.
3. The median absolute boundary difference is at most 1 s.

## Done (from the README)

Both tests pass, and one MIT interview runs end to end with a single command (`analyze_interview.py`).

---

# Addendum: Test 2, second attempt

Written on 14 Nov 2026, after the first attempt **failed** (`report_attempt1.md`: 73.6% of answers found, 63.9% within 3 s, median difference 0.00 s), and **before any change to the method was tried**.

**What the first attempt showed.**
- Finding the questions in Whisper's text works. With the Turker speaker labels in place of the speaker split, the same rules find 94.5% of the answers, 88.1% within 3 s.
- The failures come from the speaker split. Even interviews with 93–97% of words labelled correctly lose more than half of their answers, because one mislabelled question turn or answer sentence breaks the cut.

**Rule for the second attempt.**
- The method may be changed using only the students with an **odd** number (P1, PP1, P3, PP3, ...).
- The changed method is then run **once** on the students with an **even** number. The same three criteria are applied to them alone: at least 90% of answers found, at least 85% within 3 s, and a median difference of at most 1 s.
- Both interviews of a student stay on the same side.
- The odd-student numbers are reported too, but they are development numbers, not a test.
- If the even students fail, Test 2 has failed. The pyannote route (README Phase 4 step 3) is then the recommendation.
