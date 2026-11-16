# Stage 12 notes: the pipeline on a new recording (Phase 4, 14 Nov 2026)

**Outcome: partly done.**
- Analysing an answer works and reproduces the dataset exactly (Test 1 passes).
- One command runs a full interview end to end.
- Cutting the answers out of an interview without a human transcript fails its test (Test 2). It needs a proper diarization model: pyannote, which needs a Hugging Face token.

The criteria were written before each test ran (`preregistration.md`, including the addendum for the second attempt). The generated numbers are in `report.md`, and the first segmentation attempt in `report_attempt1.md`.

## What exists now

| Script | In | Out |
|---|---|---|
| `analyze_answer.py` | One audio file (any format; ffmpeg for video) and the question text | Evidence JSON (README section 11 format) |
| `segment_interview.py` | A full interview recording | The five answers: question text, start and end |
| `analyze_interview.py` | A full interview recording | `{name}.json` (every answer's evidence and, since Phase 5, the `interview_summary` block: see `stage14_interview_level/notes.md`) and `{name}_measures.csv` (internal) |
| `test_new_recording.py` | — | Tests 1 and 2 |

**Only three files are read**, besides the model weights:
- `level_cuts.json`
- `quality_gate.json`
- `validation_mit.csv`, which decides which dimensions may be used

**Same code as the dataset.** VAD settings, Whisper settings, Praat passes, sentence building, the pause classification and every model window are the dataset code, imported rather than copied, except the Praat passes. Those are re-implemented to work from samples, but with the same calls and settings.

**Speaker context.** The Praat pitch range and the speaker's typical pitch come from all the candidate's answers in the interview, as for MIT.

**Quality gate.** It is applied only in the `recorded_answer` setting, as in the dataset levels. In `live_interview` it is reported in `internal` but not applied: MIT's far-field audio, on which the levels were validated, fails the RecruitView-fitted gate almost everywhere. A gate for live interviews is an open item (it needs the team's own recordings).

**Confidence** is never computed (Phase 3 fallback); it always appears under `not_assessed`.

**GPU:** models are loaded one at a time and freed. Peaks:

| Run | Peak |
|---|---|
| One answer | 1.71 GB |
| One full interview | 1.86 GB |
| The segmentation test | 1.86 GB |

The 6-GB laptop card is enough for all of Phase 4.

**CrisperWhisper** is not needed. No audio level uses it; its filler counts are for the text branch. The README listed it as a Phase 4 GPU need for the audio branch; that was wrong.

## Test 1: one answer reproduces its dataset row (PASS)

30 RecruitView validation answers went through `analyze_answer.py` from their audio files alone.

**With the speaker context** (pitch range and typical pitch as the dataset computed them):
- Every component matches the table to rounding error. The largest difference is 0.006 in activation; the classifier probabilities differ by about 1e-8.
- Indices differ by a median of 0.06 points.
- The levels match for 27 of 30 answers on expressiveness and 29 of 30 on hesitation. Every difference is at an index within 2 points of a cut.
- The small index differences come from `level_cuts.json` storing the reference as 201 quantiles.

**Each clip alone** (no other answers from the speaker):
- Hesitation does not change.
- Expressiveness moves by a median of 1.1 points and up to 7.9. 2 of the 30 levels change away from a cut.

This is why `analyze_interview.py` takes the speaker context from all the candidate's answers, and why a single clip on its own is less reliable for expressiveness.

## Test 2: answers cut from MIT interviews without the transcripts (FAIL)

**Method:**
- Whisper over the whole interview.
- `microsoft/wavlm-base-plus-sv` speaker embeddings (ungated), k-means into two speakers. The interviewer is the one who speaks less.
- Stage 7's question and answer rules, unchanged.

| | Answers found (bar 90%) | Within 3 s (bar 85%) | Median difference (bar 1 s) |
|---|---|---|---|
| Attempt 1, all 138 interviews | 73.6% | 63.9% | 0.00 s |
| Attempt 2, development (odd-numbered students) | 84.8% | 66.6% | — |
| **Attempt 2, held out (even-numbered students, the test)** | **75.2%** | **58.5%** | 0.00 s |
| Turker speaker labels instead of the speaker model (ceiling) | 94.5% | 88.1–88.3% | 0.00 s |

**Attempt 1** embedded each whole sentence chunk.
- Its first run picked the interviewer by question keywords, which swapped the two speakers in most interviews: candidates say "team" and "leadership" more than the interviewer does. That was fixed (the interviewer is the one who speaks less) before attempt 1 was scored.
- It failed.

**Attempt 2** was pre-registered after attempt 1 failed. It was developed on odd-numbered students only, comparing:
- 1.5-s window embeddings;
- smoothing over time, which made it worse because question turns are short and get smoothed away;
- giving every sentence that ends with "?" and matches a question pattern to the interviewer, which helped.

Attempt 2 then failed on the held-out students. Its development gain did not carry over.

**Why it fails.**
- The speaker split is mostly right: a median of 96% of words agree with the Turker labels.
- But the answer cut is fragile. One question turn labelled as the candidate, or one answer sentence labelled as the interviewer, moves a boundary by a whole turn. Even interviews with 93–97% word agreement lose more than half of their answers.
- With correct speaker labels the same rules pass. So the missing piece is a stronger diarization model, not the answer rules.

**Effect on the levels.** In the end-to-end run of P1:
- Q3 has the same boundaries as Stage 7 and reproduces its levels exactly.
- Q1 and Q4 start about 2 s earlier than Stage 7, probably taking in the end of the interviewer's question.
- In the short Q4 and Q5 answers, that moved hesitation from normal to high, and from low to normal.

Boundary accuracy matters most for short answers.

## End-to-end run (one command)

`python scripts/explainability/analyze_interview.py Datasets/MIT/MIT_INTERVIEW_DATASET/Audio/P1.wav --name MIT_P1`

- Input: the original 48-kHz stereo file.
- 41 s in total for a 3.3-minute interview, peak 1.86 GB.
- All five questions found; output in `examples/MIT_P1.json`.

This meets the README's "runs end to end with a single command". The segmentation inside it has not passed its test.

## Next step (needs a decision from you or the team)

1. **pyannote diarization** (`pyannote/speaker-diarization-3.1`). It is gated, so it needs a Hugging Face token and accepting its terms. It fits on the 6-GB card. Only `label_speakers()` in `segment_interview.py` changes; then Test 2 is re-run on the even-numbered students.
2. **Or** the team member who owns the shared upstream (README section 11) supplies speaker turns. `segment_interview.segment()` then only needs the turn times.

Until then, `analyze_interview.py` works, but its answer boundaries are unvalidated. They are right for about 6 in 10 answers on MIT.

## Re-run

- `python scripts/explainability/test_new_recording.py --test answers`: about 10 min GPU. Add `--reuse` to rebuild the report only.
- `python scripts/explainability/test_new_recording.py --test segment --attempt 1|2`: about 10 min GPU, cached per interview in `outputs/audio/cache/stage12_new_recordings/`.
