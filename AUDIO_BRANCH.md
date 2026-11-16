# The audio branch: everything done so far

Last updated 16 Nov 2026.

This document covers the whole audio branch:
- what it measures and how;
- which models it uses;
- how each measure was tested, and the numbers;
- what was tried and dropped;
- where every file is;
- how the fusion model and the LLM use the outputs.

Technical terms are explained where they first appear, and again in the [glossary](#14-glossary) at the end.

**Where to go for what:**
- **Working on fusion:** read [section 10](#10-for-the-fusion-teammate-what-to-use-and-how) first, then `results/explainability/stage13_fusion_export/SCHEMA.md`.
- **Exact numbers of any stage:** the `report.md` (generated) and `notes.md` (hand-written) in that stage's folder under `results/explainability/`.

---

## Contents

1. [The short version](#1-the-short-version)
2. [The bigger picture: where audio sits in the project](#2-the-bigger-picture-where-audio-sits-in-the-project)
3. [The two datasets and what they can and cannot do](#3-the-two-datasets-and-what-they-can-and-cannot-do)
4. [The audio soft skills: what is measured and how](#4-the-audio-soft-skills-what-is-measured-and-how)
5. [How a skill is tested ("validated")](#5-how-a-skill-is-tested-validated)
6. [From a number to a level: references and settings](#6-from-a-number-to-a-level-references-and-settings)
7. [The interview level](#7-the-interview-level)
8. [Running on a new recording](#8-running-on-a-new-recording)
9. [Score prediction models (the "legacy" work and what is kept)](#9-score-prediction-models-the-legacy-work-and-what-is-kept)
10. [For the fusion teammate: what to use and how](#10-for-the-fusion-teammate-what-to-use-and-how)
11. [Everything that was tried, in order](#11-everything-that-was-tried-in-order)
12. [All findings in one list](#12-all-findings-in-one-list)
13. [Folders and files: what is where, and is the legacy folder still useful?](#13-folders-and-files-what-is-where-and-is-the-legacy-folder-still-useful)
14. [Glossary](#14-glossary)
15. [What is still open](#15-what-is-still-open)

---

## 1. The short version

**The goal.** Take an interview recording and give the candidate feedback on how they *sounded*, tied to a moment. For example: "When you were asked about leadership, you hesitated a lot. Try pausing silently to plan your first sentence." The user never sees a score. Numbers stay internal and feed the fusion model and the LLM.

**What the audio branch can say.** Two soft skills, each tested against independent human ratings:

| Soft skill | What it covers | Per answer or per interview | Levels | Status |
|---|---|---|---|---|
| **Fluency** (`hesitation` in the files) | Filled pauses ("um", heard as sound), stumbles, long pauses in the middle of sentences; at interview level also being slow to start answering (response latency) | Per answer; "slow to start" once per interview | `low` (spoke smoothly) / `normal` / `high` (hesitated a lot) | **Validated** |
| **Vocal expressiveness** | Does the voice sound lively or flat | Once per interview (it is mostly a property of the person) | `low` (flat) / `normal` / `high` (lively) | **Validated** |

**What it cannot say, after testing:**

| What | Why not |
|---|---|
| **Vocal confidence** | Every model failed or ran opposite to the human ratings on the independent dataset. Left to fusion |
| **Clarity** | The only available model rates Indian-English accent as "slurred". Dropped |
| **Pace, composure, uptalk, loudness** | Measured, but not validated or not fair to give as advice |
| **Answer length** | Measured and passed on, but the text branch judges it, because only the content can tell a rambling answer from a thorough one |
| **Trends within an interview** ("you got better towards the end") | Mostly noise. Only expressiveness changes may be stated, as plain description |

**What goes to fusion:**
- Per answer: 12 audio score predictions (out-of-fold on train), the hesitation and expressiveness index and level, a 1,536-number WavLM embedding, the evidence JSON for the LLM, and context numbers.
- Per MIT interview: aggregates, an interview summary and an audio-only baseline.

**Where things stand:**
- Phases 1–6 of the plan are done.
- Speaker segmentation of new recordings is deferred: its test failed, and the fix needs pyannote or the shared upstream.
- Phase 7 (LLM feedback) and one full-pipeline demo come after the teammate's fusion work.

---

## 2. The bigger picture: where audio sits in the project

The project has three branches (audio, text, video), a fusion step and an LLM that writes the feedback.

```
 new interview video
        │
        ▼
 SHARED UPSTREAM (one owner, run once)
   audio extraction → who speaks when → question / answer segments → transcript with word times
        │
        ├──► TEXT branch   what was said: content, structure, filler WORDS, answer length
        ├──► VIDEO branch  face, gaze, posture
        ▼
 AUDIO branch (this repo), per answer
   detectors → continuous values → percentile vs a reference population → level → evidence JSON
   WavLM embedding + audio score predictions ─────────────────────────────► FUSION model
        │
        ▼
 INTERVIEW LEVEL: per-answer outputs in order → interview level, answers to point at, latency, trends
        │
        ▼
 FUSION (teammate): combines the three branches per answer / per interview
        │
        ▼
 LLM: reads only the evidence → coach-style feedback → automatic check that every claim is in the evidence
```

**Who owns what:**

| Topic | Owner |
|---|---|
| Fluency as sound and timing, vocal expressiveness, response latency | Audio |
| Filler *words*, hedging, content, structure, whether a long answer stayed on topic | Text. Audio hands over the verbatim transcript: `stage4_asr/words.csv` and `stage8_detectors/tables/crisper_words.csv` |
| Perceived confidence | Fusion, using all three branches. Audio could not measure it reliably |

**Rules that run through everything:**
- **The user never sees a number.** Only levels and their wording reach the LLM.
- **Only the clear ends are spoken about.** `normal` produces no comment, because a wrong statement is worse than silence.
- **No feedback on things that identify the person or the microphone** (loudness, absolute pitch, voice quality).
- **No skill reaches the LLM until it has passed a test** written down before it ran, against reliable human ratings.

---

## 3. The two datasets and what they can and cannot do

### RecruitView (training and calibration)

| Fact | Consequence |
|---|---|
| 2,011 self-recorded answer clips, 331 people, 76 questions, median 27 s. Fixed split **by person**: 1,394 train / 319 val / 298 test clips (231 / 49 / 51 people) in `Datasets/Split Dataset/` | The split is never regenerated. Everything fitted uses train people only |
| 12 expert-rated scores per clip. The audio-relevant ones are `confidence_score`, `speaking_skills` and `overall_performance` | There are no labels for hesitation, pace or tone, so those skills cannot be trained here |
| **106 recordings are byte-identical copies** under different ids (52 groups), rated independently. Their scores agree only at Spearman 0.55 (confidence), 0.34 (speaking) and 0.57 (overall) | This is the **label ceiling**: no model can be judged more precisely than that. The best audio model reaches 0.50–0.53, already at the ceiling |
| Rating noise is about as large as the difference between one person's answers | RecruitView can rank *people*, but cannot tell which of one person's answers was better |
| Indian university students, monologues | Speech is more rehearsed than in a live interview |

### MIT Interview dataset (independent validation)

| Fact | Consequence |
|---|---|
| 138 full mock interviews of 69 US students, each interviewed before and after coaching, about 5 questions each | Too few for a learned time-series model. Enough for validation and simple interview-level summaries |
| 16 ratings per interview from 9 Mechanical Turk raters, plus human transcripts that mark speaker turns and "um / uh" | The checks run at interview level: each measure is averaged over the interview's answers |
| **Reliable ratings** (ICC, see glossary): `EngagingTone` 0.86, `Excited` 0.82, `NoFillers` 0.78, `RecommendHiring` 0.76, `Overall` 0.74, `NotAwkward` 0.68 | Only these are used as test targets |
| **Unreliable ratings:** `SpeakingRate` 0.03, `Paused` 0.26, `NotStressed` 0.49 | Pace and pausing cannot be validated against their own labels |
| One far-field room microphone for both speakers | Audio quality is poor. A separate "live interview" reference was needed (section 6) |

**The two rater groups disagree about vocal energy.** RecruitView's experts rated calm, low-energy voices as *more* confident; MIT's raters rated energetic voices higher. This one fact explains why confidence failed (section 4.4).

---

## 4. The audio soft skills: what is measured and how

The audio branch aims at **two soft skills** that can be heard in the voice and that passed their tests:

| Soft skill | In plain words | What the candidate can be told |
|---|---|---|
| **1. Fluency** | Does the answer flow, or does the speaker keep stalling: "um"s, stumbles, long silences mid-sentence, a long wait before starting? | "you hesitated a lot", "you spoke smoothly", "you often took a while before starting to answer" |
| **2. Vocal expressiveness** | Does the voice sound lively and engaged, or flat? | "your voice sounded quite flat", "your delivery was lively" |

Two more were wanted and tested, but failed: **vocal confidence** (4.4) and **clarity** (4.5). Some things are measured but never turned into feedback (4.6).

### 4.0 Basic ideas used by both skills

**Windows.**
- Audio models are trained on short clips of a few seconds, so they cannot judge a 60-second answer in one go. The answer is therefore looked at a few seconds at a time, like sliding a small frame along it.
- Each frame is a **window**. For fluency the window is 3 seconds long and a new one starts every 1.5 seconds: 0–3 s, 1.5–4.5 s, 3–6 s and so on. The 1.5-s step is the **hop**.
- The windows overlap, so a stumble that falls on the boundary between two windows is still fully inside one of them.
- Windows that are mostly silence are skipped. A speech detector (Silero VAD) marks which parts of the audio are speech, and a window is kept only if at least half of it is speech. The models judge how someone speaks, and silence would only dilute that.

**Probability per window, average per answer.**
- For each window a model gives a number between 0 and 1: how likely it is that the window contains, say, a filled pause.
- Averaging these numbers over all of an answer's windows gives one value for the answer. Roughly: how much of the answer sounded like that.

**Percentile.**
- A raw value such as 0.31 means nothing on its own. So it is compared with a **reference population**: many other candidates' answers.
- The **percentile** is the share of reference answers with a lower value. 85 means "more than 85% of other candidates' answers".
- There are two reference populations (section 6): self-recorded answers (RecruitView) and live interviews (MIT).

**Index and level.**
- When a skill has several components, their percentiles are averaged with fixed weights, and that average is turned into a percentile again. This gives the skill's **index**, from 0 to 100.
- The index is cut at 20 and 80: at or below 20 is `low`, at or above 80 is `high`, everything else is `normal`.
- Only `low` and `high` produce feedback. The wide middle band keeps the system silent when the case is not clear.

### 4.1 Fluency (validated)

**What it means.** Fluent speech moves forward. Hesitant speech stalls: filled pauses ("um", "uh", "er"), stumbles (repeating a word, getting stuck on a sound), long silences in the middle of a sentence, and a long wait before the answer starts.

**What the audio branch listens for, and how.** Four things inside each answer, plus one at interview level.

| # | Behaviour | In plain words | How it is measured |
|---|---|---|---|
| 1 | **Filled pauses** (`p_interjection`) | How much of the answer sounds like "um / uh" | A pretrained stutter / disfluency model, `vocametrix/wav2vec2-xlsr-53-stuttering-classification`, listens to each 3-s window and gives the probability that it contains an interjection (a filled pause). The answer value is the average over its windows. It hears the *sound* of hesitating; it does not read words |
| 2 | **Stumbles** (`p_repetition_block`) | Repeating words, or getting stuck before a word comes out | The same model, same windows: probability of "word repetition" plus probability of "block" (a stuck, silent or strained moment before a word), averaged over the windows |
| 3 | **Disfluency, second opinion** (`sf_p_disfluent`) | A second, independently trained listener for "does this sound disfluent" | `tiantiaf/wavlm-large-speech-flow` (Vox-Profile), trained on stuttering and fluency datasets. Same 3-s windows; probability of "disfluent", averaged. Two models agreeing is more trustworthy than one |
| 4 | **Long pauses inside sentences** (`long_mid_pause_rate_per_min`) | Silences of 1 s or more in the middle of a sentence, where the speaker lost the thread | The speech detector finds every silence. The transcript's word times and punctuation tell whether the words before and after the silence belong to the same sentence. Silences of 1 s or more *inside* a sentence are counted, per minute of speaking. Pauses *between* sentences are normal and not counted |
| 5 | **Slow to start** (response latency; interview level only) | How long the candidate waits after the question before saying anything | The time from the interviewer's last word to the candidate's first word, per answer. The interview value is the median over its answers. Only exists for full interviews, because a single clip has no question in it |

**How 1–4 become the answer's level.**
1. Each value becomes a percentile against the reference answers.
2. The percentiles are averaged with weights fixed in advance: filled pauses 1/6, stumbles 1/6, second classifier 1/3, mid-sentence pauses 1/3. The two classifiers together get 2/3, the pauses 1/3.
3. The average is turned into a percentile again: `hesitation_index` (0–100, higher = more hesitant).
4. ≥ 80 → `high` ("hesitated a lot"); ≤ 20 → `low` ("spoke smoothly"); otherwise nothing is said.

**Reasons in words.** When an answer is `high`, each behaviour that is itself in the top fifth is named: "many filled pauses", "stumbles", "long pauses in the middle of sentences". The second classifier has no wording of its own.

**Where.** The 3 neighbouring windows (about 6 s) with the highest disfluency, averaged over the two classifiers, become the time span, e.g. 0:19–0:25. The LLM can then say *when*.

**Slow to start** is not part of the answer index. At interview level it is a sub-item of fluency: `hesitation.slow_to_start` in the interview summary.
- `high` means the interview's median wait is in the slowest fifth of interviews. Only `high` is ever said: "often took a while before starting to answer".
- A quick start is never praised, and a pause to think is not called bad. The wording only describes.

**Proof it works** (MIT, 138 interviews, 9 independent raters; section 5 explains the test):

| Measure | Rating it should match | Correlation | After removing answer length and recording quality |
|---|---|---|---|
| Hesitation index | `NoFillers` ("didn't use fillers"), expected negative | −0.63 | −0.58 [−0.70, −0.43] |
| Hesitation index | "um / uh" counted in the human transcripts | +0.49 | +0.58 |
| Slow to start | `Overall`, expected negative | −0.36 | −0.38 [−0.52, −0.23], still −0.35 after also removing the hesitation index |
| Slow to start | `NotAwkward`, expected negative | −0.25 | −0.27 [−0.44, −0.09] |

The combined index does better than any one behaviour alone: filled pauses alone −0.51, the second classifier alone −0.56.

### 4.2 Vocal expressiveness (validated)

**What it means.** Whether the voice sounds animated and engaged (energy, emphasis, rise and fall) or flat and low-energy.

**How it is measured.**
1. **Windows:** 4 seconds long, a new one every 2 seconds. Windows with at least half speech are kept.
2. **Model:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, a speech-emotion model trained on podcast speech that listeners rated. For each window it gives:
   - **arousal:** how energetic or excited the voice sounds;
   - **dominance:** how strong or in control it sounds.
   - On this data the two always move together (correlation 0.98), so their average is used and called **activation**: how lively the voice sounds.
3. **Answer value:** the average activation over the answer's windows.
4. **Voice-pitch correction:**
   - The model gives higher activation to higher voices. Without a correction, people with low voices, often men, would be called "flat" four times as often.
   - So the activation that a speaker's typical pitch would predict (a straight line fitted on the reference population) is subtracted, giving `activation_adj`.
   - The speaker's typical pitch is the median pitch over all of their answers.
5. **Percentile** of `activation_adj` → `expressiveness_index` (0–100, higher = livelier) → ≤ 20 `low` ("sounded quite flat"), ≥ 80 `high` ("sounded lively and engaged").
6. **Reason in words**, from pitch movement measured with Praat (in semitones, so high and low voices are comparable):
   - "little rise and fall in the voice" when a `low` answer's pitch variation is also in the bottom fifth;
   - "plenty of rise and fall" for `high` answers in the top fifth.
7. **Where:** the 4-s stretch with the lowest activation (the flattest moment).

**Per interview, not per answer.** 87% of the differences in activation are between people, not between one person's answers: someone with a flat voice is flat in every answer. So the feedback is given once for the interview ("throughout the interview your voice sounded quite flat"), pointing at the flattest answer, rather than repeated five times. The interview summary marks this with `feedback_at: interview`.

**Proof it works** (MIT, controls: length, quality and the speaker's pitch):

| Measure | Rating | Correlation | After controls |
|---|---|---|---|
| Expressiveness index | `EngagingTone` | +0.63 | +0.65 [+0.52, +0.75] |
| Expressiveness index | `Excited` | +0.63 | +0.66 |
| A second emotion model (Odyssey) | `EngagingTone` | +0.72 | +0.65. It agrees with the first at +0.86 |

### 4.3 What is said about the whole interview

`interview_summary.py`, Phase 5. Besides each answer's feedback, the LLM gets one summary per interview.

| Item | What it is |
|---|---|
| Interview level per skill | The average of the answers' indices, compared with other interviews (`low` / `normal` / `high`) |
| Answers to point at | Fluency: the most and least hesitant answer. Expressiveness: the flattest and liveliest |
| Slow to start | Under fluency, see 4.1 |
| Change across the interview | Tested (section 7). Fluency changes are no bigger than chance, so never stated. Expressiveness changes may be stated as plain description ("livelier in the last two answers than in the first two") for the few interviews where the change is clear |

### 4.4 Vocal confidence (tested, failed: not used)

This was the most wanted skill. Every route was tried:

| Attempt | RecruitView | MIT `Overall` (after length, quality and activation) | Verdict |
|---|---|---|---|
| WavLM L7 + Ridge trained on RecruitView `confidence_score` | Spearman 0.50 in repeated cross-validation | −0.58 (expected positive) | **Contradicted:** it learned "calm, short, quiet" |
| Speech-emotion "dominance" output | — | Same as activation (ρ 0.98) | Measures energy, not assurance. Now used for expressiveness |
| Speaking-style tagger `tiantiaf/wavlm-large-voice-quality` (25 tags; "authoritative, flowing, enunciated" minus "hesitant, soft, hushed, stammering") | — | +0.06 [−0.14, +0.24] | No support |
| The tagger's "hesitant" tag vs `NoFillers` | — | +0.12 (expected negative) | No support |
| Retrained on RecruitView with length and quality removed from the target. Features: WavLM-base L7, WavLM-large L18, emotion-model embeddings. Ridge and an "extremes only" logistic model | Grouped-CV 0.32–0.34 (passes the ≥ 0.30 bar) | −0.34 to −0.48 | **Contradicted**, every variant |
| Hugging Face models named "confidence" | Empty cards or chance accuracy | — | Not usable |
| Fine-tuning | Not run: it needs a 16-GB GPU and is expected to overfit. The problem is rater disagreement, not model capacity | — | — |

**Why it fails.** The RecruitView raters heard calm, low-energy, lower-pitched delivery as confident; MIT's raters rate the same delivery lower. The disagreement is about the voice itself.

**Fallback.**
- The audio branch reports `confidence` under `not_assessed`.
- The LLM may only link validated behaviours to how they can come across, for example "hesitating a lot can come across as unsure".
- Perceived confidence is a fusion-level judgement.

Files: `run_confidence_detector.py`, `run_style_tagger.py`, `analyze_style_tagger.py`, `run_confidence_partb.py`, `extract_partb_embeddings.py`; results in `stage11_style_tagger/`.

### 4.5 Clarity (tested, dropped)

| Attempt | Result |
|---|---|
| The tagger's "crisp / enunciated minus slurred" tags | The typical RecruitView answer gets "slurred" 0.92, against 0.06 on MIT, even in the best-quality recordings. The tagger rates **Indian-English accent** as slurred. Never to be used |
| ASR word confidence (Stage 4) | Ran opposite to the ratings and reflects accent and microphone |
| Trailing off (sentence-end energy) | Weak support on RecruitView, none on MIT |

### 4.6 Measured but not spoken about

These stay in the tables (`stage1_descriptors/`, `stage4_asr/tables/`) and some are exported as context.

| Measure | Why it is not feedback |
|---|---|
| Answer length (`speech_span_s`) | Whether long is bad depends on content, which the text branch judges. It is exported as a number for text and fusion. It did relate to ratings (RecruitView within-person ρ ≈ −0.12; MIT `StructuredAnswers` −0.27) |
| Pace (speech rate) | The `too_fast` flag was contradicted on MIT, the `SpeakingRate` label is unreliable (ICC 0.03), and no optimal band was found |
| Composure / tension | Same activation output as expressiveness, but the two rater groups disagree about whether calm or energetic is better |
| Rising sentence ends (uptalk) | Weak and not specific |
| Loudness, pitch in Hz, jitter, shimmer | They identify the microphone or the speaker. Loudness's relation to the scores disappears once recording quality is controlled |
| Trends within an interview | Phase 5 tests: hesitation trends are noise; expressiveness trends are description only (section 7) |

---

### 4.7 How the feedback reaches the LLM

The LLM never sees audio and never sees a score. It gets two kinds of JSON and writes the feedback only from them. Phase 7, `generate_feedback.py`, is not built yet; this is the input it will use.

**A. Per answer: the evidence JSON** (`evidence.jsonl`, one line per answer; written by `analyze_answer.py` for new recordings). A real example, MIT student 8, question 3:

```json
{
 "answer_id": "MIT_P8_Q3",
 "question": "So please tell me about a time when you were working in a team and faced a challenge. ...",
 "setting": "live_interview",
 "observations": [
  {"dimension": "hesitation", "level": "high", "kind": "issue", "say": "hesitated a lot",
   "reasons": [], "where": {"start": "0:19", "end": "0:25"},
   "tip": "plan the first sentence before starting, and when you need a moment, pause silently"}
 ],
 "not_assessed": [{"dimension": "confidence", "why": "not validated against human ratings"}],
 "rules": "compared with other candidates' answers only; mention only the observations listed",
 "internal": {"hesitation": 80.9, "expressiveness": 35.7, "confidence": 75.9, "speech_span_s": 82.4}
}
```

| Field | Meaning | Does the LLM use it? |
|---|---|---|
| `question` | The question asked, so the feedback can name it | Yes |
| `observations` | Only skills at `low` or `high`. `normal` produces no observation, so an empty list means nothing to say about the voice | Yes |
| `dimension` | `hesitation` (fluency) or `expressiveness` | Yes |
| `kind` | `issue` (something to work on) or `strength` (something done well) | Yes |
| `say` | The approved plain-words statement | Yes: the core of the sentence |
| `reasons` | Which behaviours caused it, in words | Yes, if present |
| `where` | The time span where it was strongest | Yes: "around 0:19" |
| `tip` | One practical suggestion for an issue | Yes |
| `not_assessed` | Skills the audio branch cannot judge | Yes, as a "do not talk about this" list |
| `internal` | Numbers for fusion and for the automatic fact-check | **No. Must be removed before the LLM sees the JSON.** It also still contains `confidence`, the output of the failed confidence model: never use it |

**B. Per interview: the `interview_summary`** (`interview_summaries.jsonl` for MIT; inside `{name}.json` from `analyze_interview.py`). Per skill:
- `level` and `say`: e.g. "the voice sounded quite flat throughout the interview";
- `point_at`: which answers to name;
- for fluency, `slow_to_start`;
- for expressiveness, `trend` and `trend_say` where allowed.

**C. Framing rules for the LLM** (Phase 7 plan, section 9 of `markdown/README.md`):
1. Mention only what is in `observations` or the summary's `say` fields. Never add anything about confidence, clarity, pace, loudness or pitch.
2. Name the question and the moment ("when you were asked about a team challenge, around 0:19, ...").
3. Plain words and no numbers: never a score, percentile or "compared with 80% of candidates".
4. One practical tip per issue (use `tip`).
5. **Fluency** is mentioned per answer. **Expressiveness** is mentioned once for the interview (`feedback_at: interview`). `slow_to_start` is mentioned once for the interview.
6. Describe what was heard, not what the person felt or how it scored. "You hesitated a lot" is allowed. "You were nervous" and "this lowered your score" are not.
7. Validated behaviours may be linked to how they can come across ("which can come across as unsure"), never stated as a confidence judgement.
8. An automatic check afterwards: every behaviour, time and quote in the text must be in the JSON, otherwise regenerate.

**How the example above would be framed:** *"When you were asked about a time your team faced a challenge, you hesitated quite a bit around 0:19–0:25. Next time, plan your first sentence before you start, and if you need a moment, pause silently rather than filling the gap."*

---

## 5. How a skill is tested ("validated")

No dataset has per-answer labels for fluency or expressiveness. So the measures are checked against existing human ratings, mainly MIT's, using a fixed procedure (`validate_detectors_mit.py` and `interview_summary.py`):

1. **Rater agreement first.** For every MIT rating, the ICC(2,k) across the 9 raters. Ratings below 0.5 are not used: raters don't agree, so nothing can be validated against them.
2. **Pre-registration.** Before running, write down which rating each measure should relate to and in which direction (e.g. hesitation index vs `NoFillers`, negative). This stops picking whichever of the 16 ratings happens to correlate. Pre-registration files: `stage11_style_tagger/preregistration.md`, `stage12_new_recordings/preregistration.md`, `stage14_interview_level/preregistration.md`.
3. **Interview level.** MIT ratings are per interview, so the measure is averaged over the interview's answers.
4. **Spearman correlation (ρ)** between the interview value and the mean rating, over 138 interviews.
5. **Controls (partial correlation).** Repeat the correlation after removing the effect of average answer length and recording quality, plus the speaker's typical pitch for voice-based measures. A skill must not pass just by tracking length or the microphone.
6. **Bootstrap 95% confidence interval.** Resample *students* 2,000 times, keeping each student's two interviews together, and take the middle 95% of the correlations.
7. **Pass rule ("supported").** The raw and the controlled correlation are both in the expected direction, and both intervals exclude 0.
8. **Discriminant check.** A measure should relate most to its own rating, not equally to everything. Raters who like someone rate them well on everything (a "halo").

**What this does and does not show:**
- **It shows** that a measure moves with what independent raters perceive, across interviews, and not because of length or quality.
- **It does not show** that the level of any single answer is right.
- **The p20 / p80 cuts are a convention.** They were not tuned on MIT, deliberately.
- **A human listening check is still open:** 60 answers listed in `stage8_detectors/listening_check.csv`.

**RecruitView cannot validate per-answer delivery.** Its scores don't support either skill, because rating noise hides the differences between one person's answers (Stage 3). Both validated skills rest on MIT.

### 5.1 Is it circular to build the comparison table from MIT and then test on MIT?

**Mostly no.**

1. **The table never sees the ratings.** It is built only from the audio measurements, sorted into cut points. The MIT ratings are never used to build it. Results are inflated when something is fitted to the test labels; here nothing is.
2. **The test doesn't depend on the table.** It is a rank correlation: do interviews the measure ranks as more hesitant also get worse "no fillers" ratings? Turning scores into percentiles with any table keeps the order, so any table gives about the same result.
3. **Checked directly** (Phase 2, `stage10_reference_settings/tables/validation_by_reference.csv`). MIT was scored against both tables. The RecruitView table contains no MIT data, so its column is effectively a test on unseen data:

| Test (after controls) | RecruitView table (no MIT data) | MIT table |
|---|---|---|
| Fluency vs `NoFillers` | −0.58 | −0.58 |
| Expressiveness vs `EngagingTone` | +0.63 | +0.65 |
| Expressiveness vs `Excited` | +0.66 | +0.67 |

4. **Nothing else was fitted to MIT's ratings.** The detection models were trained by others on other data. The fluency weights were fixed before the MIT test. The 20 / 80 cut-offs are a convention and were not tuned on MIT.

**Where the results are optimistic:**

1. **Some choices were made after seeing MIT results.**
   - Expressiveness switched from pitch movement to the emotion model's liveliness because liveliness matched MIT ratings better (+0.68 vs +0.30).
   - The second fluency model joined the index only after passing its MIT test.
   - Picking the best of a few options on the same data you report on makes the reported numbers somewhat too good. Few options were tried and they were mostly far apart, so the inflation is probably small, but it is there.
2. **The live-interview pitch correction was fitted on MIT audio.** That fit uses no ratings, and the test also controls for pitch, so the effect is small. Still, it isn't fully independent of the test.
3. **Some MIT numbers are not evidence.**
   - "About 20% of MIT answers are `high`" holds by construction.
   - The Phase 5 check that interview levels separate the ratings is a consistency check, not proof.
4. **Not every test was pre-registered.** Phases 3 and 5 (confidence, clarity, latency, trends) were written down before they ran. The two main skills were tested more informally in Stages 7–8, before pre-registration became the rule.

**So:** both skills track independent human ratings, and that does not come from the table. The exact correlations are probably a little high.

**What would settle it,** using data that played no part in any decision:
1. The team's own interview recordings, rated by people: the cleanest test.
2. Two or three people labelling a few hundred answers as hesitant / normal / fluent and flat / normal / lively. This would also show for the first time how often a single answer's level is right, and could be used to train a classifier on top of the existing measurements.
3. The 60-answer listening check (`stage8_detectors/listening_check.csv`): the cheapest first step. It does not measure accuracy, but it catches obvious failures.

---

## 6. From a number to a level: references and settings

A percentile depends on who you compare with. Live interviews have more filled pauses, more long pauses and less spread in activation than self-recorded clips. Against recorded clips, 45% of live answers were "hesitated a lot". So there are **two reference settings** (Phase 2, `stage10_reference_settings/`):

| Setting | Reference population | Used for |
|---|---|---|
| `recorded_answer` | RecruitView train answers (unique recordings) | Self-recorded clips. The recording-quality gate applies: answers in the worst 10% of quality get no level |
| `live_interview` (the default) | MIT answers. Each MIT student is scored against the **other** 68 students ("leave-one-student-out"), so their own answers never define their reference | Live interviews. The quality gate is reported but not applied: MIT's far-field audio would fail it 91% of the time |

**Results:**
- With this, `low` and `high` each cover 18–21% of answers for every dimension in both settings.
- The MIT validation numbers did not drop (hesitation −0.58, expressiveness +0.65 after controls).

**`level_cuts.json`** (`stage8_detectors/`) holds everything needed to score a new answer, per setting:
- 201 quantiles per component;
- the voice-pitch adjustment line: intercept, slope per Hz, and a fallback pitch.

**The quality gate** (Stage 6, `fit_quality_gate.py`, `quality_gate.json`):
- A composite of three scores: SQUIM's estimated speech quality (PESQ), estimated intelligibility (STOI) and signal-to-noise ratio.
- Each is turned into a percentile among the train recordings, and the gate is set at the train 10th percentile.

---

## 7. The interview level

Phase 5, `scripts/explainability/interview_summary.py`, results in `results/explainability/stage14_interview_level/`. Pre-registered; CPU only, about 1 minute.

**What it builds per interview** (`interview_summaries.jsonl`, one line per MIT interview, 138 in total):

| Field | Meaning |
|---|---|
| `level` per dimension | The interview's mean index as a percentile among the other students' interviews, cut at 20 / 80 |
| `say` | One sentence, only at the clear ends: "hesitated a lot across the interview", "the voice sounded quite flat throughout the interview", "the delivery was lively throughout the interview" |
| `point_at` | The two answers worth naming. Hesitation: most and least hesitant. Expressiveness: flattest and liveliest |
| `feedback_at` | `interview` for expressiveness (spoken about once), `answer` for hesitation |
| `trend`, `trend_use`, `trend_say` | See the tests below |
| `hesitation.slow_to_start` | Fluency sub-item: level of the median response latency; `say` only for `high` |
| `internal` | Means, percentiles, trend sizes and p-values. Never shown to the LLM |

**Consistency check.** The interview level separates the ratings as it should:
- Mean `NoFillers` is 5.06 / 4.14 / 3.26 for low / normal / high hesitation.
- Mean `EngagingTone` is 3.82 / 4.90 / 5.59 for low / normal / high expressiveness.

**Trend tests.** A trend is the last two answers minus the first two, after removing each question's typical value: Q1, "about yourself", runs about 5 points lower on both dimensions.

| Test | Hesitation | Expressiveness |
|---|---|---|
| T1: is the change bigger than shuffling the answer order would give? (permutation test, see glossary) | **No.** 7.5% of interviews have a trend; the null's 95th percentile is 8.2% | **Yes**, barely. 8.3% against a null p95 of 6.8% (11 interviews) |
| T2: does the trend relate to the ratings once the average is known? | Opposite to expectation: interviews that got *more* hesitant at the end were rated better (+0.21 `NoFillers`, +0.24 `Overall`). Possibly raters weigh the first answers more. Not used | No (+0.04, −0.02) |
| Decision | No hesitation trend is ever stated | Description only ("livelier in the last two answers than in the first two"), never as how the interview came across |

**Audio-only interview baseline.** A Ridge model from the interview aggregates to each reliable MIT rating, out-of-fold by student. For `Overall` it beats the best single feature (expressiveness mean, 0.52). Predictions are in `tables/ridge_baseline_oof.csv`.

| Rating | Spearman |
|---|---|
| `Overall` | 0.60 [0.44, 0.72] |
| `RecommendHiring` | 0.61 |
| `EngagingTone` | 0.68 |
| `Excited` | 0.69 |
| `NoFillers` | 0.58 |
| `NotAwkward` | 0.47 |
| `Calm`, `Authentic` | ~0 |

**`interview_cuts.json`** holds the reference interview means used for new recordings:
- `live_interview`: the 138 MIT interviews;
- `recorded_answer`: RecruitView train speakers' per-person means;
- the per-question means;
- the "use" decisions above.

---

## 8. Running on a new recording

Phase 4, results in `results/explainability/stage12_new_recordings/`. No dataset table is needed at run time, only three files: `level_cuts.json`, `quality_gate.json` and `validation_mit.csv`, which decides which dimensions may be used. Phase 5 adds `interview_cuts.json`.

| Script | In | Out |
|---|---|---|
| `scripts/explainability/analyze_answer.py` | One audio or video file + the question text | The evidence JSON for that answer |
| `scripts/explainability/segment_interview.py` | A full interview recording | The five answers: question text, start, end |
| `scripts/explainability/analyze_interview.py` | A full interview recording | `{name}.json`: every answer's evidence and the `interview_summary`. Also `{name}_measures.csv`, internal |
| `scripts/explainability/test_new_recording.py` | — | Phase 4's two tests |

**Steps inside `analyze_answer.py`:**
1. Load and resample to 16 kHz mono (ffmpeg for video).
2. Silero VAD.
3. Whisper `medium.en` word times.
4. Praat pitch.
5. Sentences.
6. The emotion model.
7. The two disfluency classifiers.
8. The quality checks.
9. Percentiles and levels from `level_cuts.json`.
10. Evidence.

Models are loaded one at a time and freed. A 3-minute interview takes 40–90 s and peaks at 1.86 GB of GPU memory, so the 6-GB laptop card (RTX A3000) is enough.

**Test 1: one answer reproduces its dataset row. Pass.**
- 30 validation answers were run from their audio alone. Components match to rounding error, and levels match except for 3 answers within 2 points of a cut.
- Without the speaker's other answers, expressiveness moves (2 of 30 levels change). That is why `analyze_interview.py` takes the speaker's pitch from all of their answers.

**Test 2: cutting the answers out of an interview without the human transcript. Fail.**

| Run | Answers found (bar 90%) | Within 3 s (bar 85%) |
|---|---|---|
| Attempt 1: Whisper + `microsoft/wavlm-base-plus-sv` speaker embeddings per sentence, k-means into 2 speakers, interviewer = whoever speaks less | 73.6% | 63.9% |
| Attempt 2, held-out students: 1.5-s windows, and sentences that end in "?" and match a question pattern forced to the interviewer | 75.2% | 58.5% |
| The same answer rules with the human speaker labels (the ceiling) | 94.5% | 88.1% |

The answer rules work; the speaker model is the weak link. The fix is either:
- `pyannote/speaker-diarization-3.1`, which needs a Hugging Face token; or
- speaker turns from the shared upstream.

This is **deferred** (decision of 16 Nov). Until then, answer boundaries on new interviews are right for about 6 in 10 answers, and latency on new recordings is unreliable.

**Run** (use the project environment, `.venv`; the HF certificate fails on this network, hence offline mode):

```powershell
$env:HF_HUB_OFFLINE = "1"
.venv\Scripts\python.exe scripts/explainability/analyze_interview.py path\to\interview.wav --name P1
.venv\Scripts\python.exe scripts/explainability/analyze_answer.py path\to\answer.wav --question "Tell me about yourself" --setting recorded_answer
```

---

## 9. Score prediction models (the "legacy" work and what is kept)

Before the explainability work, the branch tried to **predict the RecruitView scores** from audio. Scripts are in `scripts/legacy/`, results in `results/audio_score_models/`, and some outputs in `outputs/audio/reports/` (git-ignored).

| Experiment | Script | Result |
|---|---|---|
| Handcrafted features (27 original + 36 new: pitch, energy, pauses, rate) with Ridge / SVR / Random Forest / ExtraTrees | `extract_audio_features.py`, `compare_handcrafted_improvements.py` | Adding the 36 features gave no consistent gain. Later found: the pitch tracker was home-made and makes octave errors on low voices |
| eGeMAPS (a standard set of 88 acoustic features from openSMILE) | `extract_audio_features.py`, `compare_audio_models.py` | Spearman ≈ 0.36–0.41, the weakest |
| WavLM / HuBERT layer sweep: every hidden layer × mean / std / mean+std pooling | `run_ssl_layer_sweep.py` | **WavLM-base-plus layer 7, mean+std pooled**, chosen for all targets. HuBERT is no better |
| Speech-only pooling via Silero VAD, pause features | `run_vad_pause.py` | Did not help: the silences carry information. Its VAD cache is reused by the explainability pipeline |
| Repeated grouped CV (5 repeats × 5 folds by person, nested hyperparameter search) | `run_repeated_grouped_cv.py` | **Benchmark:** WavLM L7 + Ridge 0.499 (confidence), 0.493 (speaking), 0.526 (overall). Handcrafted 0.45–0.50; eGeMAPS 0.36–0.41. At the label ceiling |
| GRU over WavLM frames | `train_wavlm_gru.py`, `run_wavlm_gru_ablation.py`, `run_wavlm_gru_pooling_pairs.py` | Overfitted. Train out-of-fold Spearman about 0.2, validation about 0 (`outputs/audio/reports/wavlm_gru_summary.json`) |
| Attention pooling over WavLM frames | `run_wavlm_attention_experiments.py` | Run interrupted, no result |
| Diagnostics | `diagnose_audio.py` | Label and extraction diagnostics |

**Lessons:**
- More architecture cannot help: RecruitView's labels agree with themselves only at 0.34–0.57.
- The score models lean on answer length, voice identity and recording setup.
- Interpretable delivery signals reproduce about 90% of WavLM's accuracy (Stage 3).

**What is kept for fusion.** In `export_for_fusion.py`, the same recipe (WavLM L7 + Ridge) is applied to **all 12 RecruitView targets**. These are the `pred_<target>` columns.

| Target | Train (out-of-fold) | Val | Test |
|---|---|---|---|
| openness | 0.50 | 0.42 | 0.47 |
| conscientiousness | 0.44 | 0.31 | 0.32 |
| extraversion | 0.49 | 0.36 | 0.54 |
| agreeableness | 0.50 | 0.38 | 0.50 |
| neuroticism | 0.35 | 0.24 | 0.25 |
| overall_personality | 0.51 | 0.39 | 0.47 |
| interview_score | 0.52 | 0.40 | 0.53 |
| answer_score | 0.52 | 0.36 | 0.47 |
| speaking_skills | 0.50 | 0.42 | 0.49 |
| confidence_score | 0.52 | 0.34 | 0.55 |
| facial_expression | 0.48 | 0.37 | 0.47 |
| overall_performance | 0.55 | 0.41 | 0.50 |

(Spearman; `results/explainability/stage13_fusion_export/score_model_metrics.csv`.)

---

## 10. For the fusion teammate: what to use and how

Everything is in **`results/explainability/stage13_fusion_export/`**. `SCHEMA.md` there is the short reference; this section explains what each output *stands for* and what to use it for.

### 10.1 The files

| File | One row / entry per | What it is |
|---|---|---|
| `audio_answers.csv` | Answer (2,011 RecruitView + 674 MIT) | All per-answer audio outputs as numbers and levels. **The main file** |
| `audio_embeddings_wavlm_l7.npz` | Answer (same order) | A 1,536-number summary of how each answer sounds, from WavLM (see 10.3) |
| `evidence.jsonl` | Answer | The LLM-facing JSON (section 4.7). Not a fusion input |
| `score_model_metrics.csv` | Target (12) | How well each `pred_*` column predicts its label |
| `../stage14_interview_level/tables/interview_aggregates.csv` | MIT interview (138) | The per-interview audio summary plus MIT ratings |
| `../stage14_interview_level/tables/ridge_baseline_oof.csv` | MIT interview | An audio-only prediction of each MIT rating, the baseline fusion should beat |

### 10.2 Every column of `audio_answers.csv`

**Keys: how to join with the text and video branches**

| Column | Stands for |
|---|---|
| `answer_id` | The answer. RecruitView: the dataset `id` as text, keeping the leading zeros (`0008`). MIT: `MIT_<interview>_<Q1..Q5>` |
| `dataset` | `recruitview` or `mit` |
| `split` | `train` / `val` / `test`, exactly as `Datasets/Split Dataset/`; `mit` for MIT |
| `person_id` | Who is speaking. RecruitView `user_no`; MIT `MIT<n>`, the same student for their pre and post interviews |
| `question_id` | RecruitView question id; MIT `Q1`–`Q5` |
| `is_duplicate`, `duplicate_of` | RecruitView has 106 byte-identical recordings under different ids, rated separately. Copies get identical audio outputs; `duplicate_of` names the kept copy |
| `interview_id`, `session` | MIT only: which interview (`P1`, `PP1`, ...) and whether it was `pre` or `post` coaching |

**Audio score predictions: "what would a RecruitView expert score this answer, judging from the audio alone"**

| Column | Stands for |
|---|---|
| `pred_openness`, `pred_conscientiousness`, `pred_extraversion`, `pred_agreeableness`, `pred_neuroticism`, `pred_overall_personality` | Predicted personality ratings |
| `pred_interview_score`, `pred_answer_score`, `pred_speaking_skills`, `pred_confidence_score`, `pred_facial_expression`, `pred_overall_performance` | Predicted performance ratings. Same scale as the RecruitView labels |

How they were made:
- The WavLM embedding (10.3) goes into one Ridge regression per target (a linear model with a penalty that keeps it from overfitting), trained on RecruitView train people.
- **Train rows are out-of-fold:** each train answer was predicted by a model that never saw that person. That makes them safe to train fusion on.
- Val, test and MIT rows come from one model trained on all train people.

Important:
- **The 12 are effectively one number** (94% shared; median correlation 0.97; `pred_neuroticism` is the same signal flipped). Use one, e.g. `pred_overall_performance`.
- It tracks answer length (−0.6: shorter answers score higher).
- `pred_facial_expression` and the personality columns are not "audio measures of the face or personality": they are the same general signal.
- `pred_confidence_score` must never be used as a confidence measure outside RecruitView.

**Feedback dimensions: the validated soft skills (section 4)**

| Column | Stands for |
|---|---|
| `setting` | Which reference the levels are computed against: `recorded_answer` (RecruitView) or `live_interview` (MIT) |
| `hesitation_index` | Fluency, 0–100, higher = more hesitant. The combined percentile of the four behaviours below |
| `hesitation_level` | `low` (≤ 20) / `normal` / `high` (≥ 80) / `missing` |
| `hesitation_withheld` | True when recording quality was too poor to trust the level (RecruitView only) |
| `hesitation_reasons` | Words for a `high` level, e.g. "many filled pauses" |
| `expressiveness_index` | Expressiveness, 0–100, higher = livelier, after the voice-pitch correction |
| `expressiveness_level`, `_withheld`, `_reasons` | As for hesitation (`low` = flat, `high` = lively) |

**The measurements behind them**

| Column | Stands for | Unit |
|---|---|---|
| `p_interjection` | Average probability that a 3-s window contains a filled pause ("um") | 0–1 |
| `p_repetition_block` | Average probability of a repeated word plus a block (getting stuck) | 0–1 (two class probabilities of one model, added) |
| `sf_p_disfluent` | Second classifier's average probability of "disfluent" | 0–1 |
| `long_mid_pause_rate_per_min` | Silences ≥ 1 s inside sentences, per minute of speaking | count / min |
| `activation` | Average vocal liveliness from the emotion model, before the pitch correction | about 0–1 |
| `activation_adj` | The same after removing the voice-pitch effect | difference, around 0 |
| `pitch_std_st` | How much the pitch moves over the answer | semitones |

**Context: never shown to the user**

| Column | Stands for |
|---|---|
| `quality_composite` | Recording quality, 0–100: the percentile among RecruitView train recordings of estimated speech quality, intelligibility and signal-to-noise |
| `speech_span_s` | Seconds from the first to the last speech in the answer: answer length |
| `acoustic_ok` | Whether the recording passes the quality gate |
| `speaker_f0_hz` | The speaker's typical pitch in Hz. Used only for the expressiveness correction. Do not use it as a feature to judge people on: it mainly encodes gender |

### 10.3 The embedding

`audio_embeddings_wavlm_l7.npz` contains `answer_id` and `embedding` (2,685 × 1,536).
- **WavLM** is a large speech model trained by Microsoft on many hours of audio. Its internal layer 7 describes the sound of speech: voice, rhythm, energy and so on.
- Per answer, the layer's output is averaged over time (768 numbers) and its spread over time is added (another 768).
- It is what the `pred_*` columns are built from. Use it only if you want to learn your own audio model inside fusion. It is high-dimensional and also encodes who is speaking and the microphone, so it overfits easily on 231 training people.

### 10.4 What to use for what

| Goal | Use | Don't use |
|---|---|---|
| **Late fusion predicting RecruitView scores** (per answer) | `pred_overall_performance` (or the target-matched `pred_*`, but only one of them), `hesitation_index`, `expressiveness_index`. Possibly `quality_composite` as a control | All 12 `pred_*` together (one signal), `speaker_f0_hz`, the levels (they are cut versions of the indices) |
| **Fusion that also produces feedback** | The levels and the evidence JSON, passed through to the LLM | `pred_*`. They are scores, not observations |
| **Perceived confidence** (fusion's job now) | Combine `hesitation_index`, `expressiveness_index` and the text and video signals, and check the result on MIT's `Overall` / `NotAwkward`, not just on RecruitView | `pred_confidence_score` alone: it is contradicted on MIT |
| **Interview-level fusion (MIT)** | `interview_aggregates.csv`, keyed by `interview` / `person`; the baseline in `ridge_baseline_oof.csv` | — |
| **Answer length** | `speech_span_s`, if the text branch doesn't carry it already | Counting it twice through `pred_*` and text |

**How to train it without leakage.**
1. Join the three branches on `answer_id`.
2. Fit on `split == "train"` rows only, with cross-validation grouped by `person_id`.
3. Choose settings on `val`, report once on `test`.
4. MIT is an outside check: it has no RecruitView labels, but its 16 interview ratings can test whether a fused judgement (e.g. confidence) means the same thing outside RecruitView.

**Load:**

```python
import numpy as np, pandas as pd
a = pd.read_csv("results/explainability/stage13_fusion_export/audio_answers.csv",
                dtype={"answer_id": str, "person_id": str, "duplicate_of": str})
z = np.load("results/explainability/stage13_fusion_export/audio_embeddings_wavlm_l7.npz")
emb = pd.DataFrame(z["embedding"], index=z["answer_id"])        # emb.loc[a["answer_id"]]
X = a[["pred_overall_performance", "hesitation_index", "expressiveness_index"]]
train = a["split"] == "train"                                    # pred_* here are out-of-fold
```

**Missing values.**
- 46 answers (42 RecruitView, 4 MIT) have no hesitation / expressiveness index: too little speech.
- 22 RecruitView answers have no `quality_composite`.
- Fill them with the training median plus a missing-flag, or drop those rows.

### 10.5 What still has to be agreed between branches

- One shared evidence format for all three branches. The audio version is section 4.7.
- The same answer ids and the person-level split: done on the audio side.
- Who owns the shared upstream (speaker turns, segments, one transcript). Its speaker turns would also fix audio's segmentation of new recordings.

---

## 11. Everything that was tried, in order

Kept: ✅. Kept with limits: ⚠️. Dropped or failed: ❌. Stage folders are under `results/explainability/`, scripts under `scripts/explainability/` unless noted.

| # | What | Where | Outcome |
|---|---|---|---|
| L1 | Handcrafted features + classic models | `scripts/legacy/`, `results/audio_score_models/handcrafted_improvements/` | ❌ No gain from more features. The pitch tracker was faulty |
| L2 | eGeMAPS | legacy | ❌ Weakest (0.36–0.41) |
| L3 | WavLM / HuBERT layer sweep | `ssl_layer_sweep/` | ✅ WavLM L7 mean+std chosen |
| L4 | VAD speech-only pooling, pause features | `vad_pause/` | ❌ No gain. ✅ The VAD cache is reused |
| L5 | Repeated grouped CV benchmark | `repeated_cv_v1/` | ✅ WavLM L7 + Ridge 0.50–0.53, at the label ceiling. Its folds are reused in Stage 3 |
| L6 | GRU over WavLM frames, ablations, pooling pairs | `outputs/audio/reports/wavlm_gru_*` | ❌ Overfitted (validation ≈ 0) |
| L7 | Attention pooling | `outputs/audio/reports/wavlm_attention_experiments/` | ❌ Interrupted, no result |
| 0 | Confound audit: what do RecruitView scores reward? | `stage0_confound_audit/` | Length is penalised within a person (−0.12). Loudness and absolute pitch identify speaker or microphone. Question type changes length, not scores |
| 1 | 50+ delivery descriptors per sentence and answer (rate, pauses, pitch, endings, ...), Praat pitch, verified against Whisper and a second tracker | `stage1_descriptors/` | ✅ 7 extraction bugs fixed. ⚠️ Listening check still open |
| 2 | Reference percentiles; 7 categories (Pace, Pausing/fluency, Expressiveness, Projection, Composure, Assertiveness, Conciseness); flags (`too_fast`, `long_answer`, `monotone`, ...); evidence JSON v0 | `stage2_reference/` | ⚠️ Superseded for feedback by Stage 8 levels. Kept as descriptive tables |
| 3 | RecruitView validation, concept bottleneck (how much of WavLM interpretable signals explain), rater-noise analysis | `stage3_recruitview_validation/` | The signals reach ~90% of WavLM (~97% with length), but almost nothing holds within a person. RecruitView ranks people, not answers |
| 4 | Verbatim ASR (Whisper `medium.en` word times), descriptors re-measured on real sentences | `stage4_asr/` | ✅ `words.csv` goes to the text branch. ❌ Whisper misses ~93% of fillers. ❌ ASR confidence as a clarity measure rejected |
| 5 | Speech-emotion models (audeering, Odyssey) as composure / confidence | `stage5_ser/` | ❌ As composure or confidence: it measures activation, and raters disagree on its direction. ✅ Later reused as the expressiveness detector |
| 6 | Recording-quality gate (SQUIM + SNR) | `stage6_quality/` | ✅ Loudness is a recording artefact, so there is no loudness feedback. The gate is applied for recorded answers |
| 7 | MIT validation: answers cut by aligning ASR to the Turker transcripts (674 answers), label reliability, pre-registered tests | `stage7_mit_validation/` | ✅ Expressiveness (pitch) and length supported. ❌ Composure not supported, `too_fast` contradicted. ⚠️ Pausing is mostly halo |
| 8 | Per-answer detectors and levels: vocametrix and speech-flow disfluency classifiers, CrisperWhisper (filler words, text hand-over), confidence detector | `stage8_detectors/` | ✅ Hesitation (−0.58) and expressiveness (+0.63) after controls. ❌ Confidence contradicted (−0.58). ✅ CrisperWhisper filler rate agrees with the human transcripts at +0.76, for the text branch |
| 9 | Evidence JSON v2 | `stage9_feedback/` | ✅ Only validated, audio-only observations, each with a time span and no words |
| P1 | Phase 1: audio-only fluency, activation-based expressiveness, voice-pitch adjustment, length removed | `stage8_detectors/` | ✅ Done. Low voices were "flat" 39% vs 9% before the adjustment; much evener after |
| P2 | Phase 2: one reference per setting | `stage10_reference_settings/` | ✅ `low` / `high` at 18–21% everywhere; MIT relations unchanged |
| P3 | Phase 3: style tagger (25 tags) and confidence retraining | `stage11_style_tagger/` | ❌ Confidence fails everywhere. ❌ Clarity measures accent. ⚠️ "Monotone" tag weakly supports expressiveness (−0.24); not used |
| P4 | Phase 4: new recordings, segmentation | `stage12_new_recordings/` | ✅ One answer reproduces its row; one command runs an interview. ❌ Segmentation fails its test (deferred) |
| P6 | Phase 6: fusion export | `stage13_fusion_export/` | ✅ Done for the audio side |
| P5 | Phase 5: interview level, latency, trends, baseline | `stage14_interview_level/` | ✅ Latency validated. ✅ Interview summaries. ❌ Hesitation trends are noise. ⚠️ Expressiveness trends are description only |
| — | Model searches on Hugging Face for confidence, stress, fluency, pronunciation, expressiveness (twice, last on 13 Nov) | `markdown/README.md` §6.6 | Nothing usable for confidence. Found the speech-flow classifier (used) and the style tagger (failed) |

---

## 12. All findings in one list

**About the data**
1. RecruitView's labels agree with themselves only at 0.34–0.57 (copied recordings rated twice). Models are already at that ceiling, so new architectures cannot show gains.
2. RecruitView's rating noise is as large as the differences between one person's answers. It cannot tell which answer was better, so it cannot validate per-answer feedback.
3. The two rater groups disagree about vocal energy: RecruitView's experts prefer calm, MIT's raters prefer energetic.
4. MIT's `SpeakingRate` (0.03) and `Paused` (0.26) ratings are unreliable, so pace and pausing cannot be validated against them.

**About the measures**

5. Answer length is the one delivery behaviour RecruitView raters penalise within the same person (−0.12). It is left to the text branch to judge.
6. Loudness and absolute pitch identify the speaker or microphone. Loudness's relation to scores disappears once recording quality is controlled.
7. Speech-emotion models measure the person (87% between-speaker) and cannot separate dominance from arousal (0.98). What they measure is vocal activation.
8. Whisper misses ~93% of fillers. CrisperWhisper catches them (+0.76 with the human transcripts), which matters for the text branch.
9. Activation follows voice pitch. Without an adjustment, low voices would be called flat far more often.

**About the skills**

10. **Hesitation works:** −0.58 with `NoFillers` after controls. Two classifiers plus long mid-sentence pauses beat any single component.
11. **Expressiveness works:** +0.65 with `EngagingTone` after controls. A second emotion model gives the same result.
12. **Being slow to start answering** (response latency) relates to lower ratings at interview level: −0.38 with `Overall`, still there after the hesitation index is controlled. It is part of fluency, not a separate skill.
13. **Confidence cannot be measured from these data.** Every model is null or contradicted on MIT, because of finding 3.
14. **Clarity can't be measured fairly:** the tagger rates Indian-English accent as slurred.
15. Pace (`too_fast`) is contradicted, composure is directionally ambiguous, pausing is mostly halo.

**About settings and new recordings**

16. Live interviews need their own reference. Against recorded clips, 45% of live answers would be "hesitant".
17. A single answer on a new recording reproduces the dataset pipeline exactly. Expressiveness is steadier when the speaker's other answers are available.
18. Cutting answers out of a new interview needs a real diarization model. With correct speaker labels the rules pass (94% / 88%); with the ungated speaker model they don't (75% / 59%).

**About the interview level**

19. Hesitation changes within an interview are no larger than shuffling the answers would give. Expressiveness changes slightly more than chance but add nothing to ratings beyond the average.
20. At the same average hesitation, interviews that became more hesitant at the end were rated better: raters may weigh the first answers more. Not pre-registered and not used.
21. An audio-only interview model reaches 0.60 with `Overall`, a baseline for fusion.

**About fusion**

22. The 12 audio score predictions are one signal (94% shared variance) that tracks answer length (−0.6).
23. Hesitation (0.02) and expressiveness (−0.39) carry information that this signal does not.

---

## 13. Folders and files: what is where, and is the legacy folder still useful?

### 13.1 Top level

| Path | What | In git? |
|---|---|---|
| `AUDIO_BRANCH.md` | This document | yes |
| `README.md` | Old README: describes only the legacy score models. Out of date | yes |
| `markdown/README.md` | The detailed plan (Phases 1–7), status tables, rules, file map | **no** (`/markdown` is git-ignored) |
| `markdown/EXPLAINABILITY_STAGES.md`, `EXPLAINABILITY_README.md` | Stages 0–8 step by step; early design and model catalogue (partly superseded) | no |
| `scripts/explainability/` | **The current pipeline.** Every path is defined in `paths.py` | yes |
| `scripts/legacy/` | Score-prediction experiments (section 9) | yes |
| `results/explainability/` | One folder per stage (0–14). Each has `report.md` (generated), `notes.md` (hand-written) and `tables/`. Index: `results/explainability/README.md` | yes, except `stage7_mit_validation/tables` |
| `results/audio_score_models/` | Legacy experiment results | yes |
| `outputs/` | Caches: embeddings, ASR, VAD, model windows, GRU and attention reports, legacy run | no |
| `Datasets/` | RecruitView, the fixed split, MIT | no |
| `config.yaml`, `requirements*.txt` | Settings and dependencies used by the legacy scripts | yes |
| `Utils/` | Old wrapper commands (`split_dataset.py` refuses to overwrite the fixed split) | yes |
| `archive/` | Original split generator and legacy benchmark | yes |
| `tests/test_pipeline.py` | Old tests for the legacy extractor. They import `scripts.extract_audio_features`, which moved to `scripts/legacy/`, so they probably fail as they are | yes |
| `Main.py` | A placeholder (prints "Hello, World!") | yes |
| `.venv/` | The Python environment with torch, transformers and the rest. The system `python` lacks `transformers` | no |

### 13.2 `scripts/explainability/`, by job

| Job | Scripts |
|---|---|
| Paths | `paths.py` |
| Stage 0–3: audit, descriptors, references, RecruitView validation | `stage0_confound_audit.py`, `make_question_types.py`, `extract_delivery_descriptors.py`, `verify_stage1.py`, `descriptor_tables.py`, `fit_reference_distributions.py`, `delivery_reference.py`, `build_evidence.py`, `stage3_validate_recruitview.py` |
| Stage 4: ASR | `run_verbatim_asr.py`, `extract_word_level_descriptors.py` |
| Stage 5: emotion models | `ser_models.py`, `run_ser_windows.py`, `extract_ser_descriptors.py`, `ser_model_agreement.py` |
| Stage 6: quality | `run_quality_scores.py`, `fit_quality_gate.py` |
| Stage 7: MIT | `mit_label_reliability.py`, `mit_segment_answers.py`, `mit_run_pipeline.py`, `validate_categories_mit.py` |
| Stage 8–9: detectors, levels, evidence | `run_crisper_asr.py`, `run_disfluency_classifier.py`, `voxprofile_models.py`, `run_speech_flow.py`, `run_confidence_detector.py`, **`fit_answer_levels.py`**, **`validate_detectors_mit.py`**, **`build_answer_evidence.py`**, `make_listening_list.py` |
| Phase 2 | `compare_settings.py` |
| Phase 3 | `run_style_tagger.py`, `analyze_style_tagger.py`, `run_confidence_partb.py`, `extract_partb_embeddings.py` |
| Phase 4 | **`analyze_answer.py`**, `segment_interview.py`, **`analyze_interview.py`**, `test_new_recording.py` |
| Phase 5 | **`interview_summary.py`** |
| Phase 6 | **`export_for_fusion.py`** |

Bold marks the scripts you are most likely to run or read.

### 13.3 The files that matter most

| File | Why |
|---|---|
| `results/explainability/stage8_detectors/tables/answer_levels.csv` | Every answer's components, indices and levels (both datasets) |
| `results/explainability/stage8_detectors/level_cuts.json` | Everything needed to score a new answer |
| `results/explainability/stage8_detectors/tables/validation_mit.csv` | The test verdicts; decides which dimensions may be used |
| `results/explainability/stage6_quality/quality_gate.json` | The quality gate |
| `results/explainability/stage9_feedback/evidence_v2.jsonl` | Evidence per answer |
| `results/explainability/stage13_fusion_export/` | Everything for fusion |
| `results/explainability/stage14_interview_level/interview_cuts.json`, `interview_summaries.jsonl` | Interview level |
| `results/explainability/stage4_asr/words.csv`, `stage8_detectors/tables/crisper_words.csv` | Transcripts with word times, for the text branch |

### 13.4 Is the legacy folder still useful?

**Partly. Keep it, but you don't need to run most of it.**

| Still needed | Why |
|---|---|
| `scripts/legacy/extract_audio_features.py` | It is the WavLM extractor behind the fusion embeddings and score predictions. `run_confidence_detector.py` imports it to embed MIT answers the same way RecruitView was embedded. `export_for_fusion.py` depends on that. It also reads `config.yaml` |
| `results/audio_score_models/vad_pause/features/` | The Silero VAD regions reused by Stage 1 and the quality gate |
| `results/audio_score_models/handcrafted_improvements/features/` | Read by the Stage 0 audit and the Stage 1 comparison |
| `results/audio_score_models/repeated_cv_v1/cv_splits.json` | The folds Stage 3 compared against |
| `results/audio_score_models/ssl_layer_sweep/selected_layers.json` | The reason layer 7 is used |

**Historical only** (the record of what was tried; nothing current runs them):
- `compare_audio_models.py`, `compare_handcrafted_improvements.py`, `run_ssl_layer_sweep.py`, `run_vad_pause.py`, `run_repeated_grouped_cv.py`, `diagnose_audio.py`;
- the GRU and attention scripts;
- `Utils/`, `archive/`, `tests/`, `Main.py`.

**Recommendation:** leave `scripts/legacy/` and `results/audio_score_models/` as they are; both are documented. If the repo is tidied later:
1. Move `extract_audio_features.py`'s WavLM extractor into `scripts/explainability/`.
2. Then the rest of `legacy/` can be archived.
3. Fix or delete `tests/test_pipeline.py`.
4. Replace `Main.py` with a real entry point (e.g. `analyze_interview.py`).

### 13.5 Re-running everything

The full chain is in `results/explainability/README.md` ("Re-running"). The parts most often re-run:

```powershell
.venv\Scripts\python.exe scripts/explainability/fit_answer_levels.py        # levels and level_cuts.json
.venv\Scripts\python.exe scripts/explainability/validate_detectors_mit.py   # MIT tests (~5 min)
.venv\Scripts\python.exe scripts/explainability/build_answer_evidence.py    # evidence JSON
.venv\Scripts\python.exe scripts/explainability/interview_summary.py        # Phase 5 (~1 min)
.venv\Scripts\python.exe scripts/explainability/export_for_fusion.py        # Phase 6 (~1 min)
```

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Activation (arousal / dominance)** | Outputs of a speech-emotion model. Arousal is how energetic or excited the voice sounds; dominance is how in-control it sounds. On this data they are the same thing, so their average is called activation |
| **Bootstrap confidence interval** | Resample the data with replacement many times (here, whole students) and recompute the statistic each time. The middle 95% of results is the 95% interval. If it excludes 0, the relation is unlikely to be chance |
| **Concept bottleneck** | Predicting the score only from interpretable measures, to see how much of a black-box model they explain |
| **Controls / partial correlation** | The correlation after removing the part of both variables that is explained by other variables (length, quality, voice pitch). It shows the relation is not just those |
| **Diarization** | Working out who speaks when in a recording with several speakers |
| **Embedding** | A vector of numbers a neural network produces to summarise audio. Here: WavLM layer 7, averaged over time (mean) plus its variation (std), 768 + 768 = 1,536 numbers |
| **F0 / pitch** | The fundamental frequency of the voice in Hz. Measured with Praat. "Semitones" put low and high voices on the same scale |
| **Fusion (late fusion)** | Combining the outputs of separate audio, text and video models at the decision level, rather than combining raw inputs |
| **GroupKFold / grouped cross-validation** | Cross-validation where all answers of one person are in the same fold, so the model is always tested on people it has never seen |
| **Halo** | Raters who like a candidate rate them well on everything, so a measure can correlate with a rating for the wrong reason |
| **ICC(2,k)** | Intraclass correlation: how well the average of k raters would agree with another set of k raters. Above about 0.75 is good; below 0.5 is unusable |
| **Index (0–100)** | A dimension's combined percentile for an answer. Internal only |
| **Label ceiling** | The best agreement possible given how noisy the labels are, estimated from duplicate recordings rated twice |
| **Leakage** | When a model is evaluated on data it indirectly saw during training, which makes results look better than they are |
| **Leave-one-student-out** | Each student is scored against a reference that excludes both of their own interviews |
| **Level (`low` / `normal` / `high`)** | The index cut at 20 and 80. Only `low` and `high` produce feedback |
| **Out-of-fold (OOF)** | A prediction made by a model that was trained without that data point (that person) |
| **Percentile** | The share of reference values below a given value, from 0 to 100 |
| **Permutation test** | Shuffle the order (here, of an interview's answers) every possible way, and see how often a change as large as the real one appears by chance |
| **Pre-registration** | Writing down the test, the expected direction and the pass rule before running it |
| **Praat** | Standard phonetics software used to measure pitch |
| **Reference population / setting** | The answers a new answer is compared with. `recorded_answer` = RecruitView train; `live_interview` = MIT |
| **Ridge regression** | Linear regression with a penalty that keeps weights small (`alpha` = penalty strength). Robust with many features and few examples |
| **Silero VAD** | Voice activity detection: marks which parts of the audio are speech |
| **Spearman ρ** | Rank correlation from −1 to +1: do higher values of one go with higher values of the other? Robust to outliers and scale |
| **SQUIM / PESQ / STOI / SNR** | Estimated speech quality, estimated intelligibility (both from torchaudio's SQUIM model, no clean reference needed), and signal-to-noise ratio |
| **WavLM / HuBERT** | Large self-supervised speech models from Microsoft and Meta. Their internal layers are used as general audio features |
| **Whisper / CrisperWhisper** | Speech recognition models. Whisper gives words with times but drops fillers; CrisperWhisper is a verbatim version that keeps "um / uh" |

---

## 15. What is still open

| Item | Needs | When |
|---|---|---|
| Fusion model | The teammate, using section 10 | Now |
| One full-pipeline demo (video → three branches → fusion → LLM) | Fusion done | After fusion |
| Speaker segmentation for new recordings (Phase 4, test 2) | A pyannote Hugging Face token, or speaker turns from the shared upstream | Deferred |
| Phase 7: `generate_feedback.py` (evidence → text), faithfulness check (every claim must be in the evidence), team review of ~20 feedbacks | LLM access; the dataset terms checked before sending transcripts to an external API | After fusion |
| Listening check of 60 flagged answers (`stage8_detectors/listening_check.csv`) | About an hour of listening | Any time |
| An independent test of both skills (section 5.1): the team's own rated recordings, or a few hundred answers labelled by two or three people | Recordings or listening time | Before claiming accuracy numbers outside the project |
| A quality gate and references for live interviews from the team's own recordings | Own recordings | Later |
| The shared three-branch schema | The text and video branches | With the team |
| Repo tidy-up (section 13.4) | — | Optional |
