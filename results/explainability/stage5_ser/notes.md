# Stage 5 notes: what the pretrained speech-emotion model does and does not tell us

Hand-written interpretation; `report.md` is generated and overwritten by `extract_ser_descriptors.py`.

## What was done

- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (trained by others on MSP-Podcast, listener-rated arousal /
  dominance / valence), used as-is: **no training, no fine-tuning**. 27,811 windows (4 s, 2-s hop), ~20 min on the
  laptop GPU (`run_ser_windows.py`).
- Every window is also expressed relative to the same speaker's other recordings (leave-one-out, duplicates excluded).
- Per answer: mean arousal / dominance / valence, plus timestamped moments: lowest-dominance stretch, largest
  dominance drop, longest run of "more activated AND higher-pitched than usual" (`extract_ser_descriptors.py`).
- Stage 2 builds percentiles, the composure class and three experimental flags; Stage 3 tests them against the scores.

## Findings

1. **The raw output mostly describes the person:** ICC 0.87 for arousal and dominance (jitter / shimmer: 0.80-0.85).
   Only speaker-relative values are used for classes and flags.
2. **Arousal and dominance are one dimension on this data** (window Spearman 0.98; speaker-relative 0.94). The model
   cannot separate "sounds assured" from "sounds energetic": it measures **vocal activation**. MSP-Podcast listeners'
   own ratings of the two are strongly correlated, and the model inherits that.
3. **RecruitView raters reward calm, not dominance.** Between people, higher predicted dominance / arousal goes with
   *lower* confidence scores (rho -0.50; -0.35 after duration, question type and video quality). Within a person the
   relation is weak but in the same direction, and `tension_rise` (activated + higher-pitched than usual) is weakly
   but consistently linked to lower scores on all three targets (rho -0.06 to -0.08, CIs exclude 0).
   Reading "dominance" as "sounds confident" would have produced the **opposite** of the advice the ratings support.
4. Activation also tracks livelier delivery (energy variability +0.47, fewer pauses -0.36, faster speech +0.26) and,
   oddly, recording setup (higher in low-quality videos).

## Decisions

- **Composure class re-oriented:** speaker-relative arousal; "calmer than usual / usual composure / more activated /
  tense than usual", attention pole = more activated. The orientation comes from the between-person RecruitView
  evidence, so the within-person re-test in Stage 3 is only partly independent; MIT `Calm` / `NotStressed` (Stage 7)
  is the real test.
- Stage 3 re-run: `composure_index` is now **weak** in the expected direction within participants (rho -0.07 / -0.05 /
  -0.05 for confidence / speaking / overall; CI excludes 0 for confidence). The interim jitter/shimmer proxy was "drop".
- `tension_rise`: experimental, weak but consistent rater support. Cannot tell tension from enthusiasm.
- `low_dominance`: contradicted by the ratings; kept only for MIT validation. Never advise "sound more dominant".
- `composure_drop`: no rater relation; threshold compared within the same answer-length bucket (1 SD alone fired on 70%).
- Nothing from Stage 5 reaches the LLM yet (all experimental).

## Still open

- **Second SER model** (`3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes`) for the agreement check: needs
  `trust_remote_code=True` (runs third-party Python), so it was not run without approval. Alternative: re-implement its
  head locally and load only the weights.
- MIT validation (Stage 7): `run_ser_windows.py` runs on any clip list.
- Phrasing (Stage 9): describe the voice ("your voice got faster and higher from 0:20"), never the inner state
  ("you were nervous").
