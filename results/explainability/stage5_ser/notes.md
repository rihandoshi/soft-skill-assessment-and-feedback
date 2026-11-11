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

## Second model: agreement check (`model_agreement.md`)

`3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes` (WavLM-large). Its repo needs `trust_remote_code=True`; instead
the architecture was re-implemented in `ser_models.py` from the repo's `pipeline_utils.py` (read, not run) and only
`model.safetensors` is loaded (strict). Same windows, ~25 min.

- **Answer means agree well** (Spearman 0.90-0.92 arousal / dominance); **within-speaker agreement is moderate**
  (0.73-0.75), and that is the part feedback uses. Valence agrees less (0.71-0.74).
- Arousal-dominance is 0.98 in this model too: "one activation dimension" is not an artefact of audeering.
- `tension_rise` kappa 0.63 (73% of audeering's flags also fire with odyssey); composure extremes kappa 0.49-0.58,
  opposite-class disagreements < 1%.
- **Rater findings replicate:** between people both models give rho -0.46 to -0.50 with confidence; within a
  person both are weakly negative (odyssey slightly stronger for arousal / dominance, weaker for `tension_rise`).
- Implication for Stage 8/9: a moment both models agree on is stronger evidence than one model alone; requiring
  agreement for SER-based feedback would cut flags by roughly a quarter. Not wired in yet.

## Still open

- MIT validation (Stage 7): `run_ser_windows.py` runs on any clip list.
- Phrasing (Stage 9): describe the voice ("your voice got faster and higher from 0:20"), never the inner state
  ("you were nervous").
