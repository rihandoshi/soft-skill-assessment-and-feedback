# Text + Audio Fusion Results

Validation set, n = 319 responses · 30 Oct 2026

Fusing text and audio reliably improves ranking (Spearman ρ) over text-only for **confidence** and **overall performance**, by +0.06 to +0.07. But fusion is not reliably better than **audio-only** on those two targets, and no method reliably improves CCC. Simple averaging (mean / weighted) does as well as the learned stackers.

Source: [outputs/fusion_versions/v2_nonlinear/reports/fusion_val.md](../outputs/fusion_versions/v2_nonlinear/reports/fusion_val.md). The v2 run includes every v1 method plus the non-linear ones.

## Setup

- **Text branch:** fine-tuned DeBERTa-v3-base (v2). Train predictions are 5-fold out-of-fold.
- **Audio branch:** handcrafted features + WavLM. Train predictions are GroupKFold out-of-fold.
- **Fusion training:** every fusion weight and stacker is fit on train out-of-fold predictions. Hyperparameters are chosen by mean CCC over GroupKFold(5, shuffled, seed 42) grouped by participant (`user_no`).
- **Metrics:** Spearman ρ (ranking), CCC (agreement with the true scale) and RMSE.
- **Significance:** 95% participant-bootstrap confidence intervals on the paired difference against text-only and against audio-only. An interval that excludes 0 counts as a reliable difference.

## Methods compared

| Method | Type | What it does |
|---|---|---|
| text | single branch | DeBERTa prediction alone |
| audio | single branch | Audio-branch prediction alone |
| mean | late fusion | (text + audio) / 2 |
| weighted | late fusion | w·audio + (1−w)·text, with w chosen by CCC on train OOF |
| stack_pair | late fusion | Ridge on [text, audio] for the same target |
| stack_all | late fusion | Ridge on all 7 branch predictions (4 text targets + 3 audio targets) |
| stack_mlp | late fusion | Small MLP (16 hidden units) on the same 7 predictions |
| hybrid | early fusion | Ridge on DeBERTa predictions + handcrafted audio + WavLM (32 PCs) |

Weighted fusion leaned towards audio: w_audio = 0.70 for confidence, 0.75 for speaking skills and 0.60 for overall performance.

The tables leave out the weaker methods: early_text, early_concat, early_mlp and stack_gbm. Their numbers are in the source report.

## Results: Spearman ρ and CCC (higher is better)

| Method | Confidence ρ | Confidence CCC | Speaking ρ | Speaking CCC | Overall ρ | Overall CCC |
|---|---|---|---|---|---|---|
| **Text only** | 0.309 | 0.262 | 0.467 | **0.306** | 0.448 | 0.308 |
| **Audio only** | 0.363 | 0.315 | 0.429 | 0.278 | 0.505 | **0.338** |
| Mean | 0.364 ✓ | 0.299 | **0.484** ✓ᵃ | 0.302 | 0.511 ✓ | 0.333 |
| Weighted | 0.373 ✓ | 0.308 | 0.467 ✓ᵃ | 0.292 | 0.516 ✓ | 0.336 |
| Stack pair | 0.369 ✓ | 0.293 | 0.478 ✓ᵃ | 0.284 | 0.516 ✓ | 0.312 |
| Stack all | 0.372 ✓ | 0.301 | 0.465 | 0.286 | 0.513 ✓ | 0.313 |
| Stack MLP | 0.377 ✓ | 0.294 | 0.462 | 0.280 | **0.519** ✓ | 0.316 |
| Hybrid | **0.383** | **0.326** | 0.430 | 0.294 | 0.449 | 0.295 |

- **✓**: the Spearman gain over text-only is reliable (its 95% CI excludes 0).
- **ᵃ**: the Spearman gain over audio-only is reliable.
- **Bold**: best value in the column.

### Paired Spearman gains (fusion − single branch, 95% CI)

| Method | Confidence vs text | Confidence vs audio | Speaking vs text | Speaking vs audio | Overall vs text | Overall vs audio |
|---|---|---|---|---|---|---|
| Mean | +0.055 [+0.01, +0.10] | +0.001 [−0.05, +0.05] | +0.016 [−0.02, +0.05] | +0.055 [+0.01, +0.10] | +0.063 [+0.03, +0.11] | +0.005 [−0.04, +0.05] |
| Weighted | +0.064 [+0.01, +0.12] | +0.010 [−0.02, +0.04] | +0.000 [−0.06, +0.06] | +0.038 [+0.01, +0.07] | +0.068 [+0.02, +0.12] | +0.011 [−0.03, +0.05] |
| Stack pair | +0.061 [+0.01, +0.11] | +0.007 [−0.03, +0.05] | +0.011 [−0.04, +0.06] | +0.049 [+0.01, +0.09] | +0.068 [+0.02, +0.12] | +0.010 [−0.03, +0.05] |
| Stack all | +0.063 [+0.01, +0.12] | +0.009 [−0.03, +0.05] | −0.002 [−0.06, +0.05] | +0.036 [−0.00, +0.08] | +0.065 [+0.02, +0.12] | +0.007 [−0.04, +0.05] |
| Stack MLP | +0.068 [+0.01, +0.13] | +0.014 [−0.03, +0.05] | −0.005 [−0.07, +0.05] | +0.033 [−0.00, +0.07] | +0.071 [+0.02, +0.13] | +0.013 [−0.03, +0.05] |
| Hybrid | +0.074 [−0.01, +0.16] | +0.021 [−0.06, +0.10] | −0.037 [−0.12, +0.04] | +0.001 [−0.07, +0.07] | +0.001 [−0.08, +0.09] | −0.056 [−0.12, +0.00] |

## Results: RMSE (lower is better)

| Method | Confidence | Speaking | Overall | Answer score* |
|---|---|---|---|---|
| Text only | 1.003 | 1.102 | 1.071 | 1.020 |
| Audio only | 0.944 | 1.093 | 1.025 | — |
| Mean | 0.948 | 1.073 | 1.024 | — |
| Weighted | 0.940 | 1.076 | **1.020** | — |
| Stack pair | 0.942 | **1.072** | **1.020** | — |
| Stack all | 0.941 | 1.076 | 1.028 | **0.979** |
| Stack MLP | 0.942 | 1.073 | 1.026 | 0.983 |
| Hybrid | **0.935** | 1.074 | 1.046 | 0.986 |

\*The audio branch has no answer_score prediction, so only the stackers apply to it. stack_all had the best answer_score results: ρ 0.449 vs 0.425 for text, and CCC 0.319 vs 0.328. Neither difference is reliable.

## Analysis

- **Ranking improves, scale agreement doesn't.** Every late-fusion method reliably beats text-only on Spearman for confidence (+0.055 to +0.068) and overall performance (+0.063 to +0.071). None of the CCC gains over text are reliable.
- **Most of the gain comes from audio.** On confidence and overall, every fusion gain over audio-only is within ±0.02 and not reliable. The learned weights (w_audio 0.60–0.75) also lean towards audio.
- **Speaking skills behaves the other way round.** Text is the stronger branch here. Fusion reliably beats audio-only (mean, weighted, stack_pair) but never beats text-only.
- **Simple methods are as good as complex ones.** Mean and weighted averaging match the Ridge and MLP stackers, and stack_gbm is worse. Weighted is the most consistent across all three targets.
- **Hybrid is unstable.** It has the best confidence numbers (ρ 0.383, CCC 0.326, RMSE 0.935) but drops to text-level performance on overall (ρ 0.449).
- **Answer length explains part of the signal.** A length-only baseline already reaches ρ 0.45–0.49, and the branches correlate strongly with word count (−0.74 to −0.86 on train OOF). Spearman after removing log word count ("partial | len") rises from 0.05–0.08 for text to 0.15–0.23 for fusion, so fusion does add signal beyond length.
- **The branches overlap.** Text and audio predictions correlate at 0.70–0.77 on train OOF, which limits how much fusion can add.

**Recommendation:** use **weighted late fusion** as the final model for confidence and overall performance. It reliably beats text, matches the best stackers, and has a single, interpretable parameter. For speaking skills, text-only or mean fusion are equivalent.

## Caveats and next steps

- These are validation results only. The test set has not been evaluated yet (`fuse.py --eval-test`). Run it once, after choosing the final method on validation.
- n = 319 responses gives wide CIs: most method-to-method differences are within ±0.02 and cannot be separated.
- Consider reporting the "partial | len" metric alongside Spearman so that length effects are visible.
