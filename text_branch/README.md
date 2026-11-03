# RecruitView text branch

Predicts RecruitView interview labels from **question + answer transcript** for each response, and
produces interpretable text features and embeddings for later fusion with the audio and video branches.

## Targets

| Label (RecruitView column) | Role |
|---|---|
| `answer_score` | Primary text target: relevance, coherence and structure of the answer |
| `confidence_score` | Shared with audio branch |
| `speaking_skills` | Shared with audio branch |
| `overall_performance` | Shared with audio branch |

`facial_expression` (video's target), `interview_score` (optional, overlaps `overall_performance`) and the
Big Five traits are not predicted. Predictions are model estimates of annotator ratings, not measurements
of a person.

## Inputs and data rules

- The split comes from the shared participant-level files in `split_dataset/` (train 1394 / val 319 / test 298
  responses; 231 / 49 / 51 participants). `load_splits()` asserts there is no participant overlap and no duplicate IDs.
- Model input is the question plus the Whisper-large-v3 transcript with timestamps removed.
- **`gemini_summary` is never used.** An LLM wrote it after watching the video, and it discusses traits, so it would leak label information.
- Delivery features (speech rate, pauses, fillers, repetitions) are left to the audio branch. Whisper also
  removes most "um/uh", so text would undercount them.
- Model selection uses validation only. The test split is not evaluated until the final model is chosen
  (`--eval-test` / `summarize_results.py --split test`).

## Run (from the project root)

```bash
python text_branch/build_text_features.py      # features + sentence embeddings (~1 min, downloads all-mpnet-base-v2 once)
python text_branch/train_text_baselines.py     # baselines + Ridge models, CPU (~5 min)
HF_HUB_OFFLINE=1 python text_branch/finetune_text.py   # model 5, DeBERTa-v3-base, 3 seeds (~15 min on RTX 4050)
HF_HUB_OFFLINE=1 python text_branch/finetune_text.py --loss mse_ccc --length-input --full-train --oof --tag v2
                                               # revised DeBERTa + out-of-fold train preds for fusion (~1-2 h)
python text_branch/summarize_results.py        # combined validation table
python text_branch/analyze_text.py             # paired tests vs length, recalibrated CCC, within-question
                                               # and beyond-length signal, ensemble, length decomposition
```

`build_text_features.py` also stores `clip_seconds` (read from `outputs/text/wav/`) when the WAVs exist.
It is used only by the `duration_only` diagnostic baseline and the length decomposition, never as a text
model input: clip timing belongs to the audio branch.

Dependencies: pandas, numpy, scipy, scikit-learn, pyarrow, torch, transformers>=5, sentence-transformers,
sentencepiece. With transformers>=5 the DeBERTa encoder must be loaded in fp32; this is already handled in
`finetune_text.py`, because fp16 weights make training diverge to NaN.

## Models

| Model | Input | Learner |
|---|---|---|
| `mean` | none | training-set mean (floor) |
| `question_mean` | question id | training mean of that question (question-effect floor) |
| `length_only` | log word count | Ridge |
| `duration_only` | log clip seconds (diagnostic, not a text feature) | Ridge |
| `linguistic` | 13 interpretable features (below) | Ridge |
| `embedding` | all-mpnet-base-v2 answer + question embeddings (1536-d) | Ridge |
| `embedding_ling` | linguistic features + answer embedding reduced to 32 principal components | Ridge |
| `finetune_deberta-v3-base` | "question [SEP] answer" | fine-tuned DeBERTa-v3-base, 4 joint regression outputs, 3-seed mean |

- **Ridge:** one model per target. The regularisation strength (alpha) is chosen by participant-grouped
  5-fold cross-validation on train (criterion: CCC). Training labels are winsorised at the train 1st/99th
  percentile; evaluation uses raw labels.
- **DeBERTa:** all epochs run, and the best one is picked on a participant-grouped 15% of train (mean CCC
  over the four targets), so validation stays clean. The `_v2` run changes four things:
  1. MSE + 0.5 × CCC loss replaces SmoothL1.
  2. The standardised log word count of the full answer is concatenated to the pooled vector, because mean
     pooling hides length.
  3. Each seed is retrained on all of train for the epoch count that is best on the seed-averaged
     selection curve.
  4. Out-of-fold train predictions average all seeds and use the same recipe as val/test, so a fusion layer
     fitted on them sees the same prediction distribution.

  Truncation is not an issue: no input exceeds 384 tokens (train median 87, 90th percentile 168).

**Linguistic features** (observable wording, not internal states): log word count, sentence count, mean
sentence length, mean word length, vocabulary variety (MATTR-50), hedges per 100 words ("I think", "maybe",
...), connectives per 100 words ("because", "for example", ...), numbers per 100 words ("first"/"second" count only as connectives; "no one" and "one of" are not numbers), "I"- and "we"-words
per 100 words, keyword overlap with the question, question-answer embedding cosine, and an empty-transcript flag.

## Validation results (n = 319 responses, 49 participants)

Spearman / CCC with 95% CIs from resampling participants. Full tables: `outputs/text/reports/summary_val.md`
and `outputs/text/reports/analysis_val.md` (paired tests, recalibrated CCC, within-question and beyond-length).

| Model | answer_score Spearman | CCC | R2 | overall_perf. Spearman | confidence Spearman | speaking Spearman |
|---|---|---|---|---|---|---|
| mean | – | 0.00 | 0.00 | – | – | – |
| question_mean | -0.16 [-0.28, -0.04] | -0.066 | -0.088 | -0.17 | -0.18 | -0.21 |
| length_only | **0.45** [0.32, 0.55] | 0.255 | **0.153** | **0.49** | **0.35** | 0.45 |
| duration_only (diagnostic) | 0.40 [0.27, 0.52] | 0.227 | 0.135 | 0.44 | 0.31 | 0.40 |
| linguistic | 0.42 [0.27, 0.53] | 0.241 | 0.122 | 0.47 | 0.31 | 0.45 |
| embedding | 0.27 [0.15, 0.38] | 0.167 | -0.054 | 0.23 | 0.18 | 0.25 |
| embedding_ling | 0.41 [0.26, 0.53] | 0.257 | 0.124 | 0.42 | 0.27 | 0.43 |
| finetune DeBERTa (v1) | 0.35 [0.20, 0.47] | 0.258 | 0.060 | 0.38 | 0.25 | 0.40 |
| finetune DeBERTa v2 | 0.43 [0.29, 0.54] | **0.328** | 0.109 | 0.45 | 0.31 | **0.47** |
| linguistic + DeBERTa v2 average | 0.43 | 0.294 | – | 0.47 | 0.32 | 0.47 |

Paired participant bootstrap against `length_only`, `answer_score` (95% CI):

| Model | Δ Spearman | Δ CCC | Spearman beyond length (partial) |
|---|---|---|---|
| linguistic | -0.03 [-0.09, +0.02] | -0.01 [-0.05, +0.02] | 0.01 [-0.14, +0.16] |
| embedding_ling | -0.04 [-0.11, +0.03] | 0.00 [-0.05, +0.06] | 0.07 [-0.06, +0.21] |
| DeBERTa v1 | -0.10 [-0.18, -0.02] | 0.00 [-0.06, +0.07] | 0.04 [-0.08, +0.17] |
| DeBERTa v2 | -0.02 [-0.09, +0.04] | **+0.07 [+0.02, +0.13]** | 0.09 [-0.04, +0.23] |

### Findings

1. **Answer length dominates.** Within 96% of the 76 questions, longer answers receive *lower* scores on
   every target (median within-question Spearman about -0.48 for `answer_score`). Within a single participant
   the effect is weaker (about -0.16), so it is mostly a between-person pattern in the annotations.
2. **"Length" is partly speaking rate, not only duration** (`reports/length_decomposition.md`, train). Word
   count and clip seconds correlate 0.95. Holding clip seconds fixed, more words still means a lower score
   (partial -0.23 for `answer_score`). Holding word count fixed, a longer clip is slightly *positive* (+0.08).
   Word count also beats clip duration on validation (paired Δ Spearman -0.05 [-0.09, -0.01]). Feedback should
   not reduce this to "be shorter": the data equally supports "do not rush". These are associations with
   annotator ratings, not evidence that shorter answers are better.
3. **No text model ranks responses better than word count.** Every paired Δ Spearman interval contains 0 or is
   negative. Only DeBERTa v2 is significantly better on CCC (+0.07 on `answer_score`, positive on all four
   targets): it ranks like word count but has better-calibrated predictions.
4. **Signal beyond length appears only for `speaking_skills`.** After adjusting for word count, DeBERTa v2
   (partial Spearman 0.18 [0.05, 0.31]), the ensemble (0.18) and `embedding_ling` (0.12) keep some signal. For
   `answer_score`, `confidence_score` and `overall_performance` no model's interval excludes 0.
5. **DeBERTa v2 vs v1:** Spearman 0.35 → 0.43 and CCC 0.258 → 0.328 on `answer_score`. The four changes were run
   together, so the gain is not attributed to any single one. Linear recalibration lifts Ridge CCC slightly
   (`length_only` 0.255 → 0.286) but not v2's (0.328 → 0.300), which the CCC loss already calibrates.
6. **Question identity does not help.** `question_mean` is negative on validation. Per-question means from 231
   training participants do not transfer to new participants, so models are not scoring well by recognising
   the question.
7. Frozen sentence embeddings alone do worse than predicting the mean on R2.
8. The labels are strongly inter-correlated (Spearman 0.73–0.85 on train), so text predicts a general
   impression more than four separate skills. The report should not present them as independent findings.

Caveat: the v2 recipe was fixed before looking at validation, but validation has now been viewed many times.
Treat the test split as the only unbiased estimate.

## Test results (run once, 21 Oct 2026; n = 298 responses, 51 participants)

DeBERTa v2 was chosen as the final model on validation before the test split was opened. Word count and
linguistic Ridge are reported only as reference points. Full tables: `reports/summary_test.md`, `reports/analysis_test.md`.

| Target | v2 Spearman [95% CI] | length_only | linguistic | v2 CCC | length_only CCC | v2 partial \| len |
|---|---|---|---|---|---|---|
| answer_score | 0.461 [0.33, 0.58] | 0.447 | 0.450 | 0.437 | 0.391 | 0.17 [0.05, 0.29] |
| confidence_score | 0.442 [0.29, 0.57] | 0.448 | 0.436 | 0.424 | 0.360 | 0.14 [0.01, 0.27] |
| speaking_skills | 0.484 [0.35, 0.59] | 0.483 | 0.479 | 0.359 | 0.289 | 0.15 [0.03, 0.28] |
| overall_performance | 0.480 [0.34, 0.60] | 0.467 | 0.480 | 0.427 | 0.390 | 0.18 [0.05, 0.31] |

- **Ranking:** v2 is indistinguishable from word count (all paired Δ Spearman intervals contain 0).
- **Calibration:** v2's CCC is higher on all four targets, but the paired interval excludes 0 only for
  `speaking_skills` (+0.07 [+0.01, +0.13]).
- **Signal beyond length:** unlike on validation, v2 shows signal beyond length on all four targets.
- **Ordering:** v2 beats v1 on every target on both validation and test.

## Delivery features and SVM (experiment)

This experiment tests whether verbatim fillers and acoustic speaking rate add anything, and
whether an RBF support-vector regressor (SVR) beats Ridge.

```bash
python text_branch/delivery_features.py --step wav          # ffmpeg -> outputs/text/wav (1.8 GB)
python text_branch/delivery_features.py --step syllables    # acoustic syllable nuclei, CPU (~10 min)
python text_branch/transcribe_verbatim.py --model nyralabs/CrisperWhisper2.0_turbo   # ~5 h on RTX 4050, resumable
python text_branch/delivery_features.py --step fillers
python text_branch/delivery_features.py --step merge
python text_branch/train_feature_models.py                  # Ridge vs SVR on linguistic / delivery / both
```

Extra dependencies: `pip install "crisperwhisper[transformers]" praat-parselmouth` (ffmpeg on PATH).

- **Acoustic syllable rate** follows de Jong & Wempe (2009) via Parselmouth: voiced intensity peaks with a
  ≥2 dB dip, threshold 25 dB below the 99th-percentile intensity, and pauses of at least 0.3 s. Features:
  speech rate (syllables/s of the whole clip), articulation rate (syllables/s of sounding speech), phonation
  ratio, and mean syllable duration. Check: acoustic syllable counts correlate 0.961 with transcript
  syllable counts across 1997 clips (median ratio 1.05).
- **Verbatim fillers** come from CrisperWhisper 2.0 turbo (nyralabs; non-commercial research licence;
  local inference). Turbo matched the large model's filler counts on a 10-clip check (13 vs 13) at about a
  third of the runtime. Features: fillers per 100 words and per minute, immediate repetitions, and cut-off
  words (`th-`). Non-speech tags (`[breath]`, `[laughter]`, ...) are ignored. Filler timestamps are saved in
  `outputs/text/delivery/filler_evidence.json` for report evidence.
- **Filler counts:** 4790 um/uh across 2011 clips, against 14 in the original Whisper-large-v3 transcripts.
  63% of answers contain at least one filler; the median is 2.0 per 100 words.

Validation Spearman:

| Model | answer | confidence | speaking | overall |
|---|---|---|---|---|
| linguistic, Ridge | 0.42 | 0.31 | 0.45 | 0.47 |
| linguistic, SVR | 0.40 | 0.31 | 0.44 | 0.48 |
| delivery only, Ridge | 0.16 | 0.15 | 0.19 | 0.16 |
| delivery only, SVR | 0.23 | 0.16 | 0.20 | 0.26 |
| linguistic + delivery, Ridge | 0.41 | 0.33 | 0.45 | 0.48 |
| linguistic + delivery, SVR | 0.40 | 0.32 | 0.39 | 0.48 |

Findings:
- **Filler rate does not track the labels.** Its training-set Spearman with every target is between
  -0.01 and +0.02, even for `speaking_skills`, whose definition mentions fillers. Cut-off words are only
  slightly negative (-0.04 to -0.08).
- **Faster speakers were rated somewhat lower** (speech rate vs targets: about -0.20).
- Delivery features alone carry a weak signal, and SVR uses it better than Ridge. Added to the linguistic
  features they make no meaningful difference (all changes are well inside the 95% CIs).
- SVR does not beat Ridge on linguistic features.
- Fillers are still useful as *evidence* in feedback ("3 fillers in the first 10 s"), but on this dataset
  they should not be presented as a driver of the rated scores.

## Outputs (`outputs/text/`, not committed)

| Path | Contents |
|---|---|
| `features/responses.parquet` | `id`, `user_no`, `question_id`, split, question, cleaned answer, targets, linguistic features |
| `features/emb_all-mpnet-base-v2_{answer,question}.npy` | embeddings, row-aligned with `responses.parquet` |
| `predictions/<model>.parquet` | `pred_<target>` per response. Ridge train rows are out-of-fold (safe for fusion); DeBERTa train rows are empty unless run with `--oof` (`finetune_deberta-v3-base_v2` has them, all seeds averaged) |
| `reports/` | per-model CSVs, `summary_val.{csv,md}`, configs, fine-tuning log |
| `models/finetune_deberta-v3-base/seed*/` | fine-tuned weights + label scaling |
| `wav/` | 16 kHz mono audio per response (1.8 GB) |
| `verbatim/<id>.json` | CrisperWhisper turbo verbatim transcript with word timestamps |
| `delivery/` | `syllables.parquet`, `fillers.parquet`, `delivery_features.parquet`, `filler_evidence.json` |

Join key for fusion: `id`. The audio branch names the participant column `user_id`; this branch keeps RecruitView's original `user_no`.

## Limitations

- About 1.4k training responses from university students. Results may not generalise to other populations or interview settings.
- Transcripts are ASR output. Real interviews will need the same ASR (Whisper-large-v3) to match training.
- 13 responses have empty transcripts. They are kept, with an `empty_transcript` flag.
- Lexicon features count wording. They are evidence for feedback, not measures of confidence or ability.
