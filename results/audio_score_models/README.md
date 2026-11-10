# Audio score models (RecruitView)

Earlier experiments predicting the three RecruitView targets from audio. All use the fixed participant-level
split (or participant-grouped CV); the test split is not used for selection. Scripts: `scripts/legacy/`.

| Folder | Experiment | Main result |
|---|---|---|
| `handcrafted_improvements/` | 27 original handcrafted features (pitch, energy, pauses, duration) vs 36 new ones vs all 63; Ridge / SVR / RF / ExtraTrees | Adding the 36 features gave no consistent gain. `features/` is reused by the explainability Stage 0 audit. **Caveat (found in Stage 1): the pitch features come from a home-made autocorrelation tracker with octave errors on low voices.** |
| `ssl_layer_sweep/` | Every WavLM / HuBERT hidden layer × mean / std / mean+std pooling, Ridge | WavLM layer 7 (mean+std) selected for all three targets (`selected_layers.json`, used by later runs) |
| `vad_pause/` | Silero VAD speech-only pooling and pause features vs full-audio pooling | Speech-only pooling did not help: non-speech regions carry information. `features/*_vad_regions.csv` is the VAD cache reused by the explainability pipeline |
| `repeated_cv_v1/` | 5 repeats × 5 participant-grouped folds over train+val, nested hyperparameter search; handcrafted, eGeMAPS, WavLM, HuBERT × Ridge/SVR/ExtraTrees | **Benchmark.** WavLM L7 + Ridge pooled Spearman 0.499 (confidence), 0.493 (speaking), 0.526 (overall); handcrafted Ridge 0.446 / 0.454 / 0.503; eGeMAPS ≈ 0.36–0.41. `cv_splits.json` is reused by explainability Stage 3. The per-clip OOF predictions were removed from git for size (commit `18ae498`); Stage 3 recovers them from commit `45bae4d` into `outputs/explainability/` |

GRU and attention-pooling experiments over WavLM frames are in `outputs/audio/reports/` (GRU: near-zero
correlation before the ablations; attention run interrupted).

**Label ceiling to keep in mind** (found in explainability Stage 1): RecruitView contains 52 groups of
byte-identical recordings rated independently; their scores agree only at Spearman 0.57 (overall), 0.55
(confidence) and 0.34 (speaking). Model differences of a few hundredths are within label noise.
