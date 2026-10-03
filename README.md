# RecruitView audio branch

This is the audio branch of the multimodal interview assessment project. It extracts handcrafted, eGeMAPS, Whisper timing, and WavLM/HuBERT features, compares feature sets with participant-grouped cross-validation, and exports the selected representation for multimodal fusion.

## Run

From the project root:

```powershell
python -m pip install -r requirements.txt
# Optional extractor/reporting dependencies; install the branches you need.
python -m pip install -r requirements-audio.txt
# Install PyTorch separately for your machine's CPU/CUDA setup.

# Extract features; model downloads require explicit permission.
python scripts/extract_audio_features.py --allow-model-download

# Train/compare models and export fusion features.
python scripts/compare_audio_models.py --n-jobs 4 --trials 12

# Smoke run on a few clips in each split; never reports test metrics.
python scripts/extract_audio_features.py --extractors legacy --limit 5
python scripts/compare_audio_models.py --limit 5 --trials 1
```

Optional branches need `opensmile` for eGeMAPS, `faster-whisper` for ASR timing, and PyTorch plus Transformers for WavLM/HuBERT. Model weights are cached; pass `--allow-model-download` only when a checkpoint must be fetched. See `config.yaml` to change paths, checkpoint names, feature options, or model grids.

Model comparison reads the existing embedding cache and does not rerun WavLM/HuBERT extraction. Transformer extraction uses CUDA when available. Scikit-learn Ridge (`lsqr`) trains on CPU; XGBoost uses CUDA when installed with a CUDA-capable build.

## SSL transformer layer diagnostics

Run the WavLM/HuBERT hidden-layer and pooling comparison from the project root:

```powershell
python scripts/run_ssl_layer_sweep.py
```

By default it evaluates every cached hidden-state output (index 0 is the feature-encoder output; 1–12 are Transformer blocks), with mean, population-standard-deviation, and concatenated mean+std pooling. To run a configured subset, pass indices such as `--layers 0,4,8,12`. It uses the fixed participant-disjoint validation split, selects Ridge alpha by five-fold `GroupKFold(user_id)` on training data, and computes participant-cluster bootstrap intervals for summary comparisons. The weighted layer blend is learned from training-only grouped out-of-fold predictions. It does not score the test split.

Results are written to `results/ssl_layer_sweep/` (experiment/comparison CSVs, selected layers, learned weights, configuration, report and SVG plots). Existing WavLM/HuBERT embedding caches and saved mean+std layer-sweep curves are reused when compatible. The run that produced the current results evaluated all 13 hidden-state outputs for both encoders, all three pooling methods and all three targets; it reused the cached Transformer embeddings and did not perform new model inference. Ridge fitting runs on CPU through scikit-learn; CUDA is used for extraction only when embeddings need to be computed.

In this run, train-CV-selected mean+std layers were WavLM layer 7 for all targets, and HuBERT layers 6 (confidence), 9 (speaking skills), and 6 (overall performance). The best individual layer/pooling candidate matched mean+std in five of six encoder/target pairs; HuBERT overall performance selected layer 6 std pooling by CV but did not improve validation metrics. Weighted blends moved validation RMSE by only about 0.001–0.004, and paired WavLM-vs-HuBERT participant-bootstrap intervals crossed zero for every target/method comparison, so the run does not establish a material layer-selection gain or a stable encoder advantage.

## VAD and pause diagnostics

Install the lightweight Silero VAD dependency into the existing environment (it reuses the installed PyTorch), then run:

```powershell
python -m pip install -r requirements-vad.txt
python scripts/run_vad_pause.py
```

The runner caches per-clip speech/non-speech regions, pause statistics and speech-only WavLM/HuBERT hidden-state mean/std embeddings under `outputs/audio/cache/vad_pause/`. It also reuses the existing full-audio embedding caches. It compares full-audio pooling, speech-only pooling, speech-only plus pause statistics, and a speech-only plus duration control. Layer indices come from the prior train-CV-only SSL sweep; Ridge alpha is tuned with `GroupKFold(user_id)` on the fixed training split, then evaluated once on the existing participant-disjoint validation split. Participant-cluster bootstrap intervals and paired method deltas are included. Test participants are not scored. Missing SSL checkpoint downloads remain opt-in with `--allow-model-download`.

Results are saved in `results/vad_pause/` with per-split region and pause feature CSVs, validation predictions, experiment metrics, participant bootstrap intervals, paired ablation intervals, configuration, and RMSE plots. Non-speech gaps are acoustic measurements; the pipeline does not label them as hesitations. Pause counts/durations use internal gaps between Silero speech regions; total non-speech duration and ratio include leading/trailing non-speech.

## Handcrafted feature improvements

Run the enhanced handcrafted comparison after the VAD stage has populated its cached speech regions:

```powershell
python scripts/compare_handcrafted_improvements.py
```

It compares the existing original handcrafted feature set (A), new features only (B), and original plus new features (C) for Ridge, SVR, ExtraTrees, and RandomForest. Model selection uses training-only `GroupKFold(user_id)` and the unchanged participant-disjoint validation split. It saves per-clip predictor-only CSVs, grouped-CV trial records, validation metrics/predictions, participant bootstrap intervals, paired RMSE intervals, configuration, a report and SVG plots under `results/handcrafted_improvements/`. Test features are generated for completeness but are not scored.

Added features are: word-based speech rate from supplied `[MM:SS - MM:SS]` transcript segments when available; articulation-rate and rate excluding internal pauses (no syllable counts); Silero internal pause count/ratio/mean/median/longest duration and counts above 0.5s/1s; F0 mean/std/min/max/range, voiced percentage and semitone-relative pitch/final-contour slopes; speech RMS mean/std, p95-to-p05 dynamic range and energy slope; plus relative pitch, voiced percentage and RMS over the first, middle and final answer thirds. “Pause” is an acoustic time gap, not a hesitation label. Relative pitch is normalized to the median F0 within that answer; participant identity is never an input feature.

No original feature was removed. The existing jitter and shimmer proxy columns remain in A and C. The original table had no HNR column, so none was fabricated. The feature widths are 27 original, 36 new-only, and 63 combined. The constant `sample_rate_hz` metadata field is excluded so A exactly matches the saved handcrafted benchmark. Timed word rates were available for 1,368/1,394 train clips, 318/319 validation clips and 289/298 test clips; missing values are imputed within each training fold.

Ridge validation metrics (all values on the same fixed validation participants):

| Target | Set | MAE | RMSE | R² | Pearson | Spearman |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| confidence_score | A original | 0.6397 | 0.9436 | 0.1579 | 0.4050 | 0.3629 |
| confidence_score | B new only | 0.6340 | 0.9605 | 0.1275 | 0.3602 | 0.3542 |
| confidence_score | C combined | 0.6255 | 0.9453 | 0.1549 | 0.3935 | 0.3511 |
| speaking_skills | A original | 0.7036 | 1.0926 | 0.1240 | 0.3675 | 0.4286 |
| speaking_skills | B new only | 0.6904 | 1.0883 | 0.1308 | 0.3675 | 0.4515 |
| speaking_skills | C combined | 0.6873 | 1.0826 | 0.1400 | 0.3749 | 0.4369 |
| overall_performance | A original | 0.6385 | 1.0185 | 0.1929 | 0.4416 | 0.5020 |
| overall_performance | B new only | 0.6578 | 1.0430 | 0.1535 | 0.3927 | 0.4631 |
| overall_performance | C combined | 0.6363 | 1.0235 | 0.1849 | 0.4313 | 0.4894 |

These point estimates do not establish a consistent gain from adding features: the combined Ridge RMSE participant-bootstrap intervals overlap zero versus A for all targets. Across the four model families, paired combined-versus-original RMSE intervals also overlap zero for 11 of 12 target/model comparisons; SVR speaking-skills RMSE is worse for C. See `results/handcrafted_improvements/report.md` for all 36 comparisons and uncertainty intervals. Limitations include coarse transcript segment times, missing transcript timing on some clips, low-volume/noisy F0 and VAD errors, jitter/shimmer remaining uncalibrated proxies, and evaluation on only one held-out participant cohort.

## Structure

- `scripts/extract_audio_features.py`: WAV conversion and all audio feature extraction.
- `scripts/run_vad_pause.py`: cached Silero VAD regions, pause features and speech-only SSL pooling diagnostics.
- `scripts/compare_handcrafted_improvements.py`: temporal, pitch, energy and timed-word handcrafted feature comparison.
- `scripts/compare_audio_models.py`: grouped-CV comparisons, validation selection, one final test evaluation, and fusion export.
- `config.yaml`: paths and pipeline settings.
- `requirements*.txt`: pinned core and optional dependencies.
- `Datasets/`: original data, fixed split files, and WAV manifests. The split is never regenerated.
- `outputs/audio/`: generated feature caches, reports, models, and exports (ignored by Git).
- `archive/`: preserved split generator and legacy trainer.
- `Utils/`: backward-compatible commands. `Utils/split_dataset.py` refuses to overwrite the fixed split.

The existing legacy model results are preserved under `outputs/audio/legacy_run/` as historical results. The current validation diagnostics and SSL layer sweep were run on the fixed participant-disjoint splits. Their results are diagnostic validation evidence; they are not scores from a new final test evaluation.
