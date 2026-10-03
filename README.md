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

## Structure

- `scripts/extract_audio_features.py`: WAV conversion and all audio feature extraction.
- `scripts/run_vad_pause.py`: cached Silero VAD regions, pause features and speech-only SSL pooling diagnostics.
- `scripts/compare_audio_models.py`: grouped-CV comparisons, validation selection, one final test evaluation, and fusion export.
- `config.yaml`: paths and pipeline settings.
- `requirements*.txt`: pinned core and optional dependencies.
- `Datasets/`: original data, fixed split files, and WAV manifests. The split is never regenerated.
- `outputs/audio/`: generated feature caches, reports, models, and exports (ignored by Git).
- `archive/`: preserved split generator and legacy trainer.
- `Utils/`: backward-compatible commands. `Utils/split_dataset.py` refuses to overwrite the fixed split.

The existing legacy model results are preserved under `outputs/audio/legacy_run/` as historical results. The current validation diagnostics and SSL layer sweep were run on the fixed participant-disjoint splits. Their results are diagnostic validation evidence; they are not scores from a new final test evaluation.
