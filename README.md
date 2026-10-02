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

## Structure

- `scripts/extract_audio_features.py`: WAV conversion and all audio feature extraction.
- `scripts/compare_audio_models.py`: grouped-CV comparisons, validation selection, one final test evaluation, and fusion export.
- `config.yaml`: paths and pipeline settings.
- `requirements*.txt`: pinned core and optional dependencies.
- `Datasets/`: original data, fixed split files, and WAV manifests. The split is never regenerated.
- `outputs/audio/`: generated feature caches, reports, models, and exports (ignored by Git).
- `archive/`: preserved split generator and legacy trainer.
- `Utils/`: backward-compatible commands. `Utils/split_dataset.py` refuses to overwrite the fixed split.

The existing legacy model results are preserved under `outputs/audio/legacy_run/` as historical results. The current environment completed lightweight checks and a smoke run, but lacks OpenSMILE, faster-whisper, PyTorch, and Transformers, so no new full-data comparison has been run yet. Smoke results are not model evidence.
