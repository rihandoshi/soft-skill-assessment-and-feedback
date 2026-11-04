"""Train/evaluate a GRU regressor over unpooled WavLM frame embeddings.

Uses the fixed RecruitView train/val/test participant splits and the same
GroupKFold(n_splits=config['modeling']['cv_folds']) folds as compare_audio_models.
Frame embeddings are cached separately from the pooled Ridge features.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_audio_models import (  # noqa: E402
    _audio, atomic_json, atomic_npz, load_config, load_split_data, metrics, project_path,
)

TARGETS = ("confidence_score", "speaking_skills", "overall_performance")


def _progress(iterable, **kwargs):
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


def _frame_cache(config, data, allow_download=False, limit=None, frame_layer=None):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    settings = config["extractors"]["wavlm"]
    model_name = settings["model_name"]
    cache = project_path(config["paths"]["cache_dir"]) / "huggingface"
    common = {"cache_dir": str(cache), "local_files_only": not allow_download,
              "trust_remote_code": bool(settings.get("trust_remote_code", False))}
    processor = AutoFeatureExtractor.from_pretrained(model_name, **common)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name, **common).eval().to(device)
    model.requires_grad_(False)
    if device.type == "cuda":
        print(f"[gpu] WavLM frame extraction on {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("[device] WavLM frame extraction on CPU (CUDA unavailable)", flush=True)
    chunk_samples = int(float(config["audio"]["chunk_seconds"]) * config["audio"]["sample_rate"])
    overlap = int(float(config["audio"]["chunk_overlap_seconds"]) * config["audio"]["sample_rate"])
    # Minimum waveform length accepted by the stack of WavLM convolution layers.
    receptive = 1
    jump = 1
    for kernel, stride in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (int(kernel) - 1) * jump
        jump *= int(stride)
    min_chunk_samples = receptive
    output = {}
    layer_name = f"layer_{frame_layer}" if frame_layer is not None else "last_hidden_state"
    cache_root = project_path(config["paths"]["cache_dir"]) / "frame_embeddings" / "wavlm" / model_name.replace("/", "_") / layer_name
    for split, frame in data.items():
        sequences = []
        for row in _progress(frame.itertuples(index=False), desc=f"WavLM {split}", total=len(frame)):
            cache_path = cache_root / split / f"{row.id}.npz"
            seq = None
            if cache_path.is_file():
                try:
                    with np.load(cache_path, allow_pickle=False) as archive:
                        seq = archive["frames"].astype(np.float32)
                        cached_model = str(archive["model_name"].item()) if "model_name" in archive else ""
                    if seq.ndim != 2 or seq.shape[1] != 768 or not len(seq) or not np.isfinite(seq).all():
                        raise ValueError(f"Invalid frame sequence shape or values: {seq.shape}")
                    if cached_model and cached_model != model_name:
                        raise ValueError(f"Cache checkpoint mismatch: {cached_model}")
                except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
                    print(f"[cache] Rebuilding invalid WavLM frame cache {cache_path.name}: {exc}", flush=True)
                    cache_path.unlink(missing_ok=True)
                    seq = None
            if seq is None:
                wav_path = project_path(row.audio_path)
                signal, rate, _ = _audio(wav_path)
                step = max(1, chunk_samples - overlap)
                if len(signal) < min_chunk_samples:
                    signal = np.pad(signal, (0, min_chunk_samples - len(signal)))
                starts = [0]
                while starts[-1] + chunk_samples < len(signal):
                    next_start = starts[-1] + step
                    remaining = len(signal) - next_start
                    # Shift the final window backward so the tail is never a tiny, invalid chunk.
                    if remaining < min_chunk_samples:
                        next_start = max(0, len(signal) - chunk_samples)
                    if next_start <= starts[-1]:
                        break
                    starts.append(next_start)
                pieces = []
                with torch.no_grad():
                    for start in starts:
                        chunk = signal[start:start + chunk_samples]
                        if not len(chunk):
                            continue
                        if len(chunk) < min_chunk_samples:
                            raise ValueError(f"Audio chunk for response {row.id} is too short for WavLM: {len(chunk)} samples")
                        encoded = processor(chunk, sampling_rate=rate, return_tensors="pt")
                        values = encoded["input_values"].to(device)
                        output = model(input_values=values, output_hidden_states=frame_layer is not None)
                        frame_output = output.last_hidden_state if frame_layer is None else output.hidden_states[int(frame_layer)]
                        hidden = frame_output[0].float().cpu().numpy()
                        # Remove half of overlap on each side to avoid duplicated frames.
                        if start and overlap:
                            trim = int(hidden.shape[0] * overlap / max(len(chunk), 1) / 2)
                            hidden = hidden[min(trim, len(hidden)-1):]
                        if start + chunk_samples < len(signal) and overlap:
                            trim = int(hidden.shape[0] * overlap / max(len(chunk), 1) / 2)
                            hidden = hidden[:max(1, len(hidden)-trim)]
                        pieces.append(hidden)
                seq = np.concatenate(pieces, axis=0).astype(np.float32)
                if seq.ndim != 2 or seq.shape[1] != 768:
                    raise ValueError(f"Expected WavLM frame sequence [T,768], got {seq.shape} for {row.id}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_npz(cache_path, frames=seq, model_name=np.asarray(model_name))
            sequences.append(seq)
        output[split] = sequences
    return output


def _make_model(hidden_size, target_count, pooling="last"):
    import torch
    import torch.nn as nn
    class TemporalRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(768, hidden_size, num_layers=1, batch_first=True)
            self.pooling = pooling
            self.head = nn.Linear(hidden_size, target_count)
        def forward(self, padded, lengths):
            from torch.nn.utils.rnn import pack_padded_sequence
            packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed)
            if self.pooling == "last":
                representation = hidden[-1]
            elif self.pooling == "mean":
                from torch.nn.utils.rnn import pad_packed_sequence
                outputs, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=padded.size(1))
                mask = torch.arange(outputs.size(1), device=outputs.device)[None, :] < lengths.to(outputs.device)[:, None]
                representation = (outputs * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(outputs.device).unsqueeze(-1)
            else:
                raise ValueError(f"Unsupported GRU pooling mode: {self.pooling}")
            return self.head(representation)
    return TemporalRegressor()


def _fit_predict(sequences, targets, train_indices, predict_indices, hidden_size, epochs, batch_size, lr, seed, device,
                 validation_indices=None, history_path=None):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    class Samples(Dataset):
        def __init__(self, indices): self.indices = list(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, i):
            index = self.indices[i]
            return torch.from_numpy(sequences[index]), torch.tensor(targets[index], dtype=torch.float32)
    def collate(items):
        seqs, ys = zip(*items)
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        return pad_sequence(seqs, batch_first=True), lengths, torch.stack(ys)
    loader = DataLoader(Samples(train_indices), batch_size=batch_size, shuffle=True, collate_fn=collate)
    model = _make_model(hidden_size, targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    try:
        from tqdm.auto import tqdm
        epoch_progress = tqdm(range(epochs), desc="Training epochs", leave=False)
    except ImportError:
        epoch_progress = range(epochs)
    history = []
    for epoch_number, _ in enumerate(epoch_progress, start=1):
        running_loss = 0.0
        for padded, lengths, y in loader:
            padded, y = padded.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(padded, lengths), y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
        train_mse = running_loss / max(1, len(loader))
        val_mse = float("nan")
        if validation_indices is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(validation_indices), batch_size):
                    batch = validation_indices[start:start + batch_size]
                    seqs = [torch.from_numpy(sequences[i]) for i in batch]
                    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
                    padded = pad_sequence(seqs, batch_first=True).to(device)
                    y = torch.as_tensor(targets[batch], dtype=torch.float32, device=device)
                    val_losses.append(float(loss_fn(model(padded, lengths), y).cpu()))
            val_mse = float(np.mean(val_losses))
            model.train()
        history.append({"epoch": epoch_number, "train_mse": train_mse, "validation_mse": val_mse})
        if hasattr(epoch_progress, "set_postfix"):
            epoch_progress.set_postfix(train_mse=f"{train_mse:.4f}", val_mse=f"{val_mse:.4f}" if np.isfinite(val_mse) else "n/a")
        if history_path is not None:
            pd.DataFrame(history).to_csv(history_path, index=False)
    model.eval()
    predictions = []
    with torch.no_grad():
        starts = range(0, len(predict_indices), batch_size)
        for start in _progress(starts, desc="Predicting", leave=False):
            batch = predict_indices[start:start + batch_size]
            seqs = [torch.from_numpy(sequences[i]) for i in batch]
            lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
            padded = pad_sequence(seqs, batch_first=True).to(device)
            predictions.append(model(padded, lengths).cpu().numpy())
    return np.concatenate(predictions, axis=0), model, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hidden-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, help="Smoke run: first N clips per split; grouped folds reduced per config")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from sklearn.model_selection import GroupKFold

    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    data = load_split_data(config, limit=args.limit)
    frame_sequences = _frame_cache(config, data, args.allow_download, args.limit)
    folds = int(config["modeling"].get("smoke_cv_folds", 3) if args.limit else config["modeling"].get("cv_folds", 5))
    train = data["train"].reset_index(drop=True)
    sequences = frame_sequences["train"]
    targets = train[list(TARGETS)].to_numpy(dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("Target scores contain missing or non-finite values")
    groups = train.user_id.astype(str).to_numpy()
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} training participants for GroupKFold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[gpu] GRU training on {torch.cuda.get_device_name(device)}", flush=True)
        print(f"[gpu] CUDA version {torch.version.cuda}; GPU memory {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GiB", flush=True)
    else:
        print("[device] GRU training on CPU (PyTorch CUDA unavailable)", flush=True)
    oof = np.full_like(targets, np.nan, dtype=np.float32)
    report_dir = project_path(config["paths"]["reports_dir"]) / ("smoke" if args.limit else "")
    report_dir.mkdir(parents=True, exist_ok=True)
    history_dir = report_dir / "wavlm_gru_diagnostics" / "fold_histories"
    history_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=folds)
    fold_records = []
    split_iterator = splitter.split(np.zeros(len(train)), targets, groups)
    for fold, (tr, va) in enumerate(_progress(split_iterator, desc="GroupKFold", total=folds), start=1):
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Participant overlap between GroupKFold train and validation")
        pred, _, _ = _fit_predict(sequences, targets, tr, va, args.hidden_size, args.epochs,
                                  args.batch_size, args.learning_rate, seed + fold, device,
                                  validation_indices=va, history_path=history_dir / f"fold_{fold}.csv")
        oof[va] = pred
        for j, target in enumerate(TARGETS):
            fold_records.append({"fold": fold, "target": target, **metrics(targets[va, j], pred[:, j])})
    rows = []
    for j, target in enumerate(TARGETS):
        rows.append({"split": "train_grouped_oof", "target": target, **metrics(targets[:, j], oof[:, j])})
    val = data["val"].reset_index(drop=True)
    val_y = val[list(TARGETS)].to_numpy(dtype=np.float32)
    val_pred, final_model, _ = _fit_predict(sequences, targets, np.arange(len(train)), np.arange(len(val)), args.hidden_size, args.epochs, args.batch_size, args.learning_rate, seed, device)
    for j, target in enumerate(TARGETS):
        rows.append({"split": "val", "target": target, **metrics(val_y[:, j], val_pred[:, j])})
    if not args.limit:
        test = data["test"].reset_index(drop=True)
        test_y = test[list(TARGETS)].to_numpy(dtype=np.float32)
        test_pred, _, _ = _fit_predict(sequences, targets, np.arange(len(train)), np.arange(len(test)), args.hidden_size, args.epochs, args.batch_size, args.learning_rate, seed, device)
        for j, target in enumerate(TARGETS):
            rows.append({"split": "test", "target": target, **metrics(test_y[:, j], test_pred[:, j])})
        _save_predictions(report_dir, "test", test, test_y, test_pred)
    _save_predictions(report_dir, "train_grouped_oof", train, targets, oof)
    _save_predictions(report_dir, "val", val, val_y, val_pred)
    _plot_diagnostics(history_dir, report_dir / "wavlm_gru_diagnostics", TARGETS,
                      train, targets, oof, val, val_y, val_pred,
                      test if not args.limit else None, test_y if not args.limit else None,
                      test_pred if not args.limit else None)
    pd.DataFrame(rows).to_csv(report_dir / "wavlm_gru_metrics.csv", index=False)
    pd.DataFrame(fold_records).to_csv(report_dir / "wavlm_gru_fold_metrics.csv", index=False)
    model_dir = project_path(config["paths"]["models_dir"]) / ("smoke" if args.limit else "")
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": final_model.cpu().state_dict(), "hidden_size": args.hidden_size, "input_size": 768, "targets": list(TARGETS), "model_name": config["extractors"]["wavlm"]["model_name"]}, model_dir / "wavlm_gru.pt")
    atomic_json(report_dir / "wavlm_gru_summary.json", {"seed": seed, "cv_strategy": "GroupKFold by user_id", "cv_folds": folds, "hidden_size": args.hidden_size, "epochs": args.epochs, "frozen_encoder": True, "input": "last_hidden_state frame embeddings [T,768]", "padding": "pad_sequence + pack_padded_sequence", "metrics": rows})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


def _save_predictions(report_dir, split_name, frame, actual, predicted):
    result = pd.DataFrame({"id": frame.id.astype(str), "user_id": frame.user_id.astype(str)})
    for index, target in enumerate(TARGETS):
        result[f"{target}_actual"] = actual[:, index]
        result[f"{target}_prediction"] = predicted[:, index]
    result.to_csv(report_dir / f"wavlm_gru_{split_name}_predictions.csv", index=False)


def _plot_diagnostics(history_dir, output_dir, targets, train, train_y, train_pred,
                      val, val_y, val_pred, test=None, test_y=None, test_pred=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _plot_diagnostics_svg(history_dir, output_dir, targets, train, train_y, train_pred,
                                     val, val_y, val_pred, test, test_y, test_pred)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = []
    for path in sorted(history_dir.glob("fold_*.csv")):
        history = pd.read_csv(path)
        history["fold"] = path.stem
        histories.append(history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.epoch, history.train_mse, label="Training MSE")
        ax.plot(history.epoch, history.validation_mse, label="Fold validation MSE")
        ax.set(xlabel="Epoch", ylabel="MSE (original target scale)", title=f"GRU convergence: {path.stem}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{path.stem}_loss.png", dpi=160)
        plt.close(fig)
    if histories:
        combined = pd.concat(histories, ignore_index=True)
        combined.to_csv(output_dir / "fold_loss_history.csv", index=False)
        fig, ax = plt.subplots(figsize=(9, 5))
        for fold, group in combined.groupby("fold"):
            ax.plot(group.epoch, group.train_mse, alpha=0.65, label=f"{fold} train")
            ax.plot(group.epoch, group.validation_mse, linestyle="--", alpha=0.8, label=f"{fold} validation")
        ax.set(xlabel="Epoch", ylabel="MSE (original target scale)", title="GRU loss by participant-grouped fold")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "all_folds_loss.png", dpi=160)
        plt.close(fig)
    split_data = [("train_grouped_oof", train, train_y, train_pred), ("val", val, val_y, val_pred)]
    if test is not None:
        split_data.append(("test", test, test_y, test_pred))
    distribution_rows = []
    for split_name, frame, actual, predicted in split_data:
        for index, target in enumerate(targets):
            a, p = actual[:, index], predicted[:, index]
            distribution_rows.append({"split": split_name, "target": target,
                                      "actual_mean": float(a.mean()), "actual_std": float(a.std()),
                                      "prediction_mean": float(p.mean()), "prediction_std": float(p.std()),
                                      "actual_min": float(a.min()), "actual_max": float(a.max()),
                                      "prediction_min": float(p.min()), "prediction_max": float(p.max()),
                                      "mean_bias_prediction_minus_actual": float(p.mean() - a.mean()),
                                      "prediction_to_actual_std_ratio": float(p.std() / a.std()) if a.std() else float("nan")})
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(a, bins=20, alpha=0.55, label="Actual", density=True)
            axes[0].hist(p, bins=20, alpha=0.55, label="Prediction", density=True)
            axes[0].set(title="Score distributions", xlabel="Score")
            axes[0].legend()
            axes[1].scatter(a, p, alpha=0.35, s=12)
            lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
            axes[1].plot([lo, hi], [lo, hi], "--", color="gray")
            axes[1].set(title="Predicted vs actual", xlabel="Actual", ylabel="Prediction")
            fig.suptitle(f"{target}: {split_name}")
            fig.tight_layout()
            fig.savefig(output_dir / f"{split_name}_{target}_predictions.png", dpi=160)
            plt.close(fig)
    pd.DataFrame(distribution_rows).to_csv(output_dir / "prediction_distribution_summary.csv", index=False)


def _plot_diagnostics_svg(history_dir, output_dir, targets, train, train_y, train_pred,
                          val, val_y, val_pred, test=None, test_y=None, test_pred=None):
    """Dependency-free SVG diagnostic plots when matplotlib is unavailable."""
    import html
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#2563eb", "validation": "#dc2626", "actual": "#2563eb", "prediction": "#f97316"}

    def svg_start(title, width=900, height=520):
        return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="18">{html.escape(title)}</text>']

    def write_svg(path, pieces):
        pieces.append('</svg>')
        path.write_text('\n'.join(pieces), encoding='utf-8')

    histories = []
    for history_path in sorted(history_dir.glob("fold_*.csv")):
        history = pd.read_csv(history_path)
        histories.append((history_path.stem, history))
        parts = svg_start(f"GRU convergence: {history_path.stem}")
        x0, y0, width, height = 75, 65, 780, 390
        max_y = max(2.5, float(np.nanmax(history[["train_mse", "validation_mse"]].to_numpy())) * 1.05)
        parts += [f'<path d="M{x0} {y0} V{y0+height} H{x0+width}" fill="none" stroke="#333"/>',
                  f'<text x="{x0+width/2}" y="490" text-anchor="middle" font-family="Arial">Epoch</text>',
                  f'<text x="22" y="{y0+height/2}" transform="rotate(-90 22 {y0+height/2})" text-anchor="middle" font-family="Arial">MSE (original target units)</text>']
        for name, color, column in (("Training", colors["train"], "train_mse"), ("Fold validation", colors["validation"], "validation_mse")):
            points = []
            for _, row in history.iterrows():
                x = x0 + (float(row.epoch)-1) / max(1, len(history)-1) * width
                y = y0 + height - float(row[column]) / max_y * height
                points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}"/>')
        parts += [f'<text x="650" y="60" fill="{colors["train"]}" font-family="Arial">— Training MSE</text>',
                  f'<text x="770" y="60" fill="{colors["validation"]}" font-family="Arial">— Validation MSE</text>']
        write_svg(output_dir / f"{history_path.stem}_loss.svg", parts)
    if histories:
        combined_history = pd.concat([history.assign(fold=fold) for fold, history in histories], ignore_index=True)
        combined_history.to_csv(output_dir / "fold_loss_history.csv", index=False)
        parts = svg_start("GRU training and fold-validation loss by epoch")
        x0, y0, width, height = 75, 65, 780, 390
        max_y = max(2.5, max(float(np.nanmax(h[["train_mse", "validation_mse"]].to_numpy())) for _, h in histories) * 1.05)
        parts += [f'<path d="M{x0} {y0} V{y0+height} H{x0+width}" fill="none" stroke="#333"/>',
                  f'<text x="{x0+width/2}" y="490" text-anchor="middle" font-family="Arial">Epoch</text>',
                  f'<text x="22" y="{y0+height/2}" transform="rotate(-90 22 {y0+height/2})" text-anchor="middle" font-family="Arial">MSE (original target units)</text>']
        fold_colors = ["#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706"]
        for fi, (fold, history) in enumerate(histories):
            for key, color, dash in (("train_mse", fold_colors[fi % len(fold_colors)], ""), ("validation_mse", fold_colors[fi % len(fold_colors)], ' stroke-dasharray="7 4"')):
                points = []
                for _, row in history.iterrows():
                    x = x0 + (float(row.epoch)-1) / max(1, len(history)-1) * width
                    y = y0 + height - float(row[key]) / max_y * height
                    points.append(f"{x:.1f},{y:.1f}")
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{" ".join(points)}"/>')
            parts.append(f'<text x="{85+fi*145}" y="500" fill="{fold_colors[fi % len(fold_colors)]}" font-family="Arial">Fold {fi+1}: solid train / dashed val</text>')
        write_svg(output_dir / "all_folds_loss.svg", parts)

    split_data = [("train_grouped_oof", train, train_y, train_pred), ("val", val, val_y, val_pred)]
    if test is not None:
        split_data.append(("test", test, test_y, test_pred))
    distributions = []
    for split_name, _, actual, predicted in split_data:
        for j, target in enumerate(targets):
            a, p = np.asarray(actual[:, j]), np.asarray(predicted[:, j])
            distributions.append({"split": split_name, "target": target, "actual_mean": a.mean(), "actual_std": a.std(),
                                 "prediction_mean": p.mean(), "prediction_std": p.std(), "actual_min": a.min(), "actual_max": a.max(),
                                 "prediction_min": p.min(), "prediction_max": p.max(),
                                 "mean_bias_prediction_minus_actual": p.mean()-a.mean(),
                                 "prediction_to_actual_std_ratio": p.std()/a.std() if a.std() else np.nan})
            parts = svg_start(f"{target}: {split_name}", width=1000, height=480)
            bins = np.linspace(min(a.min(), p.min()), max(a.max(), p.max()), 21)
            hist_a, _ = np.histogram(a, bins=bins, density=True)
            hist_p, _ = np.histogram(p, bins=bins, density=True)
            x0, y0, w, h = 55, 65, 420, 330
            max_density = max(float(hist_a.max()), float(hist_p.max()), 1e-6)
            bw = w / len(hist_a) / 2
            parts.append(f'<path d="M{x0} {y0} V{y0+h} H{x0+w}" fill="none" stroke="#333"/>')
            for i, (ha, hp) in enumerate(zip(hist_a, hist_p)):
                x = x0 + i * (w / len(hist_a))
                parts.append(f'<rect x="{x:.1f}" y="{y0+h-ha/max_density*h:.1f}" width="{bw:.1f}" height="{ha/max_density*h:.1f}" fill="{colors["actual"]}" opacity=".65"/>')
                parts.append(f'<rect x="{x+bw:.1f}" y="{y0+h-hp/max_density*h:.1f}" width="{bw:.1f}" height="{hp/max_density*h:.1f}" fill="{colors["prediction"]}" opacity=".65"/>')
            parts += [f'<text x="{x0+80}" y="425" fill="{colors["actual"]}" font-family="Arial">Actual distribution</text>',
                      f'<text x="{x0+230}" y="425" fill="{colors["prediction"]}" font-family="Arial">Prediction distribution</text>']
            sx, sy, sw, sh = 540, 65, 410, 330
            lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
            span = max(hi-lo, 1e-6)
            parts.append(f'<path d="M{sx} {sy} V{sy+sh} H{sx+sw}" fill="none" stroke="#333"/>')
            parts.append(f'<line x1="{sx}" y1="{sy+sh}" x2="{sx+sw}" y2="{sy}" stroke="#777" stroke-dasharray="6 4"/>')
            for av, pv in zip(a, p):
                cx = sx + (float(av)-lo)/span*sw
                cy = sy + sh - (float(pv)-lo)/span*sh
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="2.2" fill="#2563eb" opacity=".45"/>')
            parts += [f'<text x="{sx+sw/2}" y="435" text-anchor="middle" font-family="Arial">Actual score</text>',
                      f'<text x="510" y="{sy+sh/2}" transform="rotate(-90 510 {sy+sh/2})" text-anchor="middle" font-family="Arial">Predicted score</text>']
            write_svg(output_dir / f"{split_name}_{target}_predictions.svg", parts)
    pd.DataFrame(distributions).to_csv(output_dir / "prediction_distribution_summary.csv", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
