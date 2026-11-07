"""Nested participant-CV experiments for lightweight WavLM attention pooling.

Runs direct WavLM attention weighted mean/std, GRU64 attention, and GRU64
mean+attention pooling. Existing Ridge and GRU outputs are read as baselines;
their implementations and the test set are untouched.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.legacy.compare_audio_models import atomic_json, load_config, load_split_data, metrics, project_path
from scripts.legacy.run_wavlm_gru_ablation import BATCH_SIZE, EPOCHS, LEARNING_RATE, MIN_DELTA, PATIENCE
from scripts.legacy.train_wavlm_gru import TARGETS, _frame_cache, _progress

ATTENTION_DIM = 32
DROPOUT = 0.2
WEIGHT_DECAY = 0.05
GRAD_CLIP = 1.0
INNER_VALIDATION_FRACTION = 0.15
BOOTSTRAP_SEED = 42
VARIANTS = (
    {"name": "layer7_direct_attention_meanstd", "kind": "direct", "representation": "weighted_mean_std"},
    {"name": "layer7_gru64_attention", "kind": "gru", "representation": "attention"},
    {"name": "layer7_gru64_mean_attention", "kind": "gru", "representation": "mean_attention"},
)


def _seed(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(kind, representation, torch):
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    class AttentionRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.representation = representation
            self.encoder = nn.GRU(768, 64, batch_first=True) if kind == "gru" else None
            dim = 64 if kind == "gru" else 768
            self.attention = nn.Sequential(nn.Linear(dim, ATTENTION_DIM), nn.Tanh(), nn.Dropout(DROPOUT), nn.Linear(ATTENTION_DIM, 1, bias=False))
            head_dim = (768 * 2 if representation == "weighted_mean_std" else
                        (128 if representation == "mean_attention" else 64))
            self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(head_dim, len(TARGETS)))

        def forward(self, padded, lengths, return_attention=False):
            if self.encoder is not None:
                packed = pack_padded_sequence(padded, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
                packed_out, _ = self.encoder(packed)
                values, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=padded.shape[1])
            else:
                values = padded
            mask = torch.arange(values.shape[1], device=values.device)[None, :] < lengths.to(values.device)[:, None]
            scores = self.attention(values).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=1)
            weighted_mean = torch.sum(values * weights.unsqueeze(-1), dim=1)
            if self.representation == "weighted_mean_std":
                centered = values - weighted_mean[:, None, :]
                weighted_std = torch.sqrt(torch.sum(centered.square() * weights.unsqueeze(-1), dim=1).clamp_min(1e-8))
                representation = torch.cat((weighted_mean, weighted_std), dim=-1)
            elif self.representation == "mean_attention":
                masked_values = values * mask.unsqueeze(-1)
                ordinary_mean = masked_values.sum(dim=1) / lengths.to(values.device).unsqueeze(-1)
                representation = torch.cat((ordinary_mean, weighted_mean), dim=-1)
            else:
                representation = weighted_mean
            prediction = self.head(representation)
            if return_attention:
                return prediction, weights
            return prediction

    return AttentionRegressor()


def _normalise_sequences(sequences, fit_indices):
    # Parallel Welford updates over sequences avoid concatenating millions of
    # frames into a multi-gigabyte float64 temporary array.
    count = 0
    mean64 = np.zeros(768, dtype=np.float64)
    m2 = np.zeros(768, dtype=np.float64)
    for index in fit_indices:
        frames = np.asarray(sequences[int(index)], dtype=np.float64)
        n = len(frames)
        local_mean = frames.mean(axis=0)
        local_m2 = np.square(frames - local_mean).sum(axis=0)
        delta = local_mean - mean64
        total = count + n
        m2 += local_m2 + np.square(delta) * (count * n / total)
        mean64 += delta * (n / total)
        count = total
    mean = mean64.astype(np.float32)
    std = np.sqrt(m2 / max(count, 1)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return [(np.asarray(s, dtype=np.float32) - mean) / std for s in sequences], mean, std


def _loader(sequences, targets, indices, batch_size, shuffle, torch):
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset, Sampler

    class Samples(Dataset):
        def __len__(self): return len(indices)
        def __getitem__(self, item):
            i = int(indices[item])
            return torch.from_numpy(sequences[i]), torch.from_numpy(targets[i].astype(np.float32))

    class LengthBucketBatchSampler(Sampler):
        """Group similar sequence lengths to avoid padding most of each batch."""
        def __iter__(self):
            n = len(indices)
            order = torch.randperm(n).tolist() if shuffle else list(range(n))
            pool_size = batch_size * 24
            batches = []
            for start in range(0, n, pool_size):
                pool = order[start:start + pool_size]
                pool.sort(key=lambda pos: len(sequences[int(indices[pos])]))
                batches.extend(pool[i:i + batch_size] for i in range(0, len(pool), batch_size))
            if shuffle:
                batch_order = torch.randperm(len(batches)).tolist()
                batches = [batches[i] for i in batch_order]
            return iter(batches)

        def __len__(self):
            return (len(indices) + batch_size - 1) // batch_size

    def collate(items):
        xs, ys = zip(*items)
        return pad_sequence(xs, batch_first=True), torch.tensor([len(x) for x in xs]), torch.stack(ys)

    return DataLoader(Samples(), batch_sampler=LengthBucketBatchSampler(), collate_fn=collate)


def _fit_fold(sequences, targets, groups, outer_train, outer_test, variant, seed, device, torch, output_dir, fold):
    from sklearn.model_selection import GroupShuffleSplit

    rel_train, rel_val = next(GroupShuffleSplit(n_splits=1, test_size=INNER_VALIDATION_FRACTION,
                                                 random_state=seed).split(np.zeros(len(outer_train)),
                                                                          groups=np.asarray(groups)[outer_train]))
    inner_train, inner_val = outer_train[rel_train], outer_train[rel_val]
    # Fit frame normalization on inner-training participants only.
    normalized, input_mean, input_std = _normalise_sequences(sequences, inner_train)
    # Fit target normalization from the same inner-training responses only.
    target_mean = targets[inner_train].mean(axis=0).astype(np.float32)
    target_std = targets[inner_train].std(axis=0).astype(np.float32)
    target_std[target_std < 1e-6] = 1.0
    scaled_targets = (targets - target_mean) / target_std
    train_loader = _loader(normalized, scaled_targets, inner_train, BATCH_SIZE, True, torch)
    val_loader = _loader(normalized, scaled_targets, inner_val, BATCH_SIZE, False, torch)
    _seed(seed, torch)
    model = _model(variant["kind"], variant["representation"], torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    try:
        from tqdm.auto import trange
        epoch_range = trange(1, EPOCHS + 1, desc=f"{variant['name']} fold {fold} epochs", leave=False)
    except ImportError:
        epoch_range = range(1, EPOCHS + 1)
    for epoch in epoch_range:
        model.train()
        loss_sum = 0.0
        for padded, lengths, y in train_loader:
            padded, y = padded.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(padded, lengths), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
        train_loss = loss_sum / len(inner_train)
        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for padded, lengths, y in val_loader:
                val_sum += float(loss_fn(model(padded.to(device), lengths), y.to(device)).cpu()) * len(y)
        val_loss = val_sum / len(inner_val)
        history.append({"fold": fold, "epoch": epoch, "train_mse_scaled": train_loss,
                        "validation_mse_scaled": val_loss, "best_epoch_so_far": best_epoch})
        if hasattr(epoch_range, "set_postfix"):
            epoch_range.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", best=best_epoch)
        if val_loss < best_loss - MIN_DELTA:
            best_loss, best_epoch, best_state, stale = val_loss, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is None:
        raise RuntimeError(f"No best checkpoint selected for {variant['name']} fold {fold}")
    model.load_state_dict(best_state)
    ckpt = output_dir / "checkpoints" / variant["name"] / f"fold_{fold}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "input_mean": input_mean, "input_std": input_std,
                "target_mean": target_mean, "target_std": target_std, "best_epoch": best_epoch,
                "best_validation_mse_scaled": best_loss, "variant": variant}, ckpt)
    return model, normalized, target_mean, target_std, best_epoch, best_loss, history, ckpt


def _predict_with_attention(model, sequences, indices, device, torch):
    from torch.nn.utils.rnn import pad_sequence
    model.eval()
    predictions, attention = [], []
    with torch.no_grad():
        for i in indices:
            seq = torch.from_numpy(sequences[int(i)]).unsqueeze(0)
            length = torch.tensor([len(sequences[int(i)])])
            prediction, weights = model(seq.to(device), length, return_attention=True)
            predictions.append(prediction[0].cpu().numpy())
            attention.append(weights[0, :len(sequences[int(i)])].cpu().numpy())
    return np.asarray(predictions), attention


def _metric_row(config_name, target, actual, prediction):
    result = metrics(actual, prediction)
    result["actual_std"] = float(np.std(actual, ddof=0))
    result["predicted_std"] = float(np.std(prediction, ddof=0))
    return {"config": config_name, "target": target, **result}


def _plots(output_dir, config_name, target, actual, prediction, histories):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots" / config_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].scatter(actual, prediction, alpha=.45, s=16)
    lo, hi = min(np.min(actual), np.min(prediction)), max(np.max(actual), np.max(prediction))
    axes[0].plot([lo, hi], [lo, hi], "k--", linewidth=1)
    axes[0].set(xlabel="Actual", ylabel="Predicted", title=f"{target}: actual vs predicted")
    axes[1].hist(actual, bins=24, alpha=.55, label="Actual", density=True)
    axes[1].hist(prediction, bins=24, alpha=.55, label="Predicted", density=True)
    axes[1].set(title=f"{target}: distributions", xlabel="Score", ylabel="Density")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{target}_predictions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold, rows in histories.groupby("fold"):
        ax.plot(rows.epoch, rows.train_mse_scaled, alpha=.8, label=f"fold {fold} train")
        ax.plot(rows.epoch, rows.validation_mse_scaled, linestyle="--", alpha=.8, label=f"fold {fold} validation")
    ax.set(xlabel="Epoch", ylabel="Scaled target MSE", title=f"{config_name}: training curves")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "training_validation_loss.png", dpi=150)
    plt.close(fig)


def _save_attention_maps(output_dir, variant_name, train, sequences, attention_by_index):
    weights_dir = output_dir / "attention_weights" / variant_name
    weights_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for index, weights in attention_by_index.items():
        row = train.iloc[index]
        long_rows.extend({"id": str(row.id), "user_id": str(row.user_id), "frame": frame,
                          "weight": float(weight), "sequence_length": len(weights)}
                         for frame, weight in enumerate(weights))
    pd.DataFrame(long_rows).to_csv(weights_dir / "oof_attention_weights.csv.gz", index=False, compression="gzip")
    # Select three held-out responses distributed through the OOF ordering.
    chosen = sorted(attention_by_index)[::max(1, len(attention_by_index) // 3)][:3]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for index in chosen:
            weights = attention_by_index[index]
            plt.figure(figsize=(9, 2.8))
            plt.plot(np.arange(len(weights)), weights, linewidth=.8)
            plt.xlabel("WavLM frame (temporal order)")
            plt.ylabel("Attention weight")
            plt.title(f"{variant_name} | {train.iloc[index].id} | T={len(weights)}")
            plt.tight_layout()
            plt.savefig(weights_dir / f"{train.iloc[index].id}_attention.png", dpi=150)
            plt.close()
    except ImportError:
        pass


def _existing_baselines(config, train):
    reports = project_path(config["paths"]["reports_dir"])
    rows, frames = [], {}
    ridge_path = reports / "wavlm_gru_pooling_pairs" / "layer7_ridge_oof_predictions.csv"
    if ridge_path.is_file():
        frames["Layer 7 mean+std + Ridge"] = pd.read_csv(ridge_path, dtype={"id": str, "user_id": str})
    else:
        raise FileNotFoundError(f"Saved layer-7 Ridge OOF predictions are required: {ridge_path}")
    baselines = {
        "GRU128 last state, original procedure": reports / "wavlm_gru_ablation" / "layer7_gru128_existing_oof_predictions.csv",
        "GRU128 last state, early stopping + regularization": reports / "wavlm_gru_ablation" / "layer7_gru128_earlystop_reg_oof_predictions.csv",
        "GRU64 last state, early stopping + regularization": reports / "wavlm_gru_ablation" / "layer7_gru64_earlystop_reg_oof_predictions.csv",
        "GRU64 mean pooling": reports / "wavlm_gru_ablation" / "layer7_gru64_meanpool_earlystop_reg_oof_predictions.csv",
        "GRU64 max pooling": reports / "wavlm_gru_pooling_pairs" / "layer7_gru64_maxpool_earlystop_reg_oof_predictions.csv",
        "GRU64 mean+std pooling": reports / "wavlm_gru_pooling_pairs" / "layer7_gru64_meanstdpool_earlystop_reg_oof_predictions.csv",
    }
    for name, path in baselines.items():
        if path.is_file():
            frames[name] = pd.read_csv(path, dtype={"id": str, "user_id": str})
    for name, frame in frames.items():
        if not np.array_equal(frame.id.astype(str), train.id.astype(str)):
            raise ValueError(f"Baseline response ordering differs from current train split: {name}")
        for target in TARGETS:
            actual_col = f"{target}_actual"
            pred_col = f"{target}_prediction"
            if pred_col not in frame:
                continue
            actual = train[target].to_numpy(float)
            pred = frame[pred_col].to_numpy(float)
            rows.append(_metric_row(name, target, actual, pred))
    return rows


def main(argv=None):
    import torch
    from sklearn.model_selection import GroupKFold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--limit", type=int, help="Optional smoke run; uses fewer grouped folds")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    data = load_split_data(config, limit=args.limit)
    train = data["train"].reset_index(drop=True)
    sequences = _frame_cache(config, {"train": train}, args.allow_download, args.limit, frame_layer=7)["train"]
    targets = train[list(TARGETS)].to_numpy(dtype=np.float32)
    groups = train.user_id.astype(str).to_numpy()
    folds = int(config["modeling"].get("smoke_cv_folds", 3) if args.limit else config["modeling"].get("cv_folds", 5))
    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(np.zeros(len(train)), targets, groups))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'}", flush=True)
    output_dir = project_path(config["paths"]["reports_dir"]) / "wavlm_attention_experiments"
    if args.limit:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_predictions, oof_attention, fold_rows, history_rows, model_rows = {}, {}, [], [], []
    for variant in VARIANTS:
        print(f"\n=== {variant['name']} ===", flush=True)
        oof = np.full_like(targets, np.nan, dtype=np.float32)
        attention_map = {}
        for fold, (outer_train, outer_val) in enumerate(_progress(splits, desc=variant["name"], total=folds), start=1):
            if set(groups[outer_train]) & set(groups[outer_val]):
                raise AssertionError("Participant overlap in outer CV fold")
            model, normalized, y_mean, y_std, best_epoch, best_loss, history, ckpt = _fit_fold(
                sequences, targets, groups, outer_train, outer_val, variant, seed + fold, device, torch, output_dir, fold)
            history_rows.extend({"config": variant["name"], **item} for item in history)
            prediction_scaled, weights = _predict_with_attention(model, normalized, outer_val, device, torch)
            prediction = prediction_scaled * y_std + y_mean
            oof[outer_val] = prediction
            for idx, row_weights in zip(outer_val, weights):
                attention_map[int(idx)] = row_weights
            fold_record = {"config": variant["name"], "fold": fold, "best_epoch": best_epoch,
                           "best_inner_validation_mse_scaled": best_loss, "checkpoint": str(ckpt),
                           "outer_train_participants": len(np.unique(groups[outer_train])),
                           "outer_validation_participants": len(np.unique(groups[outer_val]))}
            for j, target in enumerate(TARGETS):
                fold_record.update({f"{target}_{key}": value for key, value in metrics(targets[outer_val, j], prediction[:, j]).items()})
                fold_record[f"{target}_actual_std"] = float(np.std(targets[outer_val, j]))
                fold_record[f"{target}_predicted_std"] = float(np.std(prediction[:, j]))
            fold_rows.append(fold_record)
            print(f"fold {fold}/{folds}: best_epoch={best_epoch}; inner val MSE={best_loss:.4f}", flush=True)
        oof_predictions[variant["name"]] = oof
        oof_attention[variant["name"]] = attention_map
        prediction_frame = pd.DataFrame({"id": train.id.astype(str), "user_id": train.user_id.astype(str)})
        for j, target in enumerate(TARGETS):
            prediction_frame[f"{target}_actual"] = targets[:, j]
            prediction_frame[f"{target}_prediction"] = oof[:, j]
            model_rows.append(_metric_row(variant["name"], target, targets[:, j], oof[:, j]))
        prediction_frame.to_csv(output_dir / f"{variant['name']}_oof_predictions.csv", index=False)
        _save_attention_maps(output_dir, variant["name"], train, sequences, attention_map)
        hist = pd.DataFrame([r for r in history_rows if r["config"] == variant["name"]])
        hist.to_csv(output_dir / f"{variant['name']}_loss_history.csv", index=False)
        for target in TARGETS:
            _plots(output_dir, variant["name"], target, targets[:, TARGETS.index(target)], oof[:, TARGETS.index(target)], hist)

    pd.DataFrame(fold_rows).to_csv(output_dir / "per_fold_metrics.csv", index=False)
    attention_metrics = pd.DataFrame(model_rows)
    attention_metrics.to_csv(output_dir / "attention_model_oof_metrics.csv", index=False)
    baseline_rows = _existing_baselines(config, train)
    all_rows = baseline_rows + model_rows
    pd.DataFrame(all_rows).to_csv(output_dir / "all_wavlm_experiments_comparison.csv", index=False)

    # Paired participant bootstrap against Ridge for each attention model and target.
    ridge = pd.read_csv(project_path(config["paths"]["reports_dir"]) / "wavlm_gru_pooling_pairs" / "layer7_ridge_oof_predictions.csv")
    bootstrap_rows = []
    rng = np.random.default_rng(seed + BOOTSTRAP_SEED)
    users = groups
    unique_users = np.unique(users)
    by_user = {u: np.flatnonzero(users == u) for u in unique_users}
    for name, predictions in oof_predictions.items():
        for j, target in enumerate(TARGETS):
            samples = {key: [] for key in ("mae", "rmse", "spearman", "ccc")}
            for _ in range(args.bootstrap_replicates):
                picked = rng.choice(unique_users, len(unique_users), replace=True)
                ix = np.concatenate([by_user[u] for u in picked])
                a, b = metrics(targets[ix, j], predictions[ix, j]), metrics(targets[ix, j], ridge[f"{target}_prediction"].to_numpy()[ix])
                for key in samples:
                    samples[key].append(a[key] - b[key])
            point_a = metrics(targets[:, j], predictions[:, j])
            point_b = metrics(targets[:, j], ridge[f"{target}_prediction"].to_numpy())
            for key, values in samples.items():
                bootstrap_rows.append({"config": name, "target": target, "metric": key,
                                       "difference_attention_minus_ridge": point_a[key] - point_b[key],
                                       "ci_95_low": float(np.percentile(values, 2.5)),
                                       "ci_95_high": float(np.percentile(values, 97.5)),
                                       "bootstrap_unit": "participant", "replicates": args.bootstrap_replicates})
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "paired_bootstrap_vs_ridge.csv", index=False)
    atomic_json(output_dir / "run_config.json", {
        "seed": seed, "device": str(device), "cv_strategy": "GroupKFold by user_id",
        "cv_folds": folds, "frame_layer_index": 7, "attention_dimension": ATTENTION_DIM,
        "dropout": DROPOUT, "weight_decay": WEIGHT_DECAY, "gradient_clip_norm": GRAD_CLIP,
        "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "max_epochs": EPOCHS,
        "early_stopping": {"inner_split": "GroupShuffleSplit by participant, 15%", "patience": PATIENCE,
                           "min_delta_scaled_mse": MIN_DELTA, "checkpoint": "best inner validation state"},
        "normalization": "per-frame feature mean/std fitted on inner-training participants only",
        "target_scaling": "z-score fitted on inner-training participants; inverse transformed before metrics",
        "test_set_used": False, "configs": list(VARIANTS),
    })
    print("\nOOF metrics:", flush=True)
    print(attention_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    print(f"\nSaved attention experiments and comparison artifacts to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
