"""Run four layer-7 WavLM GRU ablations on identical participant GroupKFold splits.

Configuration 1 reproduces the existing 128-unit, 40-epoch training procedure.
Configurations 2-4 use nested participant validation for early stopping, then
refit on the complete outer training fold for the selected epoch count.
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
from scripts.legacy.train_wavlm_gru import TARGETS, _frame_cache, _make_model, _progress

CONFIGS = (
    {"name": "layer7_gru128_existing", "hidden_size": 128, "early_stopping": False, "regularization": False, "pooling": "last"},
    {"name": "layer7_gru128_earlystop_reg", "hidden_size": 128, "early_stopping": True, "regularization": True, "pooling": "last"},
    {"name": "layer7_gru64_earlystop_reg", "hidden_size": 64, "early_stopping": True, "regularization": True, "pooling": "last"},
    {"name": "layer7_gru64_meanpool_earlystop_reg", "hidden_size": 64, "early_stopping": True, "regularization": True, "pooling": "mean"},
)

EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.05  # Stronger than the existing AdamW default (0.01).
PATIENCE = 5
MIN_DELTA = 1e-4
INNER_VALIDATION_FRACTION = 0.15


def _seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(sequences, targets, indices, batch_size, shuffle, torch):
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader, Dataset

    class Samples(Dataset):
        def __len__(self): return len(indices)
        def __getitem__(self, item):
            index = int(indices[item])
            return torch.from_numpy(sequences[index]), torch.from_numpy(targets[index].astype(np.float32))

    def collate(items):
        xs, ys = zip(*items)
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        return pad_sequence(xs, batch_first=True), lengths, torch.stack(ys)

    return DataLoader(Samples(), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def _evaluate_loss(model, loader, loss_fn, device):
    import torch
    model.eval()
    losses, weights = [], []
    with torch.no_grad():
        for padded, lengths, y in loader:
            y = y.to(device)
            loss = loss_fn(model(padded.to(device), lengths), y)
            losses.append(float(loss.cpu()))
            weights.append(len(y))
    return float(np.average(losses, weights=weights))


def _fit_epochs(sequences, targets, train_indices, epochs, setting, seed, device, torch, history=None):
    _seed_everything(seed, torch)
    train_loader = _loader(sequences, targets, train_indices, BATCH_SIZE, True, torch)
    model = _make_model(setting["hidden_size"], len(TARGETS), setting["pooling"]).to(device)
    decay = WEIGHT_DECAY if setting["regularization"] else 0.01  # Existing AdamW default.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=decay)
    loss_fn = torch.nn.MSELoss()
    records = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, count = 0.0, 0
        for padded, lengths, y in train_loader:
            padded, y = padded.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(padded, lengths), y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(y)
            count += len(y)
        records.append({"epoch": epoch, "train_mse": running / max(1, count)})
    if history is not None:
        history.extend(records)
    return model


def _early_stop_epochs(sequences, targets, outer_train, groups, setting, seed, device, torch):
    from sklearn.model_selection import GroupShuffleSplit

    group_values = np.asarray(groups[outer_train], dtype=str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=INNER_VALIDATION_FRACTION, random_state=seed)
    inner_train_rel, inner_val_rel = next(splitter.split(np.zeros(len(outer_train)), groups=group_values))
    inner_train = outer_train[inner_train_rel]
    inner_val = outer_train[inner_val_rel]
    train_loader = _loader(sequences, targets, inner_train, BATCH_SIZE, True, torch)
    val_loader = _loader(sequences, targets, inner_val, BATCH_SIZE, False, torch)
    _seed_everything(seed, torch)
    model = _make_model(setting["hidden_size"], len(TARGETS), setting["pooling"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = torch.nn.MSELoss()
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, count = 0.0, 0
        for padded, lengths, y in train_loader:
            padded, y = padded.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(padded, lengths), y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(y)
            count += len(y)
        train_mse = running / max(1, count)
        val_mse = _evaluate_loss(model, val_loader, loss_fn, device)
        history.append({"epoch": epoch, "train_mse": train_mse, "inner_validation_mse": val_mse})
        if val_mse < best_loss - MIN_DELTA:
            best_loss, best_epoch, best_state, stale = val_mse, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    return max(1, best_epoch), best_loss, history, len(np.unique(group_values[inner_train_rel])), len(np.unique(group_values[inner_val_rel]))


def _predict(model, sequences, indices, batch_size, device, torch):
    from torch.nn.utils.rnn import pad_sequence
    model.eval()
    results = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            seqs = [torch.from_numpy(sequences[int(i)]) for i in batch]
            lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
            padded = pad_sequence(seqs, batch_first=True).to(device)
            results.append(model(padded, lengths).cpu().numpy())
    return np.concatenate(results, axis=0)


def main(argv=None):
    import torch
    from sklearn.model_selection import GroupKFold

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int, help="Optional smoke run on first N training responses")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    data = load_split_data(config, limit=args.limit)
    train = data["train"].reset_index(drop=True)
    train_only = {"train": train}
    sequences = _frame_cache(config, train_only, args.allow_download, args.limit, frame_layer=7)["train"]
    targets = train[list(TARGETS)].to_numpy(dtype=np.float32)
    groups = train.user_id.astype(str).to_numpy()
    folds = int(config["modeling"].get("smoke_cv_folds", 3) if args.limit else config["modeling"].get("cv_folds", 5))
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} participants for GroupKFold")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(device)}; layer-7 embeddings; GRU training on CUDA", flush=True)
    else:
        print("[device] CUDA unavailable; GRU training on CPU", flush=True)

    output_dir = project_path(config["paths"]["reports_dir"]) / "wavlm_gru_ablation"
    if args.limit:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = GroupKFold(n_splits=folds)
    fold_splits = list(splitter.split(np.zeros(len(train)), targets, groups))
    summary_rows, fold_rows = [], []
    for setting in CONFIGS:
        print(f"\n=== {setting['name']} ===", flush=True)
        oof = np.full_like(targets, np.nan, dtype=np.float32)
        setting_history = []
        for fold, (outer_train, outer_val) in enumerate(_progress(fold_splits, desc=setting["name"], total=folds), start=1):
            if set(groups[outer_train]) & set(groups[outer_val]):
                raise AssertionError(f"Participant overlap in outer fold {fold}")
            fold_seed = seed + fold
            if setting["early_stopping"]:
                best_epochs, best_loss, inner_history, inner_train_users, inner_val_users = _early_stop_epochs(
                    sequences, targets, outer_train, groups, setting, fold_seed, device, torch)
                for item in inner_history:
                    setting_history.append({"fold": fold, **item})
                model = _fit_epochs(sequences, targets, outer_train, best_epochs, setting, fold_seed, device, torch)
            else:
                best_epochs, best_loss, inner_train_users, inner_val_users = EPOCHS, np.nan, 0, 0
                model = _fit_epochs(sequences, targets, outer_train, EPOCHS, setting, fold_seed, device, torch)
            prediction = _predict(model, sequences, outer_val, BATCH_SIZE, device, torch)
            oof[outer_val] = prediction
            fold_record = {"config": setting["name"], "fold": fold, "epochs_used": best_epochs,
                           "inner_best_mse": best_loss, "inner_train_participants": inner_train_users,
                           "inner_validation_participants": inner_val_users,
                           **{f"{group}_participants": len(np.unique(groups[idx]))
                              for group, idx in (("outer_train", outer_train), ("outer_validation", outer_val))}}
            for j, target in enumerate(TARGETS):
                fold_record.update({f"{target}_{key}": value for key, value in metrics(targets[outer_val, j], prediction[:, j]).items()})
            fold_rows.append(fold_record)
            print(f"fold {fold}/{folds}: epochs={best_epochs}, held-out participant MAE=" +
                  ", ".join(f"{t} {metrics(targets[outer_val,j],prediction[:,j])['mae']:.3f}" for j,t in enumerate(TARGETS)), flush=True)
        for j, target in enumerate(TARGETS):
            summary_rows.append({"config": setting["name"], "target": target, **metrics(targets[:, j], oof[:, j])})
        prediction_frame = pd.DataFrame({"id": train.id.astype(str), "user_id": train.user_id.astype(str)})
        for j, target in enumerate(TARGETS):
            prediction_frame[f"{target}_actual"] = targets[:, j]
            prediction_frame[f"{target}_prediction"] = oof[:, j]
        prediction_frame.to_csv(output_dir / f"{setting['name']}_oof_predictions.csv", index=False)
        if setting_history:
            pd.DataFrame(setting_history).to_csv(output_dir / f"{setting['name']}_early_stopping_history.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output_dir / "metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
        print(pd.DataFrame([r for r in summary_rows if r["config"] == setting["name"]]).to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    atomic_json(output_dir / "run_config.json", {
        "seed": seed, "cv_strategy": "GroupKFold by user_id", "cv_folds": folds,
        "frame_layer_index": 7, "hidden_state_index_convention": "0 is feature encoder; 1..12 are Transformer outputs",
        "epochs_max": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "early_stopping": {"inner_split": "GroupShuffleSplit by participant, 15%", "patience": PATIENCE, "min_delta": MIN_DELTA,
                           "refit_outer_training_fold_for_best_epoch_count": True},
        "mean_pool_config": "mean over valid GRU output states; padded positions excluded",
        "configs": list(CONFIGS), "smoke_limit": args.limit,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
