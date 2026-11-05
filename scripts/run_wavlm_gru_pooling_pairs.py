"""Compare GRU temporal pooling variants and paired participant-bootstrap them vs Ridge."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_audio_models import (  # noqa: E402
    atomic_json, build_pipeline, grouped_oof, load_config, load_split_data, metrics, project_path,
)
from scripts.run_wavlm_gru_ablation import (  # noqa: E402
    _early_stop_epochs, _fit_epochs, _predict, _progress, WEIGHT_DECAY,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, MIN_DELTA, INNER_VALIDATION_FRACTION,
)
from scripts.train_wavlm_gru import TARGETS, _frame_cache  # noqa: E402

VARIANTS = (
    {"name": "layer7_gru64_maxpool_earlystop_reg", "hidden_size": 64, "early_stopping": True, "regularization": True, "pooling": "max"},
    {"name": "layer7_gru64_meanstdpool_earlystop_reg", "hidden_size": 64, "early_stopping": True, "regularization": True, "pooling": "mean_std"},
)
REFERENCE_GRU = "layer7_gru64_meanpool_earlystop_reg"
RIDGE_NAME = "layer7_ridge_baseline"
BOOTSTRAP_REPLICATES = 1000


def _load_existing_gru(path, train):
    frame = pd.read_csv(path, dtype={"id": str, "user_id": str})
    expected = train[["id", "user_id"]].copy()
    expected["id"] = expected.id.astype(str)
    expected["user_id"] = expected.user_id.astype(str)
    if not frame[["id", "user_id"]].equals(expected.reset_index(drop=True)):
        raise ValueError(f"Existing GRU prediction IDs/order do not match training rows: {path}")
    actual = np.column_stack([frame[f"{target}_actual"].to_numpy(float) for target in TARGETS])
    prediction = np.column_stack([frame[f"{target}_prediction"].to_numpy(float) for target in TARGETS])
    return frame, actual, prediction


def _ridge_oof(config, train, folds, seed):
    embeddings_path = project_path(config["paths"]["embeddings_dir"]) / "wavlm" / "train_embeddings.npz"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Missing pooled WavLM baseline input: {embeddings_path}")
    with np.load(embeddings_path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str)
        pooled = archive["pooled"].astype(np.float32)
    positions = {value: index for index, value in enumerate(ids)}
    expected_ids = train.id.astype(str).to_numpy()
    missing = [value for value in expected_ids if value not in positions]
    if missing:
        raise ValueError(f"Ridge pooled cache is missing {len(missing)} training response IDs")
    order = np.asarray([positions[value] for value in expected_ids], dtype=int)
    if pooled.ndim != 3 or pooled.shape[1] <= 7 or pooled.shape[2] != 1536:
        raise ValueError(f"Expected pooled WavLM layer features [N, layers, 1536], got {pooled.shape}")
    features = pooled[order, 7, :]
    ridge_report = pd.read_csv(project_path(config["paths"]["reports_dir"]) / "comparison.csv")
    baseline = ridge_report[(ridge_report.feature_set == "D_WavLM") & (ridge_report.model == "Ridge")]
    predictions = np.full((len(train), len(TARGETS)), np.nan, dtype=float)
    records = []
    groups = train.user_id.astype(str).to_numpy()
    for column, target in enumerate(TARGETS):
        target_row = baseline[baseline.target == target]
        if len(target_row) != 1:
            raise ValueError(f"Expected exactly one saved layer-7 Ridge row for {target}")
        params = json.loads(target_row.iloc[0].params)
        alpha = float(params["alpha"])
        curve_path = project_path(config["paths"]["reports_dir"]) / f"layer_sweep_wavlm_{target}.csv"
        curve = pd.read_csv(curve_path)
        selected_layer = curve[curve.layer.astype(str) == "layer_7"]
        if len(selected_layer) != 1 or not np.isclose(float(selected_layer.iloc[0].alpha), alpha):
            raise ValueError(f"Saved Ridge setting for {target} is not layer 7 / alpha {alpha}")
        X = pd.DataFrame(features, columns=[f"wavlm_{i}" for i in range(features.shape[1])])
        y = train[target].astype(float).reset_index(drop=True)
        oof, cv_metrics, fold_metrics = grouped_oof(
            X, y, groups, "Ridge", {"alpha": alpha}, folds, seed,
            standardize=True,
            pca_components=config["modeling"].get("embedding_pca_components"),
            stable_ridge=True,
        )
        predictions[:, column] = oof
        records.append({"target": target, "alpha": alpha, **cv_metrics,
                        "fold_rmse_mean": float(np.mean([x["rmse"] for x in fold_metrics]))})
    return predictions, records


def _fit_variant(config, train, sequences, targets, groups, folds, seed, device, torch, output_dir, setting):
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(np.zeros(len(train)), targets, groups))
    oof = np.full_like(targets, np.nan, dtype=np.float32)
    fold_rows, history_rows = [], []
    for fold, (outer_train, outer_val) in enumerate(_progress(splits, desc=setting["name"], total=folds), start=1):
        best_epochs, best_loss, history, n_inner_train, n_inner_val = _early_stop_epochs(
            sequences, targets, outer_train, groups, setting, seed + fold, device, torch)
        history_rows.extend({"fold": fold, **row} for row in history)
        model = _fit_epochs(sequences, targets, outer_train, best_epochs, setting, seed + fold, device, torch)
        prediction = _predict(model, sequences, outer_val, BATCH_SIZE, device, torch)
        oof[outer_val] = prediction
        row = {"config": setting["name"], "fold": fold, "epochs_used": best_epochs,
               "inner_best_mse": best_loss, "inner_train_participants": n_inner_train,
               "inner_validation_participants": n_inner_val,
               "outer_train_participants": len(np.unique(groups[outer_train])),
               "outer_validation_participants": len(np.unique(groups[outer_val]))}
        for j, target in enumerate(TARGETS):
            row.update({f"{target}_{key}": value for key, value in metrics(targets[outer_val, j], prediction[:, j]).items()})
        fold_rows.append(row)
        print(f"fold {fold}/{folds}: epochs={best_epochs}; held-out MAE " +
              ", ".join(f"{target} {metrics(targets[outer_val,j],prediction[:,j])['mae']:.3f}" for j,target in enumerate(TARGETS)), flush=True)
    pred_frame = pd.DataFrame({"id": train.id.astype(str), "user_id": train.user_id.astype(str)})
    for j, target in enumerate(TARGETS):
        pred_frame[f"{target}_actual"] = targets[:, j]
        pred_frame[f"{target}_prediction"] = oof[:, j]
    pred_frame.to_csv(output_dir / f"{setting['name']}_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / f"{setting['name']}_fold_metrics.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output_dir / f"{setting['name']}_early_stopping_history.csv", index=False)
    summary = [{"config": setting["name"], "target": target, **metrics(targets[:, j], oof[:, j])}
               for j,target in enumerate(TARGETS)]
    return oof, summary


def _participant_bootstrap(y, pred_a, pred_b, groups, replicates, seed):
    """Paired bootstrap sampling whole participant clusters with replacement."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups, dtype=str)
    participants = np.unique(groups)
    by_participant = {user: np.flatnonzero(groups == user) for user in participants}
    point_a = metrics(y, pred_a)
    point_b = metrics(y, pred_b)
    names = ("mae", "rmse", "spearman", "ccc")
    boot = {name: np.empty(replicates, dtype=float) for name in names}
    for rep in range(replicates):
        sampled_users = rng.choice(participants, size=len(participants), replace=True)
        indices = np.concatenate([by_participant[user] for user in sampled_users])
        score_a = metrics(y[indices], pred_a[indices])
        score_b = metrics(y[indices], pred_b[indices])
        for name in names:
            boot[name][rep] = score_a[name] - score_b[name]
    return {name: {"difference_gru_minus_ridge": point_a[name] - point_b[name],
                   "ci_95_low": float(np.nanpercentile(values, 2.5)),
                   "ci_95_high": float(np.nanpercentile(values, 97.5)),
                   "bootstrap_probability_difference_gt_zero": float(np.mean(values > 0))}
            for name, values in boot.items()}


def main(argv=None):
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["seed"])
    data = load_split_data(config)
    train = data["train"].reset_index(drop=True)
    folds = int(config["modeling"].get("cv_folds", 5))
    targets = train[list(TARGETS)].to_numpy(np.float32)
    groups = train.user_id.astype(str).to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("[device] CUDA unavailable; using CPU", flush=True)
    output_dir = project_path(config["paths"]["reports_dir"]) / "wavlm_gru_pooling_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = project_path(config["paths"]["reports_dir"]) / "wavlm_gru_ablation"
    _, reference_actual, reference_prediction = _load_existing_gru(
        ablation_dir / f"{REFERENCE_GRU}_oof_predictions.csv", train)
    ridge_prediction, ridge_metrics = _ridge_oof(config, train, folds, seed)
    ridge_frame = pd.DataFrame({"id": train.id.astype(str), "user_id": train.user_id.astype(str)})
    for j, target in enumerate(TARGETS):
        ridge_frame[f"{target}_actual"] = targets[:, j]
        ridge_frame[f"{target}_prediction"] = ridge_prediction[:, j]
    ridge_frame.to_csv(output_dir / "layer7_ridge_oof_predictions.csv", index=False)

    # Confirm matching response rows and labels before any paired calculation.
    if not np.allclose(reference_actual, targets, rtol=0, atol=1e-6):
        raise ValueError("Saved GRU OOF actual values do not match the authoritative train labels")
    for target in TARGETS:
        ridge_row = next(row for row in ridge_metrics if row["target"] == target)
        baseline_row = pd.read_csv(project_path(config["paths"]["reports_dir"]) / "comparison.csv")
        baseline_row = baseline_row[(baseline_row.feature_set == "D_WavLM") & (baseline_row.model == "Ridge") & (baseline_row.target == target)].iloc[0]
        if not np.isclose(ridge_row["rmse"], float(baseline_row.cv_rmse), atol=1e-10):
            raise AssertionError(f"Reproduced Ridge OOF RMSE does not match saved baseline for {target}: {ridge_row['rmse']} vs {baseline_row.cv_rmse}")
    pd.DataFrame(ridge_metrics).to_csv(output_dir / "ridge_oof_metrics.csv", index=False)

    sequences = _frame_cache(config, {"train": train}, args.allow_download, frame_layer=7)["train"]
    variant_metrics, prediction_frames = [], {}
    for setting in VARIANTS:
        print(f"\n=== {setting['name']} ===", flush=True)
        prediction, summary = _fit_variant(config, train, sequences, targets, groups, folds, seed, device, torch, output_dir, setting)
        variant_metrics.extend(summary)
        prediction_frames[setting["name"]] = prediction
    reference_name = REFERENCE_GRU
    prediction_frames[reference_name] = reference_prediction

    paired_rows = []
    for model_name, predictions in prediction_frames.items():
        for j, target in enumerate(TARGETS):
            paired = _participant_bootstrap(targets[:, j], predictions[:, j], ridge_prediction[:, j], groups,
                                            args.bootstrap_replicates, seed + 100 + j)
            for metric_name, result in paired.items():
                paired_rows.append({"gru_config": model_name, "ridge_config": RIDGE_NAME,
                                    "target": target, "metric": metric_name,
                                    "difference_direction": "GRU minus Ridge",
                                    "difference_interpretation": "negative favors GRU for MAE/RMSE; positive favors GRU for Spearman/CCC",
                                    "bootstrap_unit": "participant", "bootstrap_replicates": args.bootstrap_replicates,
                                    **result})
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_participant_bootstrap.csv", index=False)

    combined = pd.DataFrame({"id": train.id.astype(str), "user_id": train.user_id.astype(str)})
    for j, target in enumerate(TARGETS):
        combined[f"{target}_actual"] = targets[:, j]
        combined[f"{target}_ridge_prediction"] = ridge_prediction[:, j]
        combined[f"{target}_{reference_name}_prediction"] = reference_prediction[:, j]
        for name, prediction in prediction_frames.items():
            if name == reference_name:
                continue
            combined[f"{target}_{name}_prediction"] = prediction[:, j]
    combined.to_csv(output_dir / "paired_oof_predictions.csv", index=False)
    pd.DataFrame(variant_metrics).to_csv(output_dir / "pooling_variant_metrics.csv", index=False)
    atomic_json(output_dir / "run_config.json", {
        "seed": seed, "cv_strategy": "GroupKFold by user_id", "cv_folds": folds,
        "frame_layer_index": 7, "hidden_size": 64, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE, "max_epochs": EPOCHS,
        "early_stopping": {"inner_split": f"GroupShuffleSplit by participant, {INNER_VALIDATION_FRACTION:.0%}",
                           "patience": PATIENCE, "min_delta": MIN_DELTA,
                           "refit_outer_training_fold_for_best_epoch_count": True},
        "ridge_baseline": {"representation": "WavLM layer 7 mean+std", "alphas_by_target": {row["target"]: row["alpha"] for row in ridge_metrics}},
        "participant_bootstrap_replicates": args.bootstrap_replicates,
        "configs": [*VARIANTS, {"name": REFERENCE_GRU, "source": "previous grouped OOF predictions"}],
    })
    print("\nPaired participant-bootstrap differences (GRU minus Ridge)", flush=True)
    print(pd.DataFrame(paired_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
