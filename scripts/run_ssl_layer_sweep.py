"""Layer-wise WavLM/HuBERT pooling benchmark on fixed participant splits.

All candidate layer/pooling choices use train-only GroupKFold (user_id) for
Ridge alpha selection and the existing held-out validation participants for
the final comparison. Validation and test rows are never used to fit layer
weights. Existing full hidden-state caches are reused when available.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_audio_models import (  # noqa: E402
    bootstrap_by_participant,
    build_pipeline,
    load_config,
    load_split_data,
    metrics,
    project_path,
)

TARGETS = ("confidence_score", "speaking_skills", "overall_performance")
ENCODERS = ("wavlm", "hubert")
POOLINGS = ("mean", "std", "mean+std")
COMPARISON_METHODS = ("current_mean_std", "best_individual", "weighted_layers")


def _fingerprint(config, encoder, model_name, pooled_shape, source_schema):
    payload = {
        "cache_schema_version": 1,
        "encoder": encoder,
        "model_name": model_name,
        "audio": config["audio"],
        "speech_only_pooling": bool(config["extractors"][encoder].get("speech_only_pooling", False)),
        "feature_extractor": "Hugging Face AutoFeatureExtractor; normalized mono float waveform at configured sample rate",
        "hidden_states": "all outputs; index 0 is feature-encoder output, 1..N are transformer block outputs",
        "pooling_methods": list(POOLINGS),
        "pooled_shape": list(pooled_shape),
        "source_schema": source_schema,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _load_cached_features(config, data, encoder, cache_root):
    """Load cached all-layer mean/std vectors and cache their split-aligned form."""
    source_root = project_path(config["paths"]["embeddings_dir"]) / encoder
    derived_root = cache_root / encoder
    model_name = config["extractors"][encoder]["model_name"]
    output = {}
    for split in ("train", "val"):
        source_path = source_root / f"{split}_embeddings.npz"
        schema_path = source_root / f"{split}_embedding_schema.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing cached {encoder} embeddings: {source_path}. Run feature extraction first.")
        schema = json.loads(schema_path.read_text(encoding="utf-8")) if schema_path.is_file() else {}
        with np.load(source_path, allow_pickle=False) as archive:
            ids = archive["ids"].astype(str)
            pooled = archive["pooled"].astype(np.float32)
            archived_model = str(archive["model_name"].item()) if "model_name" in archive else schema.get("model_name")
            archive_means = archive["means"].astype(np.float32) if "means" in archive else None
            archive_stds = archive["stds"].astype(np.float32) if "stds" in archive else None
        if archived_model and archived_model != model_name:
            raise ValueError(f"Cached checkpoint mismatch for {encoder}/{split}: expected {model_name}, found {archived_model}")
        if pooled.ndim != 3 or pooled.shape[-1] % 2:
            raise ValueError(f"Expected [clip, layer, 2*hidden] pooled tensor in {source_path}, got {pooled.shape}")
        hidden = pooled.shape[-1] // 2
        means = archive_means if archive_means is not None else pooled[..., :hidden]
        stds = archive_stds if archive_stds is not None else pooled[..., hidden:]
        if means.shape != stds.shape or means.shape != (*pooled.shape[:2], hidden):
            raise ValueError(f"Mean/std cache shape mismatch for {encoder}/{split}")
        expected_ids = data[split].id.astype(str).to_numpy()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {source_path}")
        position = {clip_id: index for index, clip_id in enumerate(ids)}
        absent = [clip_id for clip_id in expected_ids if clip_id not in position]
        if absent:
            raise ValueError(f"{encoder}/{split} cache is missing {len(absent)} fixed-split clips")
        order = np.asarray([position[clip_id] for clip_id in expected_ids], dtype=int)
        means = means[order]
        stds = stds[order]
        source_signature = {"schema": schema, "size_bytes": source_path.stat().st_size,
                            "mtime_ns": source_path.stat().st_mtime_ns}
        cache_fingerprint, cache_metadata = _fingerprint(config, encoder, model_name, pooled.shape[1:], source_signature)
        derived_root.mkdir(parents=True, exist_ok=True)
        derived_path = derived_root / f"{split}_all_layers_{cache_fingerprint[:16]}.npz"
        if derived_path.is_file():
            with np.load(derived_path, allow_pickle=False) as archive:
                if (str(archive["fingerprint"].item()) == cache_fingerprint
                        and np.array_equal(archive["ids"].astype(str), expected_ids)
                        and archive["means"].shape == means.shape):
                    means = archive["means"].astype(np.float32)
                    stds = archive["stds"].astype(np.float32)
                else:
                    derived_path.unlink()
        if not derived_path.is_file():
            from scripts.extract_audio_features import atomic_npz
            atomic_npz(derived_path, ids=expected_ids, means=means, stds=stds,
                       model_name=np.asarray(model_name), encoder=np.asarray(encoder),
                       fingerprint=np.asarray(cache_fingerprint), metadata=np.asarray(json.dumps(cache_metadata, sort_keys=True)))
        output[split] = {"ids": expected_ids, "means": means, "stds": stds, "model_name": model_name,
                         "fingerprint": cache_fingerprint, "metadata": cache_metadata, "source_path": str(source_path)}
    return output


def _features(cache, split, layer, pooling):
    means = cache[split]["means"][:, layer, :]
    stds = cache[split]["stds"][:, layer, :]
    if pooling == "mean":
        return means
    if pooling == "std":
        return stds
    return np.concatenate((means, stds), axis=1)


def _cv_predictions(X, y, groups, alpha, folds, seed):
    from sklearn.model_selection import GroupKFold

    unique_users = np.asarray(groups, dtype=str)
    n_splits = min(int(folds), len(np.unique(unique_users)))
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_records = []
    for fold_id, (tr, va) in enumerate(splitter.split(X, y, groups=unique_users), start=1):
        tr_users, va_users = set(unique_users[tr]), set(unique_users[va])
        if tr_users & va_users:
            raise AssertionError(f"GroupKFold participant overlap in fold {fold_id}")
        model = build_pipeline(X.iloc[tr], "Ridge", {"alpha": float(alpha)}, seed, standardize=True, stable_ridge=True)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        fold_records.append({"fold": fold_id, **metrics(y.iloc[va], pred), "train_participants": len(tr_users), "validation_participants": len(va_users)})
    return oof, fold_records


def _evaluate_representation(Xtr, ytr, Xval, yval, groups, alphas, folds, seed, cached_curve=None):
    """Tune alpha on train-only grouped folds, then refit train and score validation."""
    best = None
    if cached_curve is None:
        for alpha in alphas:
            oof, fold_records = _cv_predictions(Xtr, ytr, groups, alpha, folds, seed)
            fold_rmse = np.asarray([item["rmse"] for item in fold_records], dtype=float)
            result = {"alpha": float(alpha), "oof": oof, "folds": fold_records,
                      "cv_fold_rmse_mean": float(fold_rmse.mean()), "cv_fold_rmse_std": float(fold_rmse.std()),
                      **{f"cv_{key}": value for key, value in metrics(ytr, oof).items()}}
            if best is None or result["cv_fold_rmse_mean"] < best["cv_fold_rmse_mean"]:
                best = result
    else:
        best = {"alpha": float(cached_curve["alpha"]),
                "cv_fold_rmse_mean": float(cached_curve["cv_rmse_fold_mean"]),
                "cv_fold_rmse_std": float(cached_curve["cv_rmse_fold_std"]),
                **{f"cv_{key}": float(cached_curve[f"cv_{key}"]) for key in ("mae", "rmse", "r2", "pearson", "spearman")}}
        best["oof"], best["folds"] = _cv_predictions(Xtr, ytr, groups, best["alpha"], folds, seed)
    model = build_pipeline(Xtr, "Ridge", {"alpha": best["alpha"]}, seed, standardize=True, stable_ridge=True)
    model.fit(Xtr, ytr)
    val_pred = model.predict(Xval)
    return {**best, "model": model, "val_pred": val_pred, "val_metrics": metrics(yval, val_pred)}


def _load_reusable_mean_std_curve(config, encoder, target, layers, alpha_grid, reuse):
    if not reuse:
        return {}
    old_path = project_path(config["paths"]["reports_dir"]) / f"layer_sweep_{encoder}_{target}.csv"
    if not old_path.is_file():
        return {}
    try:
        frame = pd.read_csv(old_path)
        required = {"layer", "layer_index", "alpha", "cv_rmse_fold_mean", "cv_rmse_fold_std", "cv_mae", "cv_rmse", "cv_r2", "cv_pearson", "cv_spearman"}
        if not required.issubset(frame.columns):
            return {}
        frame = frame[frame.layer.astype(str).str.fullmatch(r"layer_\d+")].copy()
        frame["layer_index"] = pd.to_numeric(frame.layer_index, errors="coerce")
        frame = frame[frame.layer_index.isin(layers)]
        if len(frame) < len(layers):
            return {}
        frame = frame.drop_duplicates("layer_index", keep="last").set_index("layer_index")
        # Never borrow a score generated with a different alpha grid.
        if any(float(frame.loc[layer, "alpha"]) not in set(float(a) for a in alpha_grid) for layer in layers):
            return {}
        return {int(layer): frame.loc[layer].to_dict() for layer in layers}
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return {}


def _learn_simplex_weights(oof_matrix, y):
    """Nonnegative sum-to-one blend fitted only on training participant-OOF predictions."""
    from scipy.optimize import minimize

    valid = np.isfinite(y) & np.isfinite(oof_matrix).all(axis=1)
    matrix, target = oof_matrix[valid], y[valid]
    if matrix.shape[1] == 1:
        return np.ones(1, dtype=float), float(np.mean((target - matrix[:, 0]) ** 2))
    count = matrix.shape[1]
    objective = lambda weights: float(np.mean((target - matrix @ weights) ** 2))
    result = minimize(objective, np.full(count, 1 / count), method="SLSQP",
                      bounds=[(0.0, 1.0)] * count,
                      constraints=[{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}],
                      options={"maxiter": 2000, "ftol": 1e-10})
    weights = result.x if result.success else np.full(count, 1 / count)
    weights = np.clip(weights, 0, None)
    weights /= weights.sum()
    return weights, objective(weights)


def _paired_difference_ci(frame_a, frame_b, metric_name, replicates, seed):
    """Participant-cluster paired bootstrap for metric(A)-metric(B)."""
    cols = ["id", "user_id", "actual", "prediction"]
    a = frame_a[cols].rename(columns={"prediction": "pred_a"})
    b = frame_b[cols].rename(columns={"prediction": "pred_b"})
    paired = a.merge(b, on=["id", "user_id", "actual"], validate="one_to_one")
    if paired.empty:
        return {"point_delta": np.nan, "lower_95": np.nan, "upper_95": np.nan}
    def score(y, p):
        result = metrics(y, p)
        return float(result[metric_name])
    point = score(paired.actual.to_numpy(), paired.pred_a.to_numpy()) - score(paired.actual.to_numpy(), paired.pred_b.to_numpy())
    by_user = {user: group.index.to_numpy() for user, group in paired.groupby("user_id", sort=False)}
    users = np.asarray(list(by_user))
    rng = np.random.default_rng(seed)
    deltas = []
    actual = paired.actual.to_numpy(float); pred_a = paired.pred_a.to_numpy(float); pred_b = paired.pred_b.to_numpy(float)
    for _ in range(replicates):
        sample_users = rng.choice(users, size=len(users), replace=True)
        ix = np.concatenate([by_user[user] for user in sample_users])
        delta = score(actual[ix], pred_a[ix]) - score(actual[ix], pred_b[ix])
        if np.isfinite(delta):
            deltas.append(delta)
    return {"point_delta": point,
            "lower_95": float(np.quantile(deltas, .025)) if deltas else np.nan,
            "upper_95": float(np.quantile(deltas, .975)) if deltas else np.nan}


def _line_plot(rows, encoder, target, metric, path):
    poolings = POOLINGS
    series_rows = [row for row in rows if row["encoder"] == encoder and row["target"] == target
                   and str(row.get("layer_index", "")).isdigit()]
    width, height = 850, 480; left, right, top, bottom = 75, 25, 60, 75
    x0, x1, y0, y1 = left, width-right, height-bottom, top
    values = [float(row[f"val_{metric}"]) for row in series_rows if row["pooling"] in poolings and np.isfinite(float(row[f"val_{metric}"]))]
    if not values:
        return
    low, high = min(values), max(values)
    margin = max((high-low)*.12, .02); low -= margin; high += margin
    colors = {"mean": "#3572A5", "std": "#CA5B49", "mean+std": "#39825B"}
    layer_count = max(int(row["layer_index"]) for row in series_rows) + 1
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{width/2}" y="30" text-anchor="middle" font-size="18" font-weight="bold">{encoder.upper()} {target}: validation {metric.upper()} by layer</text>',
             f'<path d="M{x0},{y1}V{y0}H{x1}" fill="none" stroke="#455a64"/>']
    X=lambda layer: x0 + layer/max(1,layer_count-1)*(x1-x0)
    Y=lambda value: y0 - (value-low)/(high-low)*(y0-y1)
    for tick in np.linspace(low, high, 5):
        parts.append(f'<path d="M{x0},{Y(tick):.1f}H{x1}" stroke="#eceff1"/>')
        parts.append(f'<text x="{x0-9}" y="{Y(tick)+4:.1f}" text-anchor="end" font-size="10">{tick:.3f}</text>')
    for layer in range(layer_count):
        parts.append(f'<text x="{X(layer):.1f}" y="{y0+20}" text-anchor="middle" font-size="10">{layer}</text>')
    for pooling in poolings:
        chosen = sorted([row for row in series_rows if row["pooling"] == pooling], key=lambda row: int(row["layer_index"]))
        if not chosen:
            continue
        coords = " ".join(f'{X(int(row["layer_index"])):.1f},{Y(float(row[f"val_{metric}"])):.1f}' for row in chosen)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{colors[pooling]}" stroke-width="2"/>')
        for row in chosen:
            parts.append(f'<circle cx="{X(int(row["layer_index"])):.1f}" cy="{Y(float(row[f"val_{metric}"])):.1f}" r="3" fill="{colors[pooling]}"/>')
    for index, pooling in enumerate(poolings):
        x = x0 + index*150
        parts += [f'<path d="M{x},{height-25}h24" stroke="{colors[pooling]}" stroke-width="3"/>', f'<text x="{x+31}" y="{height-20}" font-size="12">{pooling}</text>']
    parts += [f'<text x="{(x0+x1)/2}" y="{height-3}" text-anchor="middle" font-size="12">Hidden-state output index (0=feature encoder, 1–12=Transformer blocks)</text>', "</svg>"]
    path.write_text("\n".join(parts), encoding="utf-8")


def run(config, layers="all", bootstrap_replicates=None, reuse_current=True):
    result_dir = ROOT / "results" / "ssl_layer_sweep"
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_root = project_path(config["paths"].get("cache_dir", "outputs/audio/cache")) / "ssl_layer_sweep"
    data = load_split_data(config)
    caches = {encoder: _load_cached_features(config, data, encoder, cache_root) for encoder in ENCODERS}
    layer_count = min(caches[encoder]["train"]["means"].shape[1] for encoder in ENCODERS)
    if layers == "all":
        layer_indices = list(range(layer_count))
    else:
        layer_indices = sorted(set(int(value) for value in layers.split(",")))
        if not layer_indices or min(layer_indices) < 0 or max(layer_indices) >= layer_count:
            raise ValueError(f"Layer indices must be in [0, {layer_count-1}]")
    folds = int(config["modeling"].get("cv_folds", 5))
    seed = int(config["seed"])
    alpha_grid = [float(value) for value in config["modeling"].get("embedding_ridge_alphas", [.1, 1, 10, 100, 1000, 10000])]
    bootstrap_replicates = int(bootstrap_replicates or config.get("ssl_layer_sweep", {}).get("bootstrap_replicates", config["modeling"].get("bootstrap_replicates", 1000)))
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be >= 1")

    experiments, predictions, weights_rows = [], {}, []
    n_total = len(ENCODERS) * len(TARGETS) * len(POOLINGS) * len(layer_indices)
    completed = 0
    for encoder in ENCODERS:
        for target in TARGETS:
            train = data["train"].reset_index(drop=True)
            val = data["val"].reset_index(drop=True)
            y_train_all = pd.to_numeric(train[target], errors="coerce")
            y_val_all = pd.to_numeric(val[target], errors="coerce")
            train_mask, val_mask = y_train_all.notna().to_numpy(), y_val_all.notna().to_numpy()
            y_train = y_train_all.loc[train_mask].reset_index(drop=True)
            y_val = y_val_all.loc[val_mask].reset_index(drop=True)
            groups = train.loc[train_mask, "user_id"].astype(str).to_numpy()
            val_meta = val.loc[val_mask, ["id", "user_id", "question_id"]].copy().reset_index(drop=True)
            current_curve = _load_reusable_mean_std_curve(config, encoder, target, layer_indices, alpha_grid, reuse_current)
            candidate_results = {}
            for pooling in POOLINGS:
                for layer in layer_indices:
                    X_train_array = _features(caches[encoder], "train", layer, pooling)[train_mask]
                    X_val_array = _features(caches[encoder], "val", layer, pooling)[val_mask]
                    if X_train_array.shape[1] != X_val_array.shape[1]:
                        raise AssertionError(f"Train/validation feature dimensions disagree for {encoder}/{target}/{pooling}/layer_{layer}")
                    columns = [f"embedding_{index}" for index in range(X_train_array.shape[1])]
                    X_train = pd.DataFrame(X_train_array, columns=columns)
                    X_val = pd.DataFrame(X_val_array, columns=columns)
                    if X_train.columns.intersection(TARGETS).size or {"user_no", "user_id", "question_id"}.intersection(X_train.columns):
                        raise AssertionError("Target/participant/question metadata reached an SSL feature matrix")
                    cached = current_curve.get(layer) if pooling == "mean+std" else None
                    result = _evaluate_representation(X_train, y_train, X_val, y_val, groups,
                                                      alpha_grid, folds, seed, cached_curve=cached)
                    name = f"layer_{layer}"
                    key = (encoder, target, pooling, layer)
                    candidate_results[key] = result
                    prediction_frame = val_meta.copy()
                    prediction_frame["actual"] = y_val.to_numpy(float)
                    prediction_frame["prediction"] = result["val_pred"]
                    predictions[key] = prediction_frame
                    row = {"encoder": encoder, "target": target, "layer": name, "layer_index": layer,
                           "pooling": pooling, "alpha": result["alpha"], "n_train": int(len(y_train)),
                           "n_validation": int(len(y_val)), "train_participants": int(pd.Series(groups).nunique()),
                           "validation_participants": int(val_meta.user_id.nunique()),
                           "cv_rmse_fold_mean": result["cv_fold_rmse_mean"], "cv_rmse_fold_std": result["cv_fold_rmse_std"],
                           **{f"cv_{metric}": result.get(f"cv_{metric}", np.nan) for metric in ("mae", "rmse", "r2", "pearson", "spearman")},
                           **{f"val_{metric}": result["val_metrics"][metric] for metric in ("mae", "rmse", "r2", "pearson", "spearman")},
                           "cv_cache_reused": bool(cached is not None), "status": "complete"}
                    experiments.append(row)
                    completed += 1
                    print(f"[{encoder}][{target}][{pooling}] layer {completed}/{n_total} {name} complete; alpha={result['alpha']:g}, grouped-CV RMSE={result['cv_fold_rmse_mean']:.4f}, val RMSE={result['val_metrics']['rmse']:.4f}", flush=True)

            mean_std_candidates = [candidate_results[(encoder, target, "mean+std", layer)] for layer in layer_indices]
            layer_oof = np.column_stack([result["oof"] for result in mean_std_candidates])
            layer_val_pred = np.column_stack([result["val_pred"] for result in mean_std_candidates])
            weights, weight_oof_mse = _learn_simplex_weights(layer_oof, y_train.to_numpy(float))
            val_weighted = layer_val_pred @ weights
            weights_by_layer = {layer: float(weight) for layer, weight in zip(layer_indices, weights)}
            for layer, weight in weights_by_layer.items():
                weights_rows.append({"encoder": encoder, "target": target, "layer": f"layer_{layer}", "layer_index": layer,
                                     "weight": weight, "learned_from": "training-only GroupKFold out-of-fold predictions",
                                     "pooling": "mean+std", "constraint": "weights >= 0 and sum to 1", "training_oof_mse": weight_oof_mse})
            weighted_frame = val_meta.copy()
            weighted_frame["actual"] = y_val.to_numpy(float)
            weighted_frame["prediction"] = val_weighted
            predictions[(encoder, target, "weighted_layers", -1)] = weighted_frame

            mean_std_rows = [row for row in experiments if row["encoder"] == encoder and row["target"] == target and row["pooling"] == "mean+std"]
            best_current = min(mean_std_rows, key=lambda row: row["cv_rmse_fold_mean"])
            best_individual = min((row for row in experiments if row["encoder"] == encoder and row["target"] == target), key=lambda row: row["cv_rmse_fold_mean"])
            selections = {
                "current_mean_std": best_current,
                "best_individual": best_individual,
            }
            for method, candidate in selections.items():
                candidate_layer = int(candidate["layer_index"])
                pool = candidate["pooling"]
                predictions[(encoder, target, method, -1)] = predictions[(encoder, target, pool, candidate_layer)]
            weighted_result = {"encoder": encoder, "target": target, "layer": "weighted_all_layers",
                               "layer_index": "all", "pooling": "mean+std", "alpha": "per-layer CV-selected",
                               "cv_rmse_fold_mean": np.nan, "cv_rmse_fold_std": np.nan,
                               "cv_mae": np.nan, "cv_rmse": np.nan, "cv_r2": np.nan, "cv_pearson": np.nan, "cv_spearman": np.nan,
                               **{f"val_{metric}": value for metric, value in metrics(y_val, val_weighted).items() if metric in ("mae", "rmse", "r2", "pearson", "spearman")},
                               "n_train": int(len(y_train)), "n_validation": int(len(y_val)), "train_participants": int(pd.Series(groups).nunique()),
                               "validation_participants": int(val_meta.user_id.nunique()), "cv_cache_reused": False,
                               "status": "complete", "cv_note": "No weighted CV estimate; weights are fitted on full-train OOF predictions."}
            experiments.append(weighted_result)

    experiment_frame = pd.DataFrame(experiments)
    experiment_frame.to_csv(result_dir / "layer_experiments.csv", index=False)
    pd.DataFrame(weights_rows).to_csv(result_dir / "weighted_layer_weights.csv", index=False)
    comparison_rows = []
    selected = {}
    for encoder in ENCODERS:
        for target in TARGETS:
            for method in COMPARISON_METHODS:
                key = (encoder, target, method, -1)
                frame = predictions[key]
                actual = frame.actual.to_numpy(float)
                pred = frame.prediction.to_numpy(float)
                point = metrics(actual, pred)
                ci = bootstrap_by_participant(actual, pred, frame.user_id.astype(str).to_numpy(),
                                              bootstrap_replicates, int(config["modeling"].get("bootstrap_seed", seed + 1)))
                if method == "weighted_layers":
                    selected_row = {"layer": "weighted_all_layers", "layer_index": "all", "pooling": "mean+std", "alpha": "per-layer CV-selected", "cv_rmse_fold_mean": np.nan}
                else:
                    # Recover the exact CV-selected candidate metadata, with no validation-based selection.
                    if method == "current_mean_std":
                        selected_row = min((row for row in experiments if row["encoder"] == encoder and row["target"] == target and row["pooling"] == "mean+std" and str(row["layer_index"]).isdigit()), key=lambda row: row["cv_rmse_fold_mean"])
                    else:
                        selected_row = min((row for row in experiments if row["encoder"] == encoder and row["target"] == target and row.get("pooling") in POOLINGS and row.get("layer_index") != "all"), key=lambda row: row["cv_rmse_fold_mean"])
                record = {"encoder": encoder, "target": target, "method": method,
                          "layer": selected_row["layer"], "layer_index": selected_row["layer_index"],
                          "pooling": selected_row["pooling"], "alpha": selected_row["alpha"],
                          "train_cv_rmse_fold_mean": selected_row.get("cv_rmse_fold_mean", np.nan),
                          **{metric: point[metric] for metric in ("mae", "rmse", "r2", "pearson", "spearman")},
                          **{f"{metric}_{bound}": value for metric, bounds in ci.items() for bound, value in bounds.items()},
                          "weights_train_only": method == "weighted_layers"}
                comparison_rows.append(record)
                selected[f"{encoder}/{target}/{method}"] = {key: record[key] for key in ("layer", "layer_index", "pooling", "alpha", "train_cv_rmse_fold_mean", "mae", "rmse", "r2", "pearson", "spearman")}

    paired_rows = []
    for target in TARGETS:
        for method in COMPARISON_METHODS:
            a = predictions[("wavlm", target, method, -1)]
            b = predictions[("hubert", target, method, -1)]
            ci = _paired_difference_ci(a, b, "rmse", bootstrap_replicates, seed + 101)
            paired_rows.append({"target": target, "contrast": f"WavLM_minus_HuBERT_{method}", "metric": "RMSE", **ci,
                                "interpretation": "Negative favors WavLM; positive favors HuBERT; CI crossing zero is inconclusive."})
        for encoder in ENCODERS:
            current = predictions[(encoder, target, "current_mean_std", -1)]
            for method in ("best_individual", "weighted_layers"):
                candidate = predictions[(encoder, target, method, -1)]
                ci = _paired_difference_ci(candidate, current, "rmse", bootstrap_replicates, seed + 211)
                paired_rows.append({"target": target, "contrast": f"{encoder}_{method}_minus_current_mean_std", "metric": "RMSE", **ci,
                                    "interpretation": "Negative favors candidate; positive favors current mean+std; CI crossing zero is inconclusive."})
    comparison_frame = pd.DataFrame(comparison_rows)
    comparison_frame.to_csv(result_dir / "comparison.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(result_dir / "paired_rmse_differences.csv", index=False)

    plots_dir = result_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for encoder in ENCODERS:
        for target in TARGETS:
            _line_plot(experiments, encoder, target, "r2", plots_dir / f"{encoder}_{target}_val_r2.svg")
            _line_plot(experiments, encoder, target, "rmse", plots_dir / f"{encoder}_{target}_val_rmse.svg")
    experiment_config = {
        "task": "diagnostic SSL hidden-layer/pooling sweep",
        "branch": "diagnostic-only; no extraction/model changes beyond cache fields and this runner",
        "layers_requested": layers, "evaluated_layer_indices": layer_indices,
        "layer_index_convention": "0=feature-encoder output; 1..N=Transformer block outputs",
        "pooling_methods": list(POOLINGS), "encoders": {encoder: config["extractors"][encoder]["model_name"] for encoder in ENCODERS},
        "targets": list(TARGETS), "split_method": "existing fixed participant-disjoint train/validation split; GroupKFold(user_id) for alpha selection",
        "cv_folds": folds, "ridge_solver": "lsqr", "ridge_alpha_grid": alpha_grid,
        "selection_metric": "minimum mean fold RMSE from GroupKFold on train; validation not used for selection",
        "weighted_layer_method": "nonnegative simplex weights minimizing MSE of layer-specific train-only GroupKFold OOF predictions; mean+std representations",
        "weighted_layer_validation": "weights fitted from training OOF only; validation used only for final metrics/bootstrap",
        "bootstrap_unit": "participant/user_id; validation only",
        "bootstrap_replicates": bootstrap_replicates,
        "reuse_existing_mean_std_sweep_csvs": bool(reuse_current),
        "cache_preprocessing": {encoder: caches[encoder]["train"]["metadata"] for encoder in ENCODERS},
        "existing_embedding_caches_reused": {encoder: {split: caches[encoder][split]["source_path"] for split in ("train", "val")} for encoder in ENCODERS},
        "test_used_for_model_fitting_or_scoring": False,
    }
    (result_dir / "experiment_config.json").write_text(json.dumps(experiment_config, indent=2, default=str), encoding="utf-8")
    (result_dir / "selected_layers.json").write_text(json.dumps(selected, indent=2, default=lambda value: None if pd.isna(value) else float(value)), encoding="utf-8")
    _write_report(result_dir, experiment_config, experiment_frame, comparison_frame, pd.DataFrame(paired_rows), weights_rows)
    print(f"Saved all-layer experiment results under {result_dir}", flush=True)
    print("\nSSL LAYER COMPARISON", flush=True)
    print(comparison_frame[["encoder", "target", "method", "layer", "pooling", "mae", "rmse", "r2", "pearson", "spearman"]].to_string(index=False, float_format=lambda value: f"{value:.4f}"), flush=True)
    return result_dir


def _write_report(result_dir, config, experiment_frame, comparison_frame, paired_frame, weights_rows):
    lines = ["# WavLM/HuBERT layer and pooling diagnostic", "",
             "All hidden-state outputs available in the existing cache were evaluated. Ridge alpha was selected from training-only GroupKFold folds grouped by participant, then the chosen configuration was evaluated on the fixed participant-disjoint validation split. Test rows were not used for model fitting or scoring.", "",
             "Layer 0 is the feature-encoder output; layers 1–12 are Transformer block outputs. For each layer, mean, population standard deviation, and concatenated mean+std pooling were evaluated separately. Mean/std are over time frames after the existing chunk-overlap trimming and configured speech-pooling policy.", "",
             "The weighted model is a nonnegative, sum-to-one blend of mean+std layer predictions. Its weights are learned by minimizing error against the training labels using only training participant-grouped OOF predictions. No validation information enters the weights.", "",
             "## Comparison", "", "See `comparison.csv` for MAE, RMSE, R², Pearson and Spearman; participant-cluster bootstrap intervals are included for R², Pearson and Spearman. `paired_rmse_differences.csv` reports participant-paired 95% bootstrap intervals: intervals crossing zero are inconclusive. Small point-estimate differences are not called wins.", "",
             "| Encoder | Target | Method | Chosen layer | Pooling | MAE | RMSE | R² | Pearson | Spearman |", "|---|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in comparison_frame.to_dict("records"):
        lines.append(f"| {row['encoder']} | {row['target']} | {row['method']} | {row['layer']} | {row['pooling']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | {row['pearson']:.4f} | {row['spearman']:.4f} |")
    lines += ["", "## Layer and pooling curves", "", "`layer_experiments.csv` contains every evaluated layer × pooling × target × encoder, including grouped-CV fold metrics, validation metrics, selected alpha and whether the prior mean+std CV score was reused. Plots show validation R²/RMSE by layer for each pooling method.", "",
              "## Weighted aggregation", "", "`weighted_layer_weights.csv` lists target-specific weights learned only from training OOF predictions. Weights are constrained to be nonnegative and sum to one. Weighted rows do not report a non-nested CV estimate because fitting the blend weights on full-training OOF labels would make such an estimate optimistic; their held-out validation metrics remain independent of that fitting.", "",
              "## Reproducibility and cache", "", "`experiment_config.json` records model checkpoints, preprocessing, pooling definitions, layer indices, alpha grid, split/CV settings and source cache paths. All-layer means and standard deviations are persisted under the existing ignored audio cache tree with a preprocessing fingerprint. The experiment reused the existing per-split Transformer embeddings; no WavLM/HuBERT forward pass was required.", "",
              "## Diagnostic interpretation", "", "Use paired participant-bootstrap intervals to judge whether layer selection, weighted aggregation or encoder differences are distinguishable from zero. Cross-target consistency matters; a small improvement on one target without a corresponding interval-supported pattern is inconclusive. No model winner is asserted in this report.", ""]
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--layers", default=None, help="all (default) or comma-separated hidden-state indices, e.g. 0,4,8,12")
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    parser.add_argument("--no-reuse-current-sweeps", action="store_true", help="recompute mean+std CV rather than reuse existing benchmark curves")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    options = config.get("ssl_layer_sweep", {})
    layer_setting = args.layers or options.get("layers", "all")
    run(config, layer_setting, args.bootstrap_replicates,
        reuse_current=bool(options.get("reuse_current_mean_std_sweeps", True) and not args.no_reuse_current_sweeps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
