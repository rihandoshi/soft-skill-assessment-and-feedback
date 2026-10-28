"""Repeated nested participant-grouped CV for the current audio feature branches.

Runs Step A from README_next_steps_audio.md. The fixed test split is checked
for participant overlap by reading its participant list only.
Existing handcrafted, eGeMAPS, WavLM and HuBERT caches are reused.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_audio_models import (  # noqa: E402
    EXCLUDED,
    _assert_no_target_columns,
    _target_feature_names,
    atomic_csv,
    atomic_json,
    build_pipeline,
    load_config,
    load_embeddings,
    metrics,
    project_path,
    search_model,
)


OUT_DIR = ROOT / "results" / "repeated_cv_v1"
TARGETS = ("confidence_score", "speaking_skills", "overall_performance")
DEFAULT_FEATURES = ("A_original", "C_original_plus_new", "eGeMAPS", "WavLM", "HuBERT")
DEFAULT_MODELS = ("Ridge", "SVR", "ExtraTrees")


def load_trainval_and_test_users(config: dict) -> tuple[dict[str, pd.DataFrame], set[str]]:
    """Read train/validation labels and only the test participant list."""
    split_dir = ROOT / config["paths"]["split_dir"]
    audio_dir = ROOT / config["paths"]["audio_dir"]
    targets = list(config["targets"].values())
    result = {}
    users_by_split = {}
    for split in ("train", "val"):
        split_csv = split_dir / f"{split}.csv"
        users_path = split_dir / f"{split}_users.txt"
        manifest_path = audio_dir / f"{split}_audio_metadata.csv"
        for path in (split_csv, users_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required fixed {split} input missing: {path}")
        labels = pd.read_csv(split_csv, dtype={"id": str, "user_no": str, "question_id": str})
        users_file = {line.strip() for line in users_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
        users_csv = set(labels.user_no.astype(str))
        if users_file != users_csv:
            raise ValueError(f"{split} participant list does not match its fixed split CSV")
        if labels.id.duplicated().any():
            raise ValueError(f"Duplicate clip IDs in {split_csv}")
        manifest = pd.read_csv(manifest_path, dtype={"id": str, "user_id": str, "user_no": str, "question_id": str})
        if manifest.id.duplicated().any():
            raise ValueError(f"Duplicate clip IDs in {manifest_path}")
        if set(manifest.id.astype(str)) != set(labels.id.astype(str)):
            raise ValueError(f"Audio manifest IDs do not match fixed {split} split")
        label_columns = [c for c in ("id", "user_no", "question_id", "question", *targets) if c in labels.columns]
        joined = manifest.merge(labels[label_columns], on="id", how="left", suffixes=("_manifest", "_split"), validate="one_to_one")
        for column in ("user_no", "question_id"):
            left, right = f"{column}_manifest", f"{column}_split"
            if left in joined and right in joined and not joined[left].astype(str).eq(joined[right].astype(str)).all():
                raise ValueError(f"{column} mismatch between fixed split and {split} audio manifest")
            if right in joined:
                joined[column] = joined[right]
                joined.drop(columns=[right], inplace=True)
            elif left in joined:
                joined[column] = joined[left]
        for target in targets:
            split_col = f"{target}_split"
            if split_col in joined:
                joined[target] = joined[split_col]
                joined.drop(columns=[split_col], inplace=True)
        for column in ("user_no_manifest", "question_id_manifest"):
            if column in joined:
                joined.drop(columns=[column], inplace=True)
        joined["user_id"] = joined.user_no.astype(str)
        joined["split"] = split
        result[split] = joined
        users_by_split[split] = users_csv
    if users_by_split["train"] & users_by_split["val"]:
        raise ValueError("Participant overlap found between fixed train and validation splits")
    test_users_path = split_dir / "test_users.txt"
    if not test_users_path.is_file():
        raise FileNotFoundError(f"Fixed test participant list missing: {test_users_path}")
    # No test labels, manifest, extracted features, or embeddings are read.
    test_users = {line.strip() for line in test_users_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    return result, test_users


def write_json(path: Path, value) -> None:
    atomic_json(path, value)


def aligned_feature_table(path: Path, metadata: pd.DataFrame, config: dict, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} feature table not found: {path}")
    table = pd.read_csv(path, dtype={"id": str})
    if "id" not in table:
        raise ValueError(f"{label} table has no id column: {path}")
    table["id"] = table.id.astype(str)
    if table.id.duplicated().any():
        raise ValueError(f"Duplicate clip IDs in {path}")
    target_names = _target_feature_names(config)
    forbidden = EXCLUDED | target_names
    columns = [c for c in table.columns if c not in forbidden
               and str(c).strip().casefold() not in target_names
               and pd.api.types.is_numeric_dtype(table[c])]
    if not columns:
        raise ValueError(f"No numeric predictor columns in {path}")
    indexed = table.set_index("id")
    ids = metadata.id.astype(str).tolist()
    missing = set(ids) - set(indexed.index)
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} required train/validation clips")
    X = indexed.reindex(ids)[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    _assert_no_target_columns(X, config, f"{label} repeated-CV predictors")
    if set(X.columns) & forbidden:
        raise AssertionError(f"Identifier/target leak in {label} feature matrix")
    return X


def load_feature_matrices(config: dict, split_data: dict, targets: tuple[str, ...]) -> dict:
    """Load only train/validation predictors and align every table by clip id."""
    metadata = pd.concat([split_data["train"], split_data["val"]], ignore_index=True)
    if metadata.id.astype(str).duplicated().any():
        raise ValueError("A clip occurs in both the fixed train and validation split")
    legacy_dir = project_path(config["paths"]["legacy_features_dir"])
    original_names_path = ROOT / "results" / "handcrafted_improvements" / "feature_dimensions.csv"
    dims = pd.read_csv(original_names_path)
    names_row = dims.loc[dims.feature_set.eq("A_original")]
    if names_row.empty:
        raise ValueError(f"Cannot find A_original feature schema in {original_names_path}")
    original_columns = json.loads(names_row.iloc[0].feature_names)

    matrices: dict[str, dict[str, pd.DataFrame]] = {}
    original = {}
    combined = {}
    egemaps = {}
    for split in ("train", "val"):
        meta = split_data[split]
        original[split] = aligned_feature_table(legacy_dir / f"{split}_features.csv", meta, config, f"original handcrafted/{split}")
        absent = set(original_columns) - set(original[split].columns)
        if absent:
            raise ValueError(f"Original handcrafted table is missing baseline columns: {sorted(absent)}")
        original[split] = original[split][original_columns]
        new_path = ROOT / "results" / "handcrafted_improvements" / "features" / f"{split}_new_features.csv"
        new = aligned_feature_table(new_path, meta, config, f"new handcrafted/{split}")
        if original[split].index.equals(new.index) is False or len(original[split]) != len(new):
            raise AssertionError("Handcrafted feature rows are not aligned")
        combined[split] = pd.concat([original[split], new], axis=1)
        egemaps_path = project_path(config["paths"]["features_dir"]) / "egemaps" / f"{split}_features.csv"
        egemaps[split] = aligned_feature_table(egemaps_path, meta, config, f"eGeMAPS/{split}")
        if combined[split].columns.duplicated().any():
            raise ValueError("Duplicate columns in combined handcrafted predictors")

    matrices["A_original"] = original
    matrices["C_original_plus_new"] = combined
    matrices["eGeMAPS"] = egemaps

    # The embedding NPZs already contain per-layer mean+std vectors. Select the
    # layer from the prior training-only sweep; no encoder inference is run here.
    for encoder in ("wavlm", "hubert"):
        pooled = load_embeddings(encoder, split_data, config, splits=("train", "val"))
        if pooled is None:
            raise FileNotFoundError(f"Cached {encoder} train/validation embeddings are missing")
        matrices["WavLM" if encoder == "wavlm" else "HuBERT"] = {}
        selected_path = ROOT / "results" / "ssl_layer_sweep" / "selected_layers.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        for target in targets:
            key = f"{encoder}/{target}/current_mean_std"
            if key not in selected:
                raise KeyError(f"No training-selected SSL layer recorded for {key}")
            layer_index = int(selected[key]["layer_index"])
            for split in ("train", "val"):
                values = pooled[split]
                if values.ndim != 3 or layer_index >= values.shape[1]:
                    raise ValueError(f"Unexpected {encoder} embedding shape or layer index {layer_index}")
                matrix = pd.DataFrame(values[:, layer_index, :],
                                      columns=[f"{encoder}_layer_{layer_index}_{i}" for i in range(values.shape[2])])
                _assert_no_target_columns(matrix, config, f"{encoder}/{target}/{split} predictors")
                matrices["WavLM" if encoder == "wavlm" else "HuBERT"].setdefault(split, {})[target] = matrix
    return matrices


def make_group_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, list[str], list[str]]]:
    """GroupKFold-style balanced folds, with participant order reshuffled per repeat."""
    from collections import defaultdict
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if len(unique) < n_splits:
        raise ValueError(f"Need at least {n_splits} participants, found {len(unique)}")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    tie_order = {user: i for i, user in enumerate(shuffled)}
    counts = pd.Series(groups).value_counts().to_dict()
    ordered = sorted(unique, key=lambda user: (-counts[user], tie_order[user]))
    fold_users: list[list[str]] = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for user in ordered:
        smallest = min(fold_sizes)
        candidates = [i for i, size in enumerate(fold_sizes) if size == smallest]
        fold = int(rng.choice(candidates))
        fold_users[fold].append(str(user))
        fold_sizes[fold] += int(counts[user])
    all_users = set(groups)
    result = []
    validation_seen: set[str] = set()
    for valid_users in fold_users:
        valid_set = set(valid_users)
        train_users = all_users - valid_set
        tr = np.flatnonzero(np.isin(groups, list(train_users)))
        va = np.flatnonzero(np.isin(groups, list(valid_set)))
        if set(groups[tr]) & set(groups[va]):
            raise AssertionError("Participant appears in both sides of an outer fold")
        if validation_seen & valid_set:
            raise AssertionError("Participant assigned to validation more than once in a repeat")
        validation_seen.update(valid_set)
        result.append((tr, va, sorted(train_users), sorted(valid_set)))
    if validation_seen != all_users:
        raise AssertionError("A participant is missing from validation folds")
    return result


def score_metrics(y, pred) -> dict:
    values = metrics(y, pred)
    try:
        from scipy.stats import kendalltau
        tau = kendalltau(np.asarray(y, float), np.asarray(pred, float), variant="b").statistic
        values["c_index"] = float((tau + 1.0) / 2.0) if np.isfinite(tau) else float("nan")
    except Exception:
        values["c_index"] = float("nan")
    return values


def participant_bootstrap(frame: pd.DataFrame, reps: int, seed: int, paired: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    users = frame.user_no.astype(str).unique()
    if len(users) < 2:
        return {}
    by_user = {u: part for u, part in frame.groupby(frame.user_no.astype(str), sort=False)}
    metric_names = ("rmse_delta", "pearson_delta") if paired else ("mae", "rmse", "r2", "pearson", "spearman", "c_index")
    draws = {name: [] for name in metric_names}
    for _ in range(reps):
        sampled = rng.choice(users, size=len(users), replace=True)
        boot = pd.concat([by_user[user] for user in sampled], ignore_index=True)
        if paired:
            a = score_metrics(boot.actual, boot.prediction)
            b = score_metrics(boot.actual, boot.reference_prediction)
            draws["rmse_delta"].append(a["rmse"] - b["rmse"])
            draws["pearson_delta"].append(a["pearson"] - b["pearson"])
        else:
            scores = score_metrics(boot.actual, boot.prediction)
            for name in metric_names:
                if np.isfinite(scores[name]):
                    draws[name].append(scores[name])
    return {name: (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))
            for name, values in draws.items() if values}


def stable_seed(*parts) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def atomic_checkpoint(result_dir: Path, status: pd.DataFrame, completed: set[str], signature: str) -> None:
    atomic_csv(status, result_dir / "fold_status.csv")
    atomic_json(result_dir / "checkpoint.json", {"signature": signature, "completed_tasks": sorted(completed)})


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    columns = list(frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt) if np.isfinite(value) else "")
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(str(columns[i])), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    header = "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns)) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, rule, *body])


def summarize(oof: pd.DataFrame, status: pd.DataFrame, result_dir: Path, bootstrap_reps: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = ("mae", "rmse", "r2", "pearson", "spearman", "c_index")
    result_rows = []
    if not oof.empty:
        for (feature, model, target), frame in oof.groupby(["feature_set", "model", "target"], sort=True):
            repeat_scores = []
            for repeat, part in frame.groupby("repeat", sort=True):
                scores = score_metrics(part.actual, part.prediction)
                repeat_scores.append({"repeat": int(repeat), **scores})
            repeat_table = pd.DataFrame(repeat_scores)
            pooled = score_metrics(frame.actual, frame.prediction)
            ci = participant_bootstrap(frame, bootstrap_reps, stable_seed(seed, feature, model, target))
            row = {"feature_set": feature, "model": model, "target": target,
                   "n_oof_rows": len(frame), "n_participants": int(frame.user_no.nunique()),
                   "completed_repeats": int(frame.repeat.nunique()), **{f"pooled_{k}": v for k, v in pooled.items()}}
            for metric in metric_names:
                row[f"mean_repeat_{metric}"] = float(repeat_table[metric].mean())
                row[f"std_repeat_{metric}"] = float(repeat_table[metric].std(ddof=0))
                row[f"{metric}_lower_95"] = ci.get(metric, (np.nan, np.nan))[0]
                row[f"{metric}_upper_95"] = ci.get(metric, (np.nan, np.nan))[1]
            task_status = status[(status.feature_set == feature) & (status.model == model) & (status.target == target)]
            row["failed_outer_folds"] = int(task_status.status.eq("failed").sum())
            result_rows.append(row)
    results = pd.DataFrame(result_rows)
    if not results.empty:
        atomic_csv(results, result_dir / "repeated_cv_metrics.csv")

    paired_rows = []
    if not oof.empty:
        reference = "A_original"
        for (target, model), base in oof[oof.feature_set.eq(reference)].groupby(["target", "model"]):
            key_cols = ["target", "model", "repeat", "fold", "id", "user_no"]
            base = base[key_cols + ["actual", "prediction"]].rename(columns={"prediction": "reference_prediction", "actual": "actual_reference"})
            for feature in sorted(set(oof.feature_set) - {reference}):
                candidate = oof[(oof.target == target) & (oof.model == model) & (oof.feature_set == feature)]
                cand = candidate[key_cols + ["actual", "prediction"]]
                joined = cand.merge(base, on=key_cols, how="inner", validate="one_to_one")
                if joined.empty:
                    continue
                if not np.allclose(joined.actual, joined.actual_reference, equal_nan=True):
                    raise AssertionError("Paired predictions have different target values")
                paired = joined.rename(columns={"actual": "actual"})
                cand_score = score_metrics(paired.actual, paired.prediction)
                base_score = score_metrics(paired.actual, paired.reference_prediction)
                intervals = participant_bootstrap(paired, bootstrap_reps,
                    stable_seed(seed, feature, model, target, "paired"), paired=True)
                rmse_ci = intervals.get("rmse_delta", (np.nan, np.nan))
                pearson_ci = intervals.get("pearson_delta", (np.nan, np.nan))
                paired_rows.append({"target": target, "model": model, "feature_set": feature,
                    "reference": reference, "n_participants": int(paired.user_no.nunique()),
                    "rmse_delta": cand_score["rmse"] - base_score["rmse"],
                    "rmse_delta_lower_95": rmse_ci[0], "rmse_delta_upper_95": rmse_ci[1],
                    "pearson_delta": cand_score["pearson"] - base_score["pearson"],
                    "pearson_delta_lower_95": pearson_ci[0], "pearson_delta_upper_95": pearson_ci[1],
                    "paired_rmse_reference": base_score["rmse"], "paired_rmse_candidate": cand_score["rmse"]})
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        atomic_csv(paired, result_dir / "paired_bootstrap_differences.csv")
    return results, paired


def write_report(result_dir: Path, config: dict, results: pd.DataFrame, paired: pd.DataFrame,
                 status: pd.DataFrame, repeats: int, folds: int, inner_folds: int, bootstrap_reps: int) -> None:
    lines = ["# Repeated participant-grouped CV (Step A)", "",
        f"Protocol: {repeats} repeats × {folds} outer folds, with {inner_folds}-fold inner GroupKFold hyperparameter selection. Folds are grouped by `user_no`; every participant is validation exactly once per repeat. Train and validation participants are combined. The fixed test split is excluded from features, fitting, fold generation and scoring.", "",
        "All preprocessing is fitted within each inner/outer training fold. SSL embeddings and handcrafted/eGeMAPS tables are loaded from existing caches; no feature extraction or test scoring is performed. The fixed test split is consulted only to verify participant disjointness. Metrics include pooled OOF predictions over all repeats and the mean/std of per-repeat metrics. Confidence intervals resample whole participants.", "",
        "C-index is reported as `(Kendall tau-b + 1) / 2` to handle tied ordinal labels; it is a rank-concordance summary, not a survival-analysis estimate.", "",
        f"Participant bootstrap replicates: {bootstrap_reps}. Models: Ridge, SVR, ExtraTrees. Ridge uses the stable LSQR solver. Inner hyperparameter selection minimizes mean grouped-fold RMSE.", "",
        "## Pooled OOF metrics", ""]
    if not results.empty:
        columns = ["feature_set", "model", "target", "completed_repeats", "pooled_mae", "pooled_rmse", "pooled_r2", "pooled_pearson", "pooled_spearman", "pooled_c_index", "pearson_lower_95", "pearson_upper_95", "rmse_lower_95", "rmse_upper_95"]
        lines.extend([markdown_table(results[columns]), ""])
    lines += ["## Paired comparison against original handcrafted A", "",
        "Deltas are candidate minus A_original; negative RMSE or positive Pearson favors the candidate. The pre-registered practical thresholds are an RMSE reduction of at least 0.01 or Pearson increase of at least 0.015, with the paired participant-bootstrap interval excluding zero, on at least two targets. These comparisons are not corrected for multiple testing.", ""]
    if not paired.empty:
        lines.append(markdown_table(paired))
        lines.append("")
        for (feature, model), group in paired.groupby(["feature_set", "model"]):
            rmse_wins = ((group.rmse_delta_upper_95 < 0) & (group.rmse_delta <= -0.01)).sum()
            pearson_wins = ((group.pearson_delta_lower_95 > 0) & (group.pearson_delta >= 0.015)).sum()
            targets = sorted(set(group.loc[((group.rmse_delta_upper_95 < 0) & (group.rmse_delta <= -0.01)) |
                ((group.pearson_delta_lower_95 > 0) & (group.pearson_delta >= 0.015)), "target"]))
            lines.append(f"- {feature} / {model}: {len(targets)}/3 targets meet at least one threshold ({', '.join(targets) if targets else 'none'}; RMSE threshold hits={rmse_wins}, Pearson threshold hits={pearson_wins}).")
    else:
        lines.append("No paired comparison table is available yet.")
    failed = status[status.status.eq("failed")] if not status.empty else pd.DataFrame()
    lines += ["", "## Run status", "", f"Outer tasks completed: {len(status)}; failed: {len(failed)}."]
    if not failed.empty:
        lines += ["", "| Feature set | Model | Target | Repeat | Fold | Error |", "|---|---|---|---:|---:|---|"]
        for row in failed.itertuples(index=False):
            lines.append(f"| {row.feature_set} | {row.model} | {row.target} | {row.repeat} | {row.fold} | {str(row.error).replace('|', '/')} |")
    lines += ["", "## Files", "", "- `cv_splits.json`: participant assignments for each outer repeat/fold.", "- `oof_predictions.csv`: per-clip outer-fold predictions and selected hyperparameters.", "- `repeated_cv_metrics.csv`: pooled metrics, repeat summaries, and bootstrap intervals.", "- `paired_bootstrap_differences.csv`: candidate-vs-original participant-paired deltas.", "- `fold_status.csv` and `checkpoint.json`: run progress and resume state.", ""]
    (result_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if min(args.repeats, args.folds, args.inner_folds, args.bootstrap, args.n_jobs, args.trials) < 1:
        parser.error("repeats, folds, inner-folds, bootstrap, n-jobs and trials must all be positive")

    config = load_config(args.config)
    config["modeling"]["tree_trials"] = args.trials
    data, test_users = load_trainval_and_test_users(config)
    trainval = pd.concat([data["train"], data["val"]], ignore_index=True)
    trainval_users = set(trainval.user_no.astype(str))
    overlap = trainval_users & test_users
    if overlap:
        raise AssertionError(f"Fixed test participants overlap train/validation: {sorted(overlap)[:10]}")
    if trainval.id.astype(str).duplicated().any():
        raise AssertionError("Duplicate clip IDs in combined train/validation data")

    feature_names, model_names = DEFAULT_FEATURES, DEFAULT_MODELS
    selected_path = ROOT / "results" / "ssl_layer_sweep" / "selected_layers.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    ssl_layers = {encoder: {target: int(selected[f"{encoder}/{target}/current_mean_std"]["layer_index"])
                           for target in TARGETS} for encoder in ("wavlm", "hubert")}
    signature_config = {"config_path": config["_config_path"], "targets": list(TARGETS),
        "feature_sets": list(feature_names), "models": list(model_names), "repeats": args.repeats,
        "outer_folds": args.folds, "inner_folds": args.inner_folds, "bootstrap": args.bootstrap,
        "trials": args.trials, "seed": args.seed, "participant_column": "user_no",
        "ssl_layer_indices": ssl_layers, "test_used_for_scoring_or_features": False}
    signature = hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUT_DIR / "checkpoint.json"
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise RuntimeError(f"Existing checkpoint in {OUT_DIR} uses different settings. Move/archive that result folder before starting a different run.")
        completed = set(previous.get("completed_tasks", []))
    else:
        completed = set()
    matrices = load_feature_matrices(config, data, TARGETS)
    dimensions = {}
    for name in feature_names:
        dimensions[name] = {}
        for split in ("train", "val"):
            value = matrices[name][split]
            dimensions[name][split] = ({target: frame.shape[1] for target, frame in value.items()}
                                       if isinstance(value, dict) else value.shape[1])
    write_json(OUT_DIR / "configuration.json", {**signature_config, "run_signature": signature,
        "n_train_validation_clips": len(trainval), "n_train_validation_participants": len(trainval_users),
        "n_test_participants_checked_only": len(test_users), "feature_dimensions": dimensions,
        "selection": "nested GroupKFold by user_no, mean inner-fold RMSE", "ridge_solver": "lsqr",
        "cv_scheme": "repeated participant-grouped balanced GroupKFold; test split excluded"})
    # Store the same outer assignments for every target and feature family.
    splits_payload = {"group_column": "user_no", "repeats": args.repeats, "folds": args.folds,
        "seed": args.seed, "test_participants_included": False, "assignments": []}
    base_groups = trainval.user_no.astype(str).to_numpy()
    fold_map = {}
    for repeat in range(args.repeats):
        folds = make_group_folds(base_groups, args.folds, args.seed + repeat * 1009)
        fold_map[repeat] = folds
        for fold_no, (_, _, train_users, validation_users) in enumerate(folds):
            if set(validation_users) & test_users:
                raise AssertionError("Test participant entered a validation fold")
            splits_payload["assignments"].append({"repeat": repeat, "fold": fold_no,
                "train_user_no": train_users, "validation_user_no": validation_users})
    atomic_json(OUT_DIR / "cv_splits.json", splits_payload)

    oof_path, partial_oof_path = OUT_DIR / "oof_predictions.csv", OUT_DIR / "oof_predictions.partial.csv"
    status_path = OUT_DIR / "fold_status.csv"
    if oof_path.exists() and not partial_oof_path.exists() and completed:
        # A completed prior run can be summarized again without refitting.
        partial_oof_path = oof_path
    status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
    if not status.empty:
        status = status.drop_duplicates(["feature_set", "model", "target", "repeat", "fold"], keep="last").reset_index(drop=True)
    all_tasks = len(TARGETS) * len(feature_names) * len(model_names) * args.repeats * args.folds
    task_num = len(completed)
    for target in TARGETS:
        y_all = pd.to_numeric(trainval[target], errors="coerce").to_numpy(float)
        valid_all = np.isfinite(y_all)
        groups_all = trainval.user_no.astype(str).to_numpy()
        for feature in feature_names:
            if feature in ("WavLM", "HuBERT"):
                Xall = pd.concat([matrices[feature]["train"][target], matrices[feature]["val"][target]], ignore_index=True)
            else:
                Xall = pd.concat([matrices[feature]["train"], matrices[feature]["val"]], ignore_index=True)
            _assert_no_target_columns(Xall, config, f"{feature} combined train/validation X for {target}")
            if len(Xall) != len(trainval):
                raise AssertionError(f"{feature} rows do not align with train/validation labels")
            row_valid = valid_all
            X = Xall.loc[row_valid].reset_index(drop=True)
            y = pd.Series(y_all[row_valid], dtype=float)
            groups = groups_all[row_valid]
            row_meta = trainval.loc[row_valid, ["id", "user_no"]].reset_index(drop=True)
            _assert_no_target_columns(X, config, f"{feature}/{target} model matrix")
            for model_name in model_names:
                for repeat in range(args.repeats):
                    # Original fold indices are over all metadata; labels are complete in the fixed set.
                    # If a target has missing values, map original positions to the filtered matrix.
                    index_map = np.full(len(trainval), -1, dtype=int)
                    index_map[np.flatnonzero(row_valid)] = np.arange(int(row_valid.sum()))
                    for fold_no, (raw_tr, raw_va, _, _) in enumerate(fold_map[repeat]):
                        task = f"{feature}|{model_name}|{target}|{repeat}|{fold_no}"
                        if task in completed:
                            continue
                        tr = index_map[raw_tr]
                        va = index_map[raw_va]
                        tr, va = tr[tr >= 0], va[va >= 0]
                        entry = {"feature_set": feature, "model": model_name, "target": target,
                                 "repeat": repeat, "fold": fold_no, "status": "failed", "error": ""}
                        try:
                            if len(tr) == 0 or len(va) == 0:
                                raise ValueError("No labeled rows in outer train/validation fold")
                            inner_groups = groups[tr]
                            if len(np.unique(inner_groups)) < args.inner_folds:
                                raise ValueError(f"Outer training fold has fewer than {args.inner_folds} participants")
                            best, _, _ = search_model(X.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True),
                                inner_groups, model_name, config, args.inner_folds,
                                args.seed + repeat * 1009 + fold_no, standardize=True,
                                n_jobs=args.n_jobs, stable_ridge=(model_name == "Ridge"))
                            if best is None:
                                raise RuntimeError("Every inner-CV hyperparameter candidate failed")
                            params = best["params"]
                            model = build_pipeline(X.iloc[tr].reset_index(drop=True), model_name, params,
                                args.seed + repeat * 1009 + fold_no, standardize=True,
                                n_jobs=args.n_jobs, stable_ridge=(model_name == "Ridge"))
                            model.fit(X.iloc[tr], y.iloc[tr])
                            pred = model.predict(X.iloc[va])
                            fold_scores = score_metrics(y.iloc[va], pred)
                            preds = row_meta.iloc[va].copy()
                            preds["target"] = target
                            preds["feature_set"] = feature
                            preds["model"] = model_name
                            preds["repeat"] = repeat
                            preds["fold"] = fold_no
                            preds["actual"] = y.iloc[va].to_numpy()
                            preds["prediction"] = pred
                            preds["selected_params"] = json.dumps(params, sort_keys=True)
                            preds["inner_cv_rmse"] = best.get("cv_rmse_fold_mean", np.nan)
                            # Append per outer fold so interruptions do not require
                            # repeatedly rewriting a growing, large OOF table.
                            preds.to_csv(partial_oof_path, mode="a", header=not partial_oof_path.exists(), index=False)
                            entry.update(status="ok", error="", **{f"outer_{k}": v for k, v in fold_scores.items()},
                                         selected_params=json.dumps(params, sort_keys=True),
                                         inner_cv_rmse=best.get("cv_rmse_fold_mean", np.nan),
                                         n_train=len(tr), n_validation=len(va))
                        except Exception as exc:
                            entry["error"] = f"{type(exc).__name__}: {exc}"
                            print(f"[{feature}][{target}][{model_name}][repeat {repeat + 1}/{args.repeats} fold {fold_no + 1}/{args.folds}] FAILED: {entry['error']}", flush=True)
                        status = pd.concat([status, pd.DataFrame([entry])], ignore_index=True)
                        completed.add(task)
                        task_num += 1
                        atomic_checkpoint(OUT_DIR, status, completed, signature)
                        if entry["status"] == "ok":
                            print(f"[{feature}][{target}][{model_name}][repeat {repeat + 1}/{args.repeats} fold {fold_no + 1}/{args.folds}] complete: RMSE={entry['outer_rmse']:.4f}; overall {task_num}/{all_tasks}", flush=True)

    oof = pd.read_csv(partial_oof_path, dtype={"id": str, "user_no": str}) if partial_oof_path.exists() else pd.DataFrame()
    if not oof.empty:
        task_keys = ["feature_set", "model", "target", "repeat", "fold", "id"]
        oof = oof.drop_duplicates(task_keys, keep="last").reset_index(drop=True)
        atomic_csv(oof, oof_path)
        if partial_oof_path != oof_path:
            partial_oof_path.unlink(missing_ok=True)
    results, paired = summarize(oof, status, OUT_DIR, args.bootstrap, args.seed)
    write_report(OUT_DIR, config, results, paired, status, args.repeats, args.folds, args.inner_folds, args.bootstrap)
    if not results.empty:
        print("\nREPEATED GROUPED CV SUMMARY (POOLED OOF)", flush=True)
        columns = ["feature_set", "model", "target", "pooled_mae", "pooled_rmse", "pooled_r2", "pooled_pearson", "pooled_spearman", "pooled_c_index"]
        print(results[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)
    failed = int(status.status.eq("failed").sum()) if not status.empty else 0
    print(f"\nResults: {OUT_DIR.relative_to(ROOT)} | failed outer folds: {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
