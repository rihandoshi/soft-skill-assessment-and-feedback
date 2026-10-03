from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc.name}. Install numpy and pandas in this environment.")


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "outputs" / "audio" / "features" / "legacy"
SPLIT_DIR = ROOT / "Datasets" / "Split Dataset"
OUTPUT = ROOT / "outputs" / "audio" / "legacy_run"
TARGETS = {
    "confidence": "confidence_score",
    "speaking_skills": "speaking_skills",
    "overall_performance": "overall_performance",
}
ID_COLS = {"id", "user_no", "user_id", "question_id", "split", "file_name",
           "original_video_path", "audio_path", "status", "feature_status", "feature_error"}
TRANSCRIPT_PREFIX = "transcript_"


def deps():
    try:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for modeling but is not installed. "
                         "Install it in the project environment; no package was installed automatically.") from exc
    return ExtraTreesRegressor, RandomForestRegressor, SimpleImputer, permutation_importance, GroupKFold, make_pipeline


def load_splits(include_transcript_features: bool):
    pieces = {}
    for split in ("train", "val", "test"):
        features_path = FEATURE_DIR / f"{split}_audio_features.csv"
        split_csv = SPLIT_DIR / f"{split}.csv"
        if not features_path.is_file():
            raise FileNotFoundError(f"Acoustic feature table missing: {features_path}. Run Utils/extract_audio_features.py first.")
        if not split_csv.is_file():
            raise FileNotFoundError(f"Fixed split file missing: {split_csv}")
        features = pd.read_csv(features_path, dtype={"id": str, "user_no": str, "user_id": str})
        authoritative = pd.read_csv(split_csv, dtype={"id": str, "user_no": str})
        if "id" not in features or "id" not in authoritative:
            raise ValueError(f"Both {features_path.name} and {split_csv.name} must have id columns")
        if features["id"].duplicated().any():
            raise ValueError(f"Duplicate sample ids in {features_path}")
        # Reconcile identity and labels from the split CSV, not the feature file.
        labels = authoritative[[c for c in ("id", "user_no", *TARGETS.values(), "question_id", "question")
                                if c in authoritative.columns]].copy()
        merged = features.merge(labels, on="id", how="left", validate="one_to_one", suffixes=("", "_split"))
        user_col = "user_no_split" if "user_no_split" in merged else "user_no"
        if merged[user_col].isna().any():
            raise ValueError(f"Rows in {features_path.name} do not map to {split_csv.name}")
        for identity in ("user_no", "question_id"):
            split_identity = f"{identity}_split"
            if identity in merged and split_identity in merged:
                mismatch = merged[identity].astype(str) != merged[split_identity].astype(str)
                if mismatch.any():
                    raise ValueError(f"{mismatch.sum()} {identity} mappings in {features_path.name} disagree with {split_csv.name}")
        if "split" in merged and not merged["split"].fillna(split).astype(str).eq(split).all():
            raise ValueError(f"Feature rows in {features_path.name} include an unexpected split value")
        for col in ("user_no", "question_id", "question", *TARGETS.values()):
            if f"{col}_split" in merged:
                merged[col] = merged[f"{col}_split"]
                merged.drop(columns=f"{col}_split", inplace=True)
        merged["split"] = split
        pieces[split] = merged

    train_users = set(pieces["train"]["user_no"].astype(str))
    val_users = set(pieces["val"]["user_no"].astype(str))
    test_users = set(pieces["test"]["user_no"].astype(str))
    overlaps = (train_users & val_users, train_users & test_users, val_users & test_users)
    if any(overlaps):
        raise ValueError("Participant leakage detected in fixed split files; stopping without resplitting.")

    feature_cols = [c for c in pieces["train"].columns if c not in ID_COLS and c not in TARGETS.values()
                    and not c.endswith("_split") and
                    (include_transcript_features or not c.startswith(TRANSCRIPT_PREFIX))]
    feature_cols = [c for c in feature_cols if all(pd.api.types.is_numeric_dtype(p[c]) for p in pieces.values())]
    if not feature_cols:
        raise ValueError("No numeric acoustic features found. Inspect the feature extraction output first.")
    # Numeric coercion and infinity handling; imputation remains inside each fold/model pipeline.
    for part in pieces.values():
        part[feature_cols] = part[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return pieces, feature_cols


def target_report(train: pd.DataFrame, targets: list[str]) -> list[dict[str, Any]]:
    records = []
    for target in targets:
        y = pd.to_numeric(train[target], errors="coerce").dropna()
        if y.empty:
            raise ValueError(f"No labeled training rows for {target}")
        freq = y.value_counts().head(20)
        records.append({"target": target, "dtype": str(train[target].dtype), "n": int(y.size),
                        "unique_count": int(y.nunique()), "min": float(y.min()), "max": float(y.max()),
                        "mean": float(y.mean()), "std": float(y.std(ddof=1)),
                        "top_value_frequencies": {str(k): int(v) for k, v in freq.items()},
                        "formulation": "regression; target remains continuous and is not binned"})
    return records


def metric_set(actual, pred):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not y.size:
        return {m: math.nan for m in ("mae", "rmse", "r2", "pearson", "spearman")}
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else math.nan
    # Average ranks correctly handle ties without requiring scipy.
    yr = pd.Series(y).rank(method="average").to_numpy()
    pr = pd.Series(p).rank(method="average").to_numpy()
    spear = float(np.corrcoef(yr, pr)[0, 1]) if np.std(yr) and np.std(pr) else math.nan
    return {"mae": float(np.mean(np.abs(y - p))), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)) if np.std(y) else math.nan,
            "pearson": pearson, "spearman": spear}


def candidate_specs(seed: int, n_jobs: int):
    ExtraTreesRegressor, RandomForestRegressor, _, _, _, _ = deps()
    specs = {
        "ExtraTrees": {
            "factory": lambda p: ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **p),
            "sample": lambda rng: {"n_estimators": int(rng.choice([200, 400, 700])),
                                   "max_depth": rng.choice([None, 8, 12, 20, 30]),
                                   "min_samples_split": int(rng.choice([2, 4, 8, 12])),
                                   "min_samples_leaf": int(rng.choice([1, 2, 3, 5])),
                                   "max_features": rng.choice([1.0, 0.7, 0.5, "sqrt"])},
            "optuna": lambda t: {"n_estimators": t.suggest_categorical("n_estimators", [200, 400, 700]),
                                 "max_depth": t.suggest_categorical("max_depth", [None, 8, 12, 20, 30]),
                                 "min_samples_split": t.suggest_int("min_samples_split", 2, 12),
                                 "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 5),
                                 "max_features": t.suggest_categorical("max_features", [1.0, 0.7, 0.5, "sqrt"])},
        },
        "RandomForest": {
            "factory": lambda p: RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **p),
            "sample": lambda rng: {"n_estimators": int(rng.choice([200, 400, 700])),
                                   "max_depth": rng.choice([None, 8, 12, 20, 30]),
                                   "min_samples_split": int(rng.choice([2, 4, 8, 12])),
                                   "min_samples_leaf": int(rng.choice([1, 2, 3, 5])),
                                   "max_features": rng.choice([1.0, 0.7, 0.5, "sqrt"])},
            "optuna": lambda t: {"n_estimators": t.suggest_categorical("n_estimators", [200, 400, 700]),
                                 "max_depth": t.suggest_categorical("max_depth", [None, 8, 12, 20, 30]),
                                 "min_samples_split": t.suggest_int("min_samples_split", 2, 12),
                                 "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 5),
                                 "max_features": t.suggest_categorical("max_features", [1.0, 0.7, 0.5, "sqrt"])},
        },
    }
    try:
        from xgboost import XGBRegressor
        specs["XGBoost"] = {
            "factory": lambda p: XGBRegressor(objective="reg:squarederror", eval_metric="rmse",
                                               tree_method="hist", n_jobs=n_jobs, random_state=seed, **p),
            "sample": lambda r: {"n_estimators": int(r.choice([150, 300, 500])), "max_depth": int(r.choice([2, 3, 4, 6])),
                                 "learning_rate": float(r.choice([0.02, 0.04, 0.07, 0.1])),
                                 "min_child_weight": float(r.choice([1, 3, 5, 8])),
                                 "subsample": float(r.choice([0.7, 0.85, 1.0])), "colsample_bytree": float(r.choice([0.7, 0.85, 1.0])),
                                 "reg_alpha": float(r.choice([0, 0.01, 0.1, 1])), "reg_lambda": float(r.choice([1, 3, 8])),
                                 "gamma": float(r.choice([0, 0.1, 0.3]))},
            "optuna": lambda t: {"n_estimators": t.suggest_int("n_estimators", 120, 600, step=40),
                                 "max_depth": t.suggest_int("max_depth", 2, 7), "learning_rate": t.suggest_float("learning_rate", .015, .15, log=True),
                                 "min_child_weight": t.suggest_float("min_child_weight", 1, 10), "subsample": t.suggest_float("subsample", .65, 1),
                                 "colsample_bytree": t.suggest_float("colsample_bytree", .6, 1), "reg_alpha": t.suggest_float("reg_alpha", 1e-5, 2, log=True),
                                 "reg_lambda": t.suggest_float("reg_lambda", .1, 12, log=True), "gamma": t.suggest_float("gamma", 0, .5)},
        }
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        specs["CatBoost"] = {
            "factory": lambda p: CatBoostRegressor(loss_function="RMSE", verbose=False, allow_writing_files=False,
                                                    thread_count=n_jobs, random_seed=seed, **p),
            "sample": lambda r: {"iterations": int(r.choice([200, 400, 700])), "depth": int(r.choice([3, 4, 5, 6, 7])),
                                 "learning_rate": float(r.choice([.02, .04, .07, .1])), "l2_leaf_reg": float(r.choice([1, 3, 5, 8, 12])),
                                 "random_strength": float(r.choice([0, .2, .5, 1])), "bagging_temperature": float(r.choice([0, .5, 1, 2]))},
            "optuna": lambda t: {"iterations": t.suggest_int("iterations", 150, 700, step=50), "depth": t.suggest_int("depth", 3, 8),
                                 "learning_rate": t.suggest_float("learning_rate", .015, .15, log=True), "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 12),
                                 "random_strength": t.suggest_float("random_strength", 0, 1.5), "bagging_temperature": t.suggest_float("bagging_temperature", 0, 2)},
        }
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        specs["LightGBM"] = {
            "factory": lambda p: LGBMRegressor(objective="regression", verbosity=-1, n_jobs=n_jobs,
                                                random_state=seed, **p),
            "sample": lambda r: {"n_estimators": int(r.choice([150, 300, 500])), "learning_rate": float(r.choice([.02, .04, .07, .1])),
                                 "num_leaves": int(r.choice([7, 15, 23, 31])), "max_depth": int(r.choice([-1, 3, 5, 7])),
                                 "min_child_samples": int(r.choice([5, 10, 15, 20])), "subsample": float(r.choice([.7, .85, 1.0])),
                                 "colsample_bytree": float(r.choice([.7, .85, 1.0])), "reg_alpha": float(r.choice([0, .01, .1, 1])),
                                 "reg_lambda": float(r.choice([1, 3, 8]))},
            "optuna": lambda t: {"n_estimators": t.suggest_int("n_estimators", 120, 600, step=40),
                                 "learning_rate": t.suggest_float("learning_rate", .015, .15, log=True), "num_leaves": t.suggest_int("num_leaves", 5, 31),
                                 "max_depth": t.suggest_categorical("max_depth", [-1, 3, 5, 7]), "min_child_samples": t.suggest_int("min_child_samples", 5, 25),
                                 "subsample": t.suggest_float("subsample", .65, 1), "colsample_bytree": t.suggest_float("colsample_bytree", .6, 1),
                                 "reg_alpha": t.suggest_float("reg_alpha", 1e-5, 2, log=True), "reg_lambda": t.suggest_float("reg_lambda", .1, 12, log=True)},
        }
    except ImportError:
        pass
    return specs


def model_pipeline(spec, params, seed):
    _, _, SimpleImputer, _, _, make_pipeline = deps()
    # Impute inside each CV fold; no global scaling for trees.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), spec["factory"](params))


def run_cv(spec, X, y, groups, folds, trials, seed, model_name):
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=folds)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None
    rng = np.random.default_rng(seed)
    results: list[dict[str, Any]] = []
    cache = {}

    def score(params):
        fold_rmses = []
        all_actual, all_pred = [], []
        for tr, va in splitter.split(X, y, groups):
            pipe = model_pipeline(spec, params, seed)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[va])
            actual = y.iloc[va].to_numpy()
            fold_rmses.append(float(np.sqrt(np.mean((actual - pred) ** 2))))
            all_actual.extend(actual.tolist())
            all_pred.extend(np.asarray(pred).tolist())
        return float(np.mean(fold_rmses)), float(np.std(fold_rmses)), metric_set(all_actual, all_pred)

    if optuna is not None:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        def objective(trial):
            params = spec["optuna"](trial)
            mean, std, metrics = score(params)
            cache[trial.number] = (params, mean, std, metrics)
            results.append({"model": model_name, "trial": trial.number, "cv_rmse": mean,
                            "cv_rmse_std": std, **{f"cv_{k}": v for k, v in metrics.items()},
                            "params": json.dumps(params), "error": ""})
            return mean
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best_trial = study.best_trial
        best_params, best_mean, best_std, best_metrics = cache[best_trial.number]
        optimizer = "Optuna TPE"
    else:
        for i in range(trials):
            params = spec["sample"](rng)
            try:
                mean, std, metrics = score(params)
            except Exception as exc:
                results.append({"model": model_name, "trial": i, "cv_rmse": math.nan,
                                "cv_rmse_std": math.nan, "params": json.dumps(params), "error": str(exc)})
                continue
            cache[i] = (params, mean, std, metrics)
            results.append({"model": model_name, "trial": i, "cv_rmse": mean,
                            "cv_rmse_std": std, **{f"cv_{k}": v for k, v in metrics.items()},
                            "params": json.dumps(params), "error": ""})
        if not cache:
            return None, results, "seeded random search"
        best_i = min(cache, key=lambda i: cache[i][1])
        best_params, best_mean, best_std, best_metrics = cache[best_i]
        optimizer = "seeded random search (Optuna unavailable)"
    return {"params": best_params, "cv_rmse": best_mean, "cv_rmse_std": best_std,
            "cv_metrics": best_metrics, "optimizer": optimizer}, results, optimizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=[*TARGETS, "all"], default="all")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--include-transcript-features", action="store_true",
                        help="Allow transcript-derived features; default is strict waveform-only features")
    args = parser.parse_args()
    if args.trials < 1 or args.cv_folds < 2 or args.n_jobs < 1:
        parser.error("trials and n-jobs must be positive, and cv-folds must be >= 2")
    try:
        ExtraTreesRegressor, RandomForestRegressor, SimpleImputer, permutation_importance, GroupKFold, make_pipeline = deps()
        parts, candidate_features = load_splits(args.include_transcript_features)
    except (SystemExit, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    targets = list(TARGETS.values()) if args.target == "all" else [TARGETS[args.target]]
    train, val, test = (parts[x].copy() for x in ("train", "val", "test"))
    if train["user_no"].nunique() < args.cv_folds:
        print("ERROR: fewer training participants than CV folds", file=sys.stderr)
        return 2
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for folder in ("models", "predictions", "metrics", "feature_importance", "plots", "analysis"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    # Drop target-wise constants and exact duplicate feature columns using TRAIN only.
    numeric = train[candidate_features]
    constants = [c for c in candidate_features if numeric[c].nunique(dropna=True) <= 1]
    candidate_features = [c for c in candidate_features if c not in constants]
    duplicates = []
    keep = []
    for col in candidate_features:
        if any(train[col].equals(train[other]) for other in keep):
            duplicates.append(col)
        else:
            keep.append(col)
    candidate_features = keep
    if not candidate_features:
        print("ERROR: no nonconstant numeric features remain", file=sys.stderr)
        return 2
    specs = candidate_specs(args.seed, args.n_jobs)
    if not specs:
        print("ERROR: no supported model dependencies found; install scikit-learn and at least one optional booster.",
              file=sys.stderr)
        return 2
    print(f"Candidates available: {', '.join(specs)}")
    print(f"Optuna {'available' if __import__('importlib').util.find_spec('optuna') else 'unavailable; seeded random search will be used'}")

    duplicate_feature_rows = int(train.duplicated(subset=candidate_features, keep=False).sum())
    try:
        from importlib.metadata import PackageNotFoundError, version
        package_versions = {}
        for package in ("scikit-learn", "xgboost", "catboost", "lightgbm", "optuna", "joblib"):
            try:
                package_versions[package] = version(package)
            except PackageNotFoundError:
                package_versions[package] = None
    except ImportError:
        package_versions = {}
    config = {"seed": args.seed, "cv_strategy": "GroupKFold by participant/user_no", "cv_folds": args.cv_folds,
              "trials_per_model_and_feature_set": args.trials, "selection_metric": "mean grouped CV RMSE; validation used for candidate/ensemble selection",
              "fixed_test_policy": "test predictions/metrics generated once after all choices; test not used for tuning",
              "training_rows": len(train), "training_participants": int(train.user_no.nunique()),
              "validation_rows": len(val), "validation_participants": int(val.user_no.nunique()),
              "test_rows": len(test), "test_participants": int(test.user_no.nunique()),
              "feature_columns": candidate_features, "constant_features_removed": constants,
              "duplicate_features_removed": duplicates, "exact_duplicate_feature_rows_in_train": duplicate_feature_rows,
              "transcript_features_included": args.include_transcript_features,
              "models_available": list(specs), "versions": {"python": platform.python_version(), "numpy": np.__version__,
              "pandas": pd.__version__, **package_versions}}
    (OUTPUT / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    available_targets = [target for target in targets if target in train]
    absent_targets = [target for target in targets if target not in train]
    (OUTPUT / "metrics" / "target_distributions.json").write_text(
        json.dumps({"available": target_report(train, available_targets), "absent": absent_targets}, indent=2), encoding="utf-8")
    for target in targets:
        if target not in train:
            print(f"Skipping absent target {target}")
            continue
        # Only rows with target labels; feature imputation is pipeline/fold local.
        train_t = train.loc[pd.to_numeric(train[target], errors="coerce").notna()].copy()
        val_t = val.loc[pd.to_numeric(val[target], errors="coerce").notna()].copy()
        test_t = test.loc[pd.to_numeric(test[target], errors="coerce").notna()].copy()
        y = pd.to_numeric(train_t[target]).astype(float)
        groups = train_t.user_no.astype(str).to_numpy()
        if pd.Series(groups).nunique() < args.cv_folds:
            print(f"Skipping {target}: too few training participants for {args.cv_folds} folds")
            continue
        # Train-only ExtraTrees ranking defines the reduced set (top half, floor 5).
        ranker = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                               ExtraTreesRegressor(n_estimators=350, min_samples_leaf=2,
                                                   random_state=args.seed, n_jobs=args.n_jobs))
        ranker.fit(train_t[candidate_features], y)
        importances = ranker[-1].feature_importances_
        ranked = sorted(zip(candidate_features, importances), key=lambda x: x[1], reverse=True)
        reduced = [name for name, _ in ranked[:max(5, math.ceil(len(ranked) / 2))]]
        feature_sets = {"all": candidate_features, "reduced_train_importance": reduced}
        candidate_records = []
        trial_records = []
        fitted_candidates = {}
        for set_name, features in feature_sets.items():
            X = train_t[features].reset_index(drop=True)
            yy = y.reset_index(drop=True)
            gg = groups
            for model_name, spec in specs.items():
                print(f"[{target}] tuning {model_name}/{set_name} ({args.trials} trials)")
                best, trials, _ = run_cv(spec, X, yy, gg, args.cv_folds, args.trials,
                                         args.seed, model_name)
                for record in trials:
                    record.update({"target": target, "feature_set": set_name})
                trial_records.extend(trials)
                if best is None:
                    continue
                pipe = model_pipeline(spec, best["params"], args.seed)
                pipe.fit(X, yy)
                vp = pipe.predict(val_t[features]) if len(val_t) else np.array([])
                vm = metric_set(pd.to_numeric(val_t[target]), vp) if len(val_t) else {}
                record = {"target": target, "model": model_name, "feature_set": set_name,
                          "cv_rmse": best["cv_rmse"], "cv_rmse_std": best["cv_rmse_std"],
                          "cv_metrics": best["cv_metrics"],
                          "validation_metrics": vm, "params": best["params"], "features": features,
                          "optimizer": best["optimizer"]}
                candidate_records.append(record)
                fitted_candidates[(model_name, set_name)] = (pipe, best, features)
        if not candidate_records:
            print(f"No viable candidates for {target}")
            continue
        # Candidate is selected by CV RMSE; validation is reported and used only to
        # choose optional convex blend weights among the CV-best model families.
        candidate_records.sort(key=lambda r: r["cv_rmse"])
        cv_best = candidate_records[0]
        val_records = [{"target": r["target"], "model": r["model"], "feature_set": r["feature_set"],
                        **r["validation_metrics"], "cv_rmse": r["cv_rmse"], "params": json.dumps(r["params"])}
                       for r in candidate_records]
        prediction_rows = []
        model_val_predictions = {}
        best_id = (cv_best["model"], cv_best["feature_set"])
        best_pipe, best_info, best_features = fitted_candidates[best_id]
        for idx, (_, row) in enumerate(val_t.iterrows()):
            prediction_rows.append({"id": row.id, "user_no": row.user_no, "question_id": row.get("question_id", ""),
                                    "target": target, "actual": row[target], "prediction": best_pipe.predict(val_t.iloc[[idx]][best_features])[0],
                                    "selected_model": cv_best["model"], "feature_set": cv_best["feature_set"]})
        for key, (pipe, _, features) in fitted_candidates.items():
            if len(val_t):
                model_val_predictions[key] = pipe.predict(val_t[features])
        # Convex equal-grid blend search uses validation only and is retained only
        # when it improves RMSE over the CV-best individual candidate.
        blend = None
        if len(model_val_predictions) >= 2 and len(val_t):
            shortlist = sorted(candidate_records, key=lambda r: (r["cv_rmse"], r["validation_metrics"].get("rmse", math.inf)))[:4]
            keys = [(r["model"], r["feature_set"]) for r in shortlist]
            matrix = np.column_stack([model_val_predictions[k] for k in keys])
            actual = pd.to_numeric(val_t[target]).to_numpy()
            individual_rmse = metric_set(actual, model_val_predictions[best_id])["rmse"]
            best_mix = None
            # Coarse 0.1-grid weights, normalized to sum to one.
            for raw in np.ndindex(*([11] * len(keys))):
                weights = np.asarray(raw, dtype=float) / 10
                if weights.sum() == 0:
                    continue
                weights /= weights.sum()
                pred = matrix @ weights
                m = metric_set(actual, pred)
                if best_mix is None or m["rmse"] < best_mix["metrics"]["rmse"]:
                    best_mix = {"weights": weights.tolist(), "keys": keys, "metrics": m}
            if best_mix and best_mix["metrics"]["rmse"] < individual_rmse:
                blend = best_mix
        selected = cv_best
        if blend:
            blend_val = np.column_stack([model_val_predictions[k] for k in blend["keys"]]) @ np.asarray(blend["weights"])
            prediction_rows = [{"id": row.id, "user_no": row.user_no, "question_id": row.get("question_id", ""),
                                "target": target, "actual": row[target], "prediction": pred,
                                "selected_model": "validation_weighted_blend", "feature_set": "mixed"}
                               for (_, row), pred in zip(val_t.iterrows(), blend_val)]
        pd.DataFrame(prediction_rows).to_csv(OUTPUT / "predictions" / f"{target}_val_predictions.csv", index=False)
        pd.DataFrame(val_records).to_csv(OUTPUT / "metrics" / f"{target}_validation_candidates.csv", index=False)
        pd.DataFrame(trial_records).to_csv(OUTPUT / "metrics" / f"{target}_cv_trials.csv", index=False)

        # Generate OOF predictions for the CV-best single model, for future fusion.
        spec = specs[cv_best["model"]]
        oof = np.full(len(train_t), np.nan)
        splitter = GroupKFold(n_splits=args.cv_folds)
        Xbest = train_t[cv_best["features"]].reset_index(drop=True)
        for tr, va in splitter.split(Xbest, y.reset_index(drop=True), groups):
            fold_model = model_pipeline(spec, cv_best["params"], args.seed)
            fold_model.fit(Xbest.iloc[tr], y.reset_index(drop=True).iloc[tr])
            oof[va] = fold_model.predict(Xbest.iloc[va])
        pd.DataFrame({"id": train_t.id, "user_no": train_t.user_no, "question_id": train_t.question_id,
                      "target": target, "actual": y, "oof_prediction": oof}).to_csv(
                          OUTPUT / "predictions" / f"{target}_oof_predictions.csv", index=False)

        # Refit on train+validation; use fixed test only now for the one final evaluation.
        trainval = pd.concat([train_t, val_t], ignore_index=True)
        ytv = pd.to_numeric(trainval[target]).astype(float)
        final_estimators = []
        if blend:
            final_predictions = []
            blend_entries = []
            for (model_name, set_name), weight in zip(blend["keys"], blend["weights"]):
                entry = next(r for r in candidate_records if r["model"] == model_name and r["feature_set"] == set_name)
                features = entry["features"]
                pipe = model_pipeline(specs[model_name], entry["params"], args.seed)
                pipe.fit(trainval[features], ytv)
                final_predictions.append(pipe.predict(test_t[features]) * weight)
                final_estimators.append((pipe, features, weight))
                blend_entries.append({"model": model_name, "feature_set": set_name, "weight": weight,
                                      "params": entry["params"], "features": features})
            test_pred = np.sum(final_predictions, axis=0)
            final_spec = None
        else:
            features = cv_best["features"]
            final_spec = specs[cv_best["model"]]
            final_model = model_pipeline(final_spec, cv_best["params"], args.seed)
            final_model.fit(trainval[features], ytv)
            final_estimators = [(final_model, features, 1.0)]
            test_pred = final_model.predict(test_t[features])
            blend_entries = []
        test_metrics = metric_set(pd.to_numeric(test_t[target]), test_pred)
        test_out = test_t[["id", "user_no", "question_id", target]].copy()
        test_out.rename(columns={target: "actual"}, inplace=True)
        test_out["prediction"] = test_pred
        test_out["absolute_error"] = np.abs(test_out.actual - test_out.prediction)
        test_out.to_csv(OUTPUT / "predictions" / f"{target}_test_predictions.csv", index=False)
        test_out.to_csv(OUTPUT / "analysis" / f"{target}_test_errors.csv", index=False)
        pd.DataFrame([{"target": target, **test_metrics}]).to_csv(
            OUTPUT / "metrics" / f"{target}_final_test_metrics.csv", index=False)

        # Fit importance model on train+validation and compute held-out feature permutation importance.
        imp_pipe = model_pipeline(specs[cv_best["model"]], cv_best["params"], args.seed)
        imp_pipe.fit(trainval[cv_best["features"]], ytv)
        _, _, _, permutation_importance, _, _ = deps()
        pi = permutation_importance(imp_pipe, test_t[cv_best["features"]],
                                    pd.to_numeric(test_t[target]), n_repeats=5,
                                    random_state=args.seed, scoring="neg_root_mean_squared_error", n_jobs=args.n_jobs)
        importance = pd.DataFrame({"feature": cv_best["features"], "permutation_importance_mean": pi.importances_mean,
                                   "permutation_importance_std": pi.importances_std}).sort_values(
                                       "permutation_importance_mean", ascending=False)
        importance.to_csv(OUTPUT / "feature_importance" / f"{target}_permutation_importance.csv", index=False)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 6))
            top = importance.head(10).sort_values("permutation_importance_mean")
            ax.barh(top.feature, top.permutation_importance_mean,
                    xerr=top.permutation_importance_std, color="#3977a8")
            ax.set_title(f"Top audio features for {target}")
            ax.set_xlabel("Permutation importance (increase in RMSE when shuffled)")
            fig.tight_layout()
            fig.savefig(OUTPUT / "plots" / f"{target}_feature_importance.png", dpi=160)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(test_out.actual, test_out.prediction, alpha=.65, s=18)
            bounds = [min(test_out.actual.min(), test_out.prediction.min()),
                      max(test_out.actual.max(), test_out.prediction.max())]
            ax.plot(bounds, bounds, "--", color="gray")
            ax.set(xlabel="Actual", ylabel="Predicted", title=f"Test predictions: {target}")
            fig.tight_layout()
            fig.savefig(OUTPUT / "plots" / f"{target}_test_predictions.png", dpi=160)
            plt.close(fig)
        except ImportError:
            (OUTPUT / "plots" / "PLOTS_UNAVAILABLE.txt").write_text(
                "Install matplotlib in the project environment to generate plots.\n", encoding="utf-8")

        # Analysis reports use test only after final model is locked; no tuning follows.
        question_records = []
        for question_id, group in test_out.groupby("question_id"):
            question_records.append({"question_id": question_id, "n": len(group),
                                     **metric_set(group.actual, group.prediction)})
        question_analysis = pd.DataFrame(question_records)
        question_analysis[question_analysis["n"] >= 5].to_csv(
            OUTPUT / "analysis" / f"{target}_test_by_question.csv", index=False)
        participant_input = test_out.merge(
            test_t[["id", "total_duration_seconds"]].drop_duplicates("id"), on="id", how="left")
        participants = participant_input.groupby("user_no").agg(
            n=("id", "size"), mae=("absolute_error", "mean"),
            mean_answer_duration_seconds=("total_duration_seconds", "mean"))
        participants["response_count_group"] = pd.qcut(
            participants.n.rank(method="first"), q=min(3, len(participants)),
            labels=["few", "middle", "many"][:min(3, len(participants))])
        participants.to_csv(OUTPUT / "analysis" / f"{target}_test_by_participant.csv")
        participant_groups = participants.groupby("response_count_group", observed=False).agg(
            participant_count=("n", "size"), mean_responses=("n", "mean"),
            mean_participant_mae=("mae", "mean"), mean_answer_duration_seconds=("mean_answer_duration_seconds", "mean"))
        participant_groups.to_csv(OUTPUT / "analysis" / f"{target}_test_participant_groups.csv")
        duration_bins = pd.qcut(test_t.total_duration_seconds.rank(method="first"), q=4,
                                labels=["shortest", "short", "long", "longest"])
        duration_errors = test_out.assign(duration_group=duration_bins.to_numpy()).groupby("duration_group").agg(
            n=("id", "size"), mae=("absolute_error", "mean"))
        signed = (test_out.assign(duration_group=duration_bins.to_numpy(),
                                  signed_error=test_out.prediction - test_out.actual)
                  .groupby("duration_group").signed_error.mean())
        duration_errors["mean_signed_error"] = signed
        duration_errors.to_csv(OUTPUT / "analysis" / f"{target}_test_duration_groups.csv")
        errors = test_out.sort_values("absolute_error", ascending=False).head(25).merge(
            test_t[["id", "question", "total_duration_seconds", "pause_count", "pause_max_duration_seconds",
                    "speech_duration_seconds", "f0_hz_mean"]].drop_duplicates("id"), on="id", how="left")
        errors.to_csv(OUTPUT / "analysis" / f"{target}_largest_errors.csv", index=False)

        model_dir = OUTPUT / "models" / target
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump(final_estimators, model_dir / "final_model.joblib")
        except Exception as exc:
            warnings.warn(f"Model serialization unavailable: {exc}")
        summary = {"target": target, "train_target_distribution": next(x for x in target_report(train, [target]) if x["target"] == target),
                   "best_single_candidate_by_cv": cv_best, "cv_metrics": cv_best["cv_metrics"],
                   "validation_candidates": val_records,
                   "ensemble_improved_validation_rmse": bool(blend), "validation_ensemble": blend,
                   "final_test_metrics": test_metrics,
                   "top_10_feature_importance": importance.head(10).to_dict(orient="records"),
                   "test_error_median": float(test_out.absolute_error.median()),
                   "test_error_p90": float(test_out.absolute_error.quantile(.9)),
                   "test_error_max": float(test_out.absolute_error.max()),
                   "test_question_analysis_path": f"analysis/{target}_test_by_question.csv",
                   "test_participant_analysis_path": f"analysis/{target}_test_by_participant.csv",
                   "test_largest_errors_path": f"analysis/{target}_largest_errors.csv"}
        (OUTPUT / "metrics" / f"{target}_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        print(f"{target}: CV RMSE={cv_best['cv_rmse']:.4f}; final test={test_metrics}; "
              f"ensemble improved validation={bool(blend)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
