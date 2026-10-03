"""Diagnostic-only baselines, confound checks, grouped CV, bootstrap CIs and residuals.

This script never fits on the held-out test set. It uses the fixed participant-
disjoint train/validation split and the existing model comparison validation
predictions for residual diagnostics of each target's selected baseline.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_audio_models import load_config, load_split_data, project_path  # noqa: E402


TARGETS = ("confidence_score", "speaking_skills", "overall_performance")
BASELINE_FEATURES = {
    "duration": ["audio_duration_seconds"],
    "word_count": ["transcript_word_count"],
    "transcript_rate": ["transcript_words_per_second"],
    "mean_rms": ["rms_energy_mean"],
    "combined_simple": ["audio_duration_seconds", "transcript_word_count", "transcript_words_per_second", "rms_energy_mean"],
}
TOKEN_RE = re.compile(r"\b[\w]+(?:['’][\w]+)?\b", flags=re.UNICODE)


def _finite_pair(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _corr(x, y, method="pearson"):
    xx, yy = _finite_pair(x, y)
    if len(xx) < 3 or np.std(xx) == 0 or np.std(yy) == 0:
        return float("nan")
    if method == "spearman":
        return float(pd.Series(xx).corr(pd.Series(yy), method="spearman"))
    return float(np.corrcoef(xx, yy)[0, 1])


def _metrics(y, pred):
    y, pred = _finite_pair(y, pred)
    if not len(y):
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "pearson": np.nan, "spearman": np.nan}
    residual = y - pred
    denom = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / denom) if denom > 0 else np.nan,
        "pearson": _corr(y, pred),
        "spearman": _corr(y, pred, "spearman"),
    }


def _cluster_bootstrap(frame, actual_col, pred_col, user_col, replicates, seed):
    """Percentile CIs from resampling participants as clusters, retaining all their rows."""
    users = frame[user_col].astype(str).unique()
    if len(users) < 2:
        return {metric: {"lower_95": np.nan, "upper_95": np.nan} for metric in ("pearson", "spearman", "r2")}
    rng = np.random.default_rng(seed)
    by_user = {user: frame.loc[frame[user_col].astype(str).eq(user)] for user in users}
    draws = {metric: [] for metric in ("pearson", "spearman", "r2")}
    for _ in range(replicates):
        sampled = rng.choice(users, size=len(users), replace=True)
        # Concatenating each draw preserves multiplicity when a participant is sampled twice.
        sample = pd.concat([by_user[user] for user in sampled], ignore_index=True)
        values = _metrics(sample[actual_col], sample[pred_col])
        for metric in draws:
            if np.isfinite(values[metric]):
                draws[metric].append(values[metric])
    return {
        metric: {
            "lower_95": float(np.quantile(values, 0.025)) if values else np.nan,
            "upper_95": float(np.quantile(values, 0.975)) if values else np.nan,
        }
        for metric, values in draws.items()
    }


def _eta_squared(values, groups):
    frame = pd.DataFrame({"value": pd.to_numeric(pd.Series(values), errors="coerce"), "group": pd.Series(groups).astype("string")})
    frame = frame.dropna()
    if len(frame) < 3 or frame.group.nunique() < 2:
        return np.nan
    overall = frame.value.mean()
    total = float(((frame.value - overall) ** 2).sum())
    if total <= 0:
        return np.nan
    between = sum(len(group) * (group.value.mean() - overall) ** 2 for _, group in frame.groupby("group", observed=True))
    return float(between / total)


def _distribution_stats(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(vals):
        return {"n": 0}
    quantiles = np.quantile(vals, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {
        "n": int(len(vals)), "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(quantiles[0]), "p01": float(quantiles[1]), "p05": float(quantiles[2]), "q1": float(quantiles[3]),
        "median": float(quantiles[4]), "q3": float(quantiles[5]), "p95": float(quantiles[6]), "p99": float(quantiles[7]), "max": float(quantiles[8]),
        "unique_values": int(pd.Series(vals).nunique()), "fraction_at_min": float(np.mean(vals == np.min(vals))), "fraction_at_max": float(np.mean(vals == np.max(vals))),
    }


def _read_features(config, group, split):
    directory = project_path(config["paths"]["legacy_features_dir"] if group == "legacy" else config["paths"]["features_dir"]) 
    candidates = ([directory / f"{split}_features.csv", directory / group / f"{split}_features.csv"] if group == "egemaps" else [directory / f"{split}_features.csv", directory / "legacy" / f"{split}_features.csv"])
    for path in candidates:
        if path.is_file():
            frame = pd.read_csv(path, dtype={"id": str})
            if "id" in frame:
                return frame.drop_duplicates("id", keep="first")
    return pd.DataFrame(columns=["id"])


def _make_data(config):
    data = load_split_data(config)
    for split, frame in data.items():
        frame = frame.copy()
        # Metadata duration is measured from the extracted audio manifest.
        duration_candidates = [c for c in ("duration_seconds", "duration") if c in frame]
        if duration_candidates:
            frame["audio_duration_seconds"] = pd.to_numeric(frame[duration_candidates[0]], errors="coerce")
        else:
            frame["audio_duration_seconds"] = np.nan
        transcript = frame["transcript"] if "transcript" in frame else pd.Series("", index=frame.index)
        frame["transcript_word_count"] = transcript.fillna("").astype(str).map(lambda value: len(TOKEN_RE.findall(value))).astype(float)
        frame["transcript_words_per_second"] = frame.transcript_word_count / frame.audio_duration_seconds.replace(0, np.nan)
        for group in ("legacy", "egemaps"):
            features = _read_features(config, group, split)
            if len(features):
                if group == "legacy":
                    wanted = [c for c in ("rms_energy_mean", "rms_energy_std", "f0_hz_mean", "speech_fraction", "speech_duration_seconds", "pause_rate_per_minute") if c in features]
                else:
                    wanted = [c for c in ("loudness_sma3_amean", "equivalentSoundLevel_dBp", "HNRdBACF_sma3nz_amean", "F0semitoneFrom27.5Hz_sma3nz_amean") if c in features]
                    wanted = [c for c in wanted if c not in frame]
                    features = features.rename(columns={c: f"egemaps_{c}" for c in wanted})
                    wanted = [f"egemaps_{c}" for c in wanted]
                selected = features[["id", *wanted]].copy()
                for column in wanted:
                    selected[column] = pd.to_numeric(selected[column], errors="coerce")
                frame = frame.merge(selected, on="id", how="left", validate="one_to_one")
        data[split] = frame
    return data


def _assert_baseline_features(feature_names, target):
    forbidden = {target, "confidence", "confidence_score", "speaking_skills", "overall_performance", "user_no", "user_id", "question_id"}
    overlap = forbidden.intersection(feature_names)
    if overlap:
        raise AssertionError(f"Diagnostic X contains target/identity/question columns: {sorted(overlap)}")


def _fit_predict(train, evaluate, features, target):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    _assert_baseline_features(features, target)
    Xtr = train[features].apply(pd.to_numeric, errors="coerce")
    Xev = evaluate[features].apply(pd.to_numeric, errors="coerce")
    ytr = pd.to_numeric(train[target], errors="coerce")
    keep = ytr.notna()
    if not keep.any():
        raise ValueError(f"No nonmissing training labels for {target}")
    pipeline = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10.0, solver="lsqr"))
    pipeline.fit(Xtr.loc[keep], ytr.loc[keep])
    return pipeline.predict(Xev)


def _baseline_evaluation(data, config, output_dir, replicates):
    from sklearn.model_selection import GroupKFold

    rows, cv_rows = [], []
    prediction_frames = {}
    folds = int(config["modeling"].get("cv_folds", 5))
    train = data["train"].reset_index(drop=True)
    val = data["val"].reset_index(drop=True)
    groups = train.user_no.astype(str)
    if groups.nunique() < 2:
        raise ValueError("Grouped CV needs at least two training participants")
    n_splits = min(folds, groups.nunique())
    splitter = GroupKFold(n_splits=n_splits)
    for target in TARGETS:
        if target not in train or target not in val:
            continue
        for feature_name, features in BASELINE_FEATURES.items():
            absent = [name for name in features if name not in train or not pd.to_numeric(train[name], errors="coerce").notna().any()]
            if absent:
                rows.append({"target": target, "feature_set": feature_name, "status": "unavailable", "reason": f"No reliable source values for {absent}"})
                continue
            try:
                val_y = pd.to_numeric(val[target], errors="coerce")
                val_mask = val_y.notna()
                val_pred = _fit_predict(train, val.loc[val_mask], features, target)
                validation = val.loc[val_mask, ["id", "user_no", "question_id", target]].copy()
                validation["prediction"] = val_pred
                validation["actual"] = val_y.loc[val_mask].to_numpy(float)
                validation["residual"] = validation.actual - validation.prediction
                prediction_frames[(target, feature_name)] = validation
                point = _metrics(validation.actual, validation.prediction)
                ci = _cluster_bootstrap(validation, "actual", "prediction", "user_no", replicates, int(config["modeling"].get("bootstrap_seed", 43)))
                rows.append({"target": target, "feature_set": feature_name, "features": json.dumps(features), "status": "ok", **point,
                             **{f"{metric}_{bound}": value for metric, bounds in ci.items() for bound, value in bounds.items()},
                             "participants": int(validation.user_no.nunique())})
                oof = np.full(len(train), np.nan)
                fold_records = []
                for fold_num, (tr, va) in enumerate(splitter.split(train, groups=groups), start=1):
                    train_users = set(groups.iloc[tr].astype(str))
                    validation_users = set(groups.iloc[va].astype(str))
                    if train_users & validation_users:
                        raise AssertionError(f"GroupKFold participant leakage in fold {fold_num}: {sorted(train_users & validation_users)}")
                    fold = train.iloc[va].copy()
                    yfold = pd.to_numeric(fold[target], errors="coerce")
                    valid = yfold.notna()
                    preds = _fit_predict(train.iloc[tr], fold.loc[valid], features, target)
                    oof[np.asarray(va)[valid.to_numpy()]] = preds
                    fold_metrics = _metrics(yfold.loc[valid], preds)
                    fold_records.append({"target": target, "feature_set": feature_name, "fold": fold_num, "participants": int(groups.iloc[va].nunique()), **fold_metrics})
                cv_rows.extend(fold_records)
                cv_frame = pd.DataFrame({"actual": pd.to_numeric(train[target], errors="coerce"), "prediction": oof, "user_no": groups})
                cv_frame = cv_frame.dropna(subset=["actual", "prediction"])
                pooled = _metrics(cv_frame.actual, cv_frame.prediction)
                cv_rows.append({"target": target, "feature_set": feature_name, "fold": "pooled_oof", "participants": int(cv_frame.user_no.nunique()), **pooled})
            except Exception as exc:
                rows.append({"target": target, "feature_set": feature_name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    validation_path = output_dir / "baseline_validation.csv"
    pd.DataFrame(rows).to_csv(validation_path, index=False)
    pd.DataFrame(cv_rows).to_csv(output_dir / "grouped_cv.csv", index=False)
    return rows, cv_rows, prediction_frames


def _confounds(data, output_dir):
    # The rows describe association only. user_no is never provided to a model.
    all_val = data["val"].copy()
    acoustic = [c for c in ("rms_energy_mean", "rms_energy_std", "f0_hz_mean", "speech_fraction", "pause_rate_per_minute",
                            "egemaps_loudness_sma3_amean", "egemaps_equivalentSoundLevel_dBp", "egemaps_HNRdBACF_sma3nz_amean",
                            "egemaps_F0semitoneFrom27.5Hz_sma3nz_amean") if c in all_val]
    columns = ["audio_duration_seconds", *acoustic]
    rows = []
    for target in TARGETS:
        for column in columns:
            if column not in all_val:
                continue
            x = pd.to_numeric(all_val[column], errors="coerce")
            y = pd.to_numeric(all_val[target], errors="coerce")
            rows.append({"target": target, "factor": column, "kind": "numeric", "n": int((x.notna() & y.notna()).sum()),
                         "pearson": _corr(x, y), "spearman": _corr(x, y, "spearman"), "eta_squared": np.nan,
                         "note": "Descriptive validation association; not a causal effect."})
        for column in ("question_id", "video_quality"):
            if column not in all_val:
                continue
            rows.append({"target": target, "factor": column, "kind": "grouped", "n": int(pd.to_numeric(all_val[target], errors="coerce").notna().sum()),
                         "pearson": np.nan, "spearman": np.nan, "eta_squared": _eta_squared(all_val[target], all_val[column]),
                         "group_count": int(all_val[column].nunique(dropna=True)), "note": "Eta-squared for target variation across groups; descriptive only."})
    pd.DataFrame(rows).to_csv(output_dir / "confound_associations.csv", index=False)
    participant_rows = []
    for target in TARGETS:
        frame = all_val[["user_no", target]].copy()
        frame[target] = pd.to_numeric(frame[target], errors="coerce")
        frame = frame.dropna()
        grouped = frame.groupby("user_no")[target]
        means = grouped.mean()
        counts = grouped.size().astype(float)
        n_groups = len(counts)
        grand_mean = frame[target].mean()
        ss_between = float((counts * (means - grand_mean) ** 2).sum())
        ss_within = float(sum(((group - group.mean()) ** 2).sum() for _, group in grouped))
        ms_between = ss_between / (n_groups - 1) if n_groups > 1 else np.nan
        ms_within = ss_within / (len(frame) - n_groups) if len(frame) > n_groups else np.nan
        effective_group_size = (len(frame) - float((counts**2).sum()) / len(frame)) / (n_groups - 1) if n_groups > 1 else np.nan
        icc1 = ((ms_between - ms_within) / (ms_between + (effective_group_size - 1) * ms_within)
                if pd.notna(ms_between) and pd.notna(ms_within) and pd.notna(effective_group_size)
                and (ms_between + (effective_group_size - 1) * ms_within) > 0 else np.nan)
        mean_within_var = grouped.var(ddof=1).mean()
        participant_rows.append({"target": target, "n_clips": len(frame), "n_participants": int(grouped.ngroups), "overall_mean": grand_mean,
                                 "participant_mean_sd": means.std(ddof=1), "mean_within_participant_sd": float(np.sqrt(mean_within_var)) if pd.notna(mean_within_var) and mean_within_var >= 0 else np.nan,
                                 "icc1_unbalanced": icc1,
                                 "interpretation": "Descriptive clustering diagnostic only; participant identity is not a predictive feature."})
    pd.DataFrame(participant_rows).to_csv(output_dir / "participant_statistics.csv", index=False)


def _target_analysis(data, output_dir):
    frames = [data[s] for s in ("train", "val", "test")]
    combined = pd.concat(frames, ignore_index=True)
    stats = []
    for split in ("train", "val", "test", "all_fixed_splits"):
        frame = combined if split == "all_fixed_splits" else data[split]
        for target in TARGETS:
            if target in frame:
                stats.append({"split": split, "target": target, **_distribution_stats(frame[target])})
    pd.DataFrame(stats).to_csv(output_dir / "target_statistics.csv", index=False)
    corr_rows = []
    for a in TARGETS:
        for b in TARGETS:
            corr_rows.append({"target_a": a, "target_b": b, "pearson": _corr(combined[a], combined[b]), "spearman": _corr(combined[a], combined[b], "spearman"), "n": int((combined[a].notna() & combined[b].notna()).sum())})
    pd.DataFrame(corr_rows).to_csv(output_dir / "target_correlations.csv", index=False)
    return stats, corr_rows


def _existing_best_model_residuals(config, data, output_dir, replicates):
    report_dir = project_path(config["paths"]["reports_dir"])
    summary_path = report_dir / "comparison_summary.json"
    if not summary_path.is_file():
        return [], "Existing comparison summary not found; skipped selected-model residual analysis."
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows, plot_frames = [], {}
    for target in TARGETS:
        selected = summary.get("selected", {}).get(target)
        prediction_path = report_dir / "predictions" / f"{target}_val_predictions.csv"
        if not selected or not prediction_path.is_file():
            continue
        pred = pd.read_csv(prediction_path, dtype={"id": str, "user_id": str, "question_id": str})
        # Join split-derived confounds by the stable clip key; labels/predictions remain from the saved val file.
        meta = data["val"].copy()
        cols = [c for c in ("id", "audio_duration_seconds", "video_quality", "question_id", "user_no") if c in meta]
        pred = pred.merge(meta[cols].drop_duplicates("id"), on="id", how="left", suffixes=("", "_meta"), validate="one_to_one")
        pred["residual"] = pd.to_numeric(pred.actual, errors="coerce") - pd.to_numeric(pred.prediction, errors="coerce")
        point = _metrics(pred.actual, pred.prediction)
        ci = _cluster_bootstrap(pred, "actual", "prediction", "user_no", replicates, int(config["modeling"].get("bootstrap_seed", 43)))
        base = {"target": target, "feature_set": selected.get("feature_set"), "model": selected.get("model"), **point,
                **{f"{metric}_{bound}": value for metric, bounds in ci.items() for bound, value in bounds.items()}}
        for column in ("audio_duration_seconds", "question_id", "video_quality"):
            if column not in pred:
                continue
            if column == "audio_duration_seconds":
                assoc = _corr(pred[column], pred.residual)
                kind = "pearson"
            else:
                assoc = _eta_squared(pred.residual, pred[column])
                kind = "eta_squared"
            rows.append({**base, "residual_factor": column, "association_type": kind, "residual_association": assoc})
        rows.append({**base, "residual_factor": "distribution", "association_type": "summary", "residual_mean": float(pred.residual.mean()), "residual_std": float(pred.residual.std(ddof=1)),
                     "residual_p05": float(pred.residual.quantile(.05)), "residual_median": float(pred.residual.median()), "residual_p95": float(pred.residual.quantile(.95))})
        plot_frames[target] = pred
    pd.DataFrame(rows).to_csv(output_dir / "best_model_residuals.csv", index=False)
    _write_plots(data, plot_frames, output_dir)
    return rows, None if plot_frames else "No saved validation predictions were available for residual analysis."


def _write_plots(data, residual_frames, output_dir):
    (output_dir / "PLOTS_UNAVAILABLE.txt").unlink(missing_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([data[s] for s in ("train", "val", "test")], ignore_index=True)
    for target in TARGETS:
        _hist_svg(combined[target], f"Target distribution: {target}", target, plot_dir / f"target_{target}.svg")
    if residual_frames:
        for target, frame in residual_frames.items():
            _scatter_svg(frame.actual, frame.prediction, f"{target}: predicted vs actual", "Actual", "Predicted", plot_dir / f"predicted_actual_{target}.svg", identity=True)
            _scatter_svg(frame.audio_duration_seconds, frame.residual, f"{target}: residual vs duration", "Audio duration (s)", "Actual - predicted", plot_dir / f"residual_duration_{target}.svg", zero_line=True)
            _hist_svg(frame.residual, f"{target}: residual distribution", "Actual - predicted", plot_dir / f"residual_distribution_{target}.svg")
            _box_svg(frame.residual, frame.question_id, f"{target}: residual by question", "Question ID", plot_dir / f"residual_question_{target}.svg")
            if "video_quality" in frame and frame.video_quality.notna().any():
                _box_svg(frame.residual, frame.video_quality, f"{target}: residual by video quality", "Video quality", plot_dir / f"residual_video_quality_{target}.svg")


def _svg_text(x, y, value, size=12, anchor="middle", weight="normal"):
    from html import escape
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="#263238">{escape(str(value))}</text>'


def _svg_canvas(title, x_label, y_label, path, draw):
    width, height = 760, 470
    left, right, top, bottom = 82, 24, 55, 75
    x0, x1, y0, y1 = left, width - right, height - bottom, top
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>', _svg_text(width / 2, 28, title, 17, weight="bold"),
            f'<path d="M{x0},{y1} V{y0} H{x1}" fill="none" stroke="#455a64" stroke-width="1.2"/>',
            _svg_text((x0+x1)/2, height-20, x_label, 13),
            f'<text x="20" y="{(y0+y1)/2:.1f}" font-size="13" text-anchor="middle" transform="rotate(-90 20 {(y0+y1)/2:.1f})" fill="#263238">{y_label}</text>']
    draw(body, x0, x1, y0, y1)
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def _scatter_svg(x, y, title, x_label, y_label, path, identity=False, zero_line=False):
    xx, yy = _finite_pair(x, y)
    if not len(xx):
        return
    xmin, xmax = float(np.min(xx)), float(np.max(xx)); ymin, ymax = float(np.min(yy)), float(np.max(yy))
    padx = (xmax - xmin) * .04 or 1.0; pady = (ymax - ymin) * .08 or 1.0
    xmin -= padx; xmax += padx; ymin -= pady; ymax += pady
    def draw(body, x0, x1, y0, y1):
        X = lambda v: x0 + (v - xmin) / (xmax - xmin) * (x1 - x0)
        Y = lambda v: y0 - (v - ymin) / (ymax - ymin) * (y0 - y1)
        for tick in np.linspace(0, 1, 5):
            xv = xmin + tick * (xmax-xmin); yv = ymin + tick * (ymax-ymin)
            body.append(f'<path d="M{X(xv):.1f},{y1} V{y0}" stroke="#eceff1"/>')
            body.append(_svg_text(X(xv), y0+20, f"{xv:.2g}", 10))
            body.append(f'<path d="M{x0},{Y(yv):.1f} H{x1}" stroke="#eceff1"/>')
            body.append(_svg_text(x0-10, Y(yv)+4, f"{yv:.2g}", 10, anchor="end"))
        if identity:
            a, b = max(xmin, ymin), min(xmax, ymax)
            body.append(f'<path d="M{X(a):.1f},{Y(a):.1f} L{X(b):.1f},{Y(b):.1f}" stroke="#d1495b" stroke-dasharray="6 4"/>')
        if zero_line and ymin <= 0 <= ymax:
            body.append(f'<path d="M{x0},{Y(0):.1f} H{x1}" stroke="#d1495b" stroke-dasharray="6 4"/>')
        for a, b in zip(xx, yy):
            body.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="3" fill="#3979a8" fill-opacity=".48"/>')
    _svg_canvas(title, x_label, y_label, path, draw)


def _hist_svg(values, title, x_label, path, bins=24):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if not len(vals):
        return
    counts, edges = np.histogram(vals, bins=bins)
    ymax = max(1, int(counts.max()))
    def draw(body, x0, x1, y0, y1):
        for i, count in enumerate(counts):
            bx0 = x0 + (edges[i]-edges[0])/(edges[-1]-edges[0])*(x1-x0)
            bx1 = x0 + (edges[i+1]-edges[0])/(edges[-1]-edges[0])*(x1-x0)
            by = y0 - count/ymax*(y0-y1)
            body.append(f'<rect x="{bx0:.1f}" y="{by:.1f}" width="{max(0,bx1-bx0-1):.1f}" height="{y0-by:.1f}" fill="#496a81"/>')
        for tick in range(5):
            value = edges[0] + tick/4*(edges[-1]-edges[0]); xpos = x0+tick/4*(x1-x0)
            body.append(_svg_text(xpos, y0+20, f"{value:.2g}", 10))
        body.append(_svg_text(x0-10, y1+5, str(ymax), 10, anchor="end"))
        body.append(_svg_text(x0-10, y0, "0", 10, anchor="end"))
    _svg_canvas(title, x_label, "Count", path, draw)


def _box_svg(values, groups, title, x_label, path):
    frame = pd.DataFrame({"value": pd.to_numeric(pd.Series(values), errors="coerce"), "group": pd.Series(groups).astype("string")}).dropna()
    if frame.empty:
        return
    categories = sorted(frame.group.unique(), key=str)
    if len(categories) > 30:
        # Keep the most represented 29 levels and pool the remainder for readable plots.
        frequent = set(frame.group.value_counts().head(29).index)
        frame["group"] = frame.group.map(lambda v: v if v in frequent else "other")
        categories = sorted(frame.group.unique(), key=str)
    ymin, ymax = float(frame.value.min()), float(frame.value.max()); pad = (ymax-ymin)*.08 or 1.0; ymin-=pad; ymax+=pad
    def quantile(a, q): return float(np.quantile(a, q))
    def draw(body, x0, x1, y0, y1):
        Y=lambda v: y0-(v-ymin)/(ymax-ymin)*(y0-y1)
        for tick in np.linspace(ymin,ymax,5):
            body.append(f'<path d="M{x0},{Y(tick):.1f} H{x1}" stroke="#eceff1"/>')
            body.append(_svg_text(x0-8,Y(tick)+4,f"{tick:.2g}",10,anchor="end"))
        width=(x1-x0)/max(1,len(categories));
        for i,cat in enumerate(categories):
            vals=frame.loc[frame.group.eq(cat),"value"].to_numpy(float)
            cx=x0+(i+.5)*width; q1=quantile(vals,.25); med=quantile(vals,.5); q3=quantile(vals,.75); low=quantile(vals,.05); high=quantile(vals,.95)
            bw=min(32,width*.55)
            body.append(f'<path d="M{cx:.1f},{Y(low):.1f} V{Y(high):.1f} M{cx-bw/4:.1f},{Y(low):.1f} H{cx+bw/4:.1f} M{cx-bw/4:.1f},{Y(high):.1f} H{cx+bw/4:.1f}" stroke="#455a64"/>')
            body.append(f'<rect x="{cx-bw/2:.1f}" y="{Y(q3):.1f}" width="{bw:.1f}" height="{max(1,Y(q1)-Y(q3)):.1f}" fill="#91b8d0" stroke="#3979a8"/>')
            body.append(f'<path d="M{cx-bw/2:.1f},{Y(med):.1f} H{cx+bw/2:.1f}" stroke="#d1495b" stroke-width="2"/>')
            body.append(_svg_text(cx,y0+18,str(cat),10))
    _svg_canvas(title, x_label, "Residual (actual - predicted)", path, draw)


def _write_report(output_dir, baseline_rows, cv_rows, target_stats, confounds, residual_rows, skip_message, replicates):
    lines = ["# Audio diagnostic and robust-evaluation report", "", "Diagnostic-only results on the existing fixed split. No model/feature selection was changed; test labels were used only for descriptive target distributions/correlations, not model fitting or scoring.", "",
             "## Feature availability", "", "Clip duration and transcript word count are available. Transcript words/second is an approximate rate over full clip duration; word-level timings and speech-only rate are unavailable. Mean RMS is read from the existing legacy acoustic feature cache. Participant IDs are used only for grouping and cluster resampling.", "",
             f"## Evaluation protocol", "", f"Simple-feature baselines use train-only median imputation, standard scaling and Ridge (alpha=10, LSQR). Validation uses the fixed validation participants. GroupKFold uses user_no on training participants only. Participant bootstrap uses {replicates} resamples of validation participants.", "",
             "## Validation simple-feature baselines", "", "| Target | Features | MAE | RMSE | R² | Pearson | Spearman | R² 95% CI | Pearson 95% CI | Spearman 95% CI |", "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    for row in baseline_rows:
        if row.get("status") != "ok":
            lines.append(f"| {row.get('target')} | {row.get('feature_set')} | unavailable | | | | | | | |")
            continue
        lines.append(f"| {row['target']} | {row['feature_set']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} | {row['pearson']:.3f} | {row['spearman']:.3f} | [{row['r2_lower_95']:.3f}, {row['r2_upper_95']:.3f}] | [{row['pearson_lower_95']:.3f}, {row['pearson_upper_95']:.3f}] | [{row['spearman_lower_95']:.3f}, {row['spearman_upper_95']:.3f}] |")
    lines += ["", "## Grouped cross-validation", "", "Pooled out-of-fold training metrics (fold details are in `grouped_cv.csv`):", "", "| Target | Features | MAE | RMSE | R² | Pearson | Spearman |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in cv_rows:
        if row.get("fold") == "pooled_oof":
            lines.append(f"| {row['target']} | {row['feature_set']} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.3f} | {row['pearson']:.3f} | {row['spearman']:.3f} |")
    lines += ["", "## Confounds and residuals", "", "See `confound_associations.csv`, `participant_statistics.csv`, `best_model_residuals.csv`, and `plots/`. Numeric associations include Pearson/Spearman; question/video-quality associations use eta-squared. These are diagnostic associations, not evidence of causality. Participant identity was never included in X.", "", "## Existing selected baseline residual analysis", ""]
    if skip_message:
        lines.append(skip_message)
    else:
        lines.append("Residuals use saved validation predictions for the selected model per target from the most recent comparison summary. Bootstrap intervals are participant-clustered.")
    lines += ["", "## Target analysis", "", "See `target_statistics.csv` for split-wise distributions, quantiles, ranges and endpoint concentration; `target_correlations.csv` contains all-split target correlations. No labels were changed or created.", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(config, replicates=1000):
    output_dir = project_path(config["paths"]["reports_dir"]) / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _make_data(config)
    baseline_rows, cv_rows, _ = _baseline_evaluation(data, config, output_dir, replicates)
    _confounds(data, output_dir)
    target_stats, target_corrs = _target_analysis(data, output_dir)
    residual_rows, skip_message = _existing_best_model_residuals(config, data, output_dir, replicates)
    _write_report(output_dir, baseline_rows, cv_rows, target_stats, None, residual_rows, skip_message, replicates)
    summary = {"branch_scope": "diagnostic only", "test_model_metrics_computed": False, "participant_identity_in_predictors": False,
               "bootstrap_unit": "user_no participant cluster", "bootstrap_replicates": replicates,
               "available_trivial_features": ["audio_duration_seconds", "transcript_word_count", "transcript_words_per_second (approximate)", "rms_energy_mean"],
               "word_timing_available": False, "baseline_results": baseline_rows,
               "successful_grouped_cv_pooled_rows": [r for r in cv_rows if r.get("fold") == "pooled_oof"],
               "target_correlations": target_corrs, "best_model_residual_rows": residual_rows, "residual_analysis_note": skip_message,
               "outputs": ["report.md", "baseline_validation.csv", "grouped_cv.csv", "confound_associations.csv", "participant_statistics.csv", "target_statistics.csv", "target_correlations.csv", "best_model_residuals.csv", "plots/"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: None if pd.isna(x) else float(x)), encoding="utf-8")
    print(f"Diagnostics written to: {output_dir}", flush=True)
    successful = [row for row in baseline_rows if row.get("status") == "ok"]
    if successful:
        print("\nVALIDATION SIMPLE-FEATURE BASELINES", flush=True)
        print(pd.DataFrame(successful)[["target", "feature_set", "mae", "rmse", "r2", "pearson", "spearman"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"\nGrouped CV pooled rows: {sum(r.get('fold') == 'pooled_oof' for r in cv_rows)}", flush=True)
    if skip_message:
        print(skip_message, flush=True)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    replicates = args.bootstrap_replicates or int(config["modeling"].get("bootstrap_replicates", 1000))
    if replicates < 1:
        parser.error("--bootstrap-replicates must be >= 1")
    run(config, replicates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
