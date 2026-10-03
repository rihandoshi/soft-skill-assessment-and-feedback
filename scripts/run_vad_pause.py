"""Silero VAD and pause-aware SSL pooling diagnostic on the fixed participant split."""
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
    atomic_csv, atomic_json, atomic_npz, bootstrap_by_participant, build_pipeline,
    load_config, load_split_data, metrics, project_path,
)
from scripts.extract_audio_features import TransformerEmbeddingExtractor, read_wav  # noqa: E402

TARGETS = ("confidence_score", "speaking_skills", "overall_performance")
ENCODERS = ("wavlm", "hubert")
VAD_OPTIONS = {"sample_rate": 16000, "threshold": 0.5, "min_speech_duration_ms": 100,
               "min_silence_duration_ms": 100, "speech_pad_ms": 30}


def vad_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "options": VAD_OPTIONS,
               "model": "silero-vad", "implementation": "silero_vad.get_speech_timestamps"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def get_vad_segments(waveform, sample_rate, vad_model, torch, device):
    from silero_vad import get_speech_timestamps
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.asarray(waveform, dtype=np.float32)).to(device), vad_model,
        threshold=VAD_OPTIONS["threshold"], sampling_rate=sample_rate,
        min_speech_duration_ms=VAD_OPTIONS["min_speech_duration_ms"],
        min_silence_duration_ms=VAD_OPTIONS["min_silence_duration_ms"],
        speech_pad_ms=VAD_OPTIONS["speech_pad_ms"],
        return_seconds=True,
    )
    return [(max(0.0, float(item["start"])), min(len(waveform) / sample_rate, float(item["end"])))
            for item in timestamps if float(item["end"]) > float(item["start"])]


def pause_features(segments, duration):
    speech = [(float(a), float(b)) for a, b in segments if b > a]
    speech_duration = min(float(duration), sum(b - a for a, b in speech))
    gaps = [max(0.0, speech[i + 1][0] - speech[i][1]) for i in range(len(speech) - 1)]
    gaps = [gap for gap in gaps if gap > 0]
    non_speech = max(0.0, float(duration) - speech_duration)
    return {
        "duration_seconds": float(duration), "speech_duration_seconds": speech_duration,
        "non_speech_duration_seconds": non_speech,
        "pause_ratio": non_speech / duration if duration > 0 else np.nan,
        "internal_pause_duration_seconds": float(sum(gaps)),
        "internal_pause_ratio": float(sum(gaps)) / duration if duration > 0 else np.nan,
        "number_of_pauses": len(gaps),
        "mean_pause_duration_seconds": float(np.mean(gaps)) if gaps else 0.0,
        "median_pause_duration_seconds": float(np.median(gaps)) if gaps else 0.0,
        "maximum_pause_duration_seconds": float(max(gaps)) if gaps else 0.0,
        "pauses_over_0_5_seconds": int(sum(gap > 0.5 for gap in gaps)),
        "pauses_over_1_second": int(sum(gap > 1.0 for gap in gaps)),
    }


def acoustic_features(waveform, sample_rate):
    frame, hop = int(.025 * sample_rate), int(.010 * sample_rate)
    if len(waveform) < frame:
        waveform = np.pad(waveform, (0, frame - len(waveform)))
    rms = np.asarray([np.sqrt(np.mean(waveform[i:i + frame] ** 2))
                      for i in range(0, max(1, len(waveform) - frame + 1), hop)])
    return {"mean_rms": float(rms.mean()) if len(rms) else 0.0,
            "median_rms": float(np.median(rms)) if len(rms) else 0.0}


def markdown_table(frame, floatfmt=".4f"):
    columns = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)) and np.isfinite(value):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        rows.append(cells)
    output = ["| " + " | ".join(columns) + " |",
              "| " + " | ".join(["---"] * len(columns)) + " |"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def _split_cache(config, split):
    return project_path(config["paths"]["cache_dir"]) / "vad_pause" / "vad" / f"{split}.json"


def create_vad_cache(config, data, split, vad_model, torch, device, result_dir):
    cache_path = _split_cache(config, split)
    cached = {}
    if cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("options") == VAD_OPTIONS:
                cached = payload.get("clips", {})
        except (OSError, json.JSONDecodeError):
            pass
    output, segment_rows, feature_rows = {}, [], []
    for row_index, row in enumerate(data[split].itertuples(index=False), start=1):
        path = project_path(row.audio_path)
        signature = vad_fingerprint(path)
        item = cached.get(str(row.id))
        if not item or item.get("audio_fingerprint") != signature:
            waveform, sample_rate = read_wav(path)
            if sample_rate != VAD_OPTIONS["sample_rate"]:
                raise ValueError(f"VAD requires 16 kHz input: {path} ({sample_rate} Hz)")
            segments = get_vad_segments(waveform, sample_rate, vad_model, torch, device)
            item = {"audio_fingerprint": signature, "segments": segments,
                    "duration_seconds": len(waveform) / sample_rate}
            cached[str(row.id)] = item
        segments = [(float(a), float(b)) for a, b in item["segments"]]
        duration = float(item["duration_seconds"])
        if "pause_statistics" not in item:
            item["pause_statistics"] = pause_features(segments, duration)
        output[str(row.id)] = segments
        for index, (start, end) in enumerate(segments):
            segment_rows.append({"id": str(row.id), "user_id": str(row.user_id), "split": split,
                                 "region_type": "speech", "region_index": index,
                                 "start_seconds": start, "end_seconds": end, "duration_seconds": end - start})
        # Explicit complement intervals provide non-speech regions including leading/trailing silence.
        cursor, non_idx = 0.0, 0
        for start, end in segments:
            if start > cursor:
                segment_rows.append({"id": str(row.id), "user_id": str(row.user_id), "split": split,
                                     "region_type": "non_speech", "region_index": non_idx,
                                     "start_seconds": cursor, "end_seconds": start, "duration_seconds": start - cursor})
                non_idx += 1
            cursor = max(cursor, end)
        if cursor < duration:
            segment_rows.append({"id": str(row.id), "user_id": str(row.user_id), "split": split,
                                 "region_type": "non_speech", "region_index": non_idx,
                                 "start_seconds": cursor, "end_seconds": duration, "duration_seconds": duration - cursor})
        acoustic = {}
        if split in ("train", "val"):
            waveform, sample_rate = read_wav(path)
            acoustic = acoustic_features(waveform, sample_rate)
        feature_rows.append({"id": str(row.id), "user_id": str(row.user_id), "question_id": str(row.question_id),
                             "split": split, **{target: getattr(row, target) for target in TARGETS},
                             **item["pause_statistics"], **acoustic})
        if row_index % 50 == 0 or row_index == len(data[split]):
            atomic_json(cache_path, {"schema_version": 1, "options": VAD_OPTIONS,
                                     "implementation": "silero-vad", "clips": cached})
        if row_index % 100 == 0:
            print(f"[vad][{split}] cached {row_index}/{len(data[split])} clips", flush=True)
    atomic_json(cache_path, {"schema_version": 1, "options": VAD_OPTIONS,
                             "implementation": "silero-vad", "clips": cached})
    atomic_csv(pd.DataFrame(segment_rows), result_dir / "features" / f"{split}_vad_regions.csv")
    feature_frame = pd.DataFrame(feature_rows)
    atomic_csv(feature_frame.drop(columns=list(TARGETS)), result_dir / "features" / f"{split}_pause_features.csv")
    return output, feature_frame


def load_full_cache(config, data, encoder, split):
    path = project_path(config["paths"]["embeddings_dir"]) / encoder / f"{split}_embeddings.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Full-audio SSL cache missing: {path}")
    with np.load(path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(str)
        if "means" in cache and "stds" in cache:
            means, stds = cache["means"].astype(np.float32), cache["stds"].astype(np.float32)
        else:
            pooled = cache["pooled"].astype(np.float32)
            hidden = pooled.shape[-1] // 2
            means, stds = pooled[..., :hidden], pooled[..., hidden:]
    position = {item: i for i, item in enumerate(ids)}
    expected = data[split].id.astype(str).to_numpy()
    missing = [item for item in expected if item not in position]
    if missing:
        raise ValueError(f"{encoder}/{split} embeddings lack {len(missing)} fixed-split rows")
    order = [position[item] for item in expected]
    return {"means": means[order], "stds": stds[order], "ids": expected}


def extract_speech_cache(config, data, split, encoder, segments, extractor, cache_root):
    rows = data[split]
    hidden, ids = [], rows.id.astype(str).to_numpy()
    cache_root.mkdir(parents=True, exist_ok=True)
    for row_index, row in enumerate(rows.itertuples(index=False), start=1):
        clip_id = str(row.id)
        path = project_path(row.audio_path)
        signature = vad_fingerprint(path)
        segment_key = hashlib.sha256(json.dumps(segments[clip_id]).encode()).hexdigest()
        variant = f"silero-vad-v1:{signature}:{segment_key}"
        cache_path = cache_root / f"{hashlib.sha1(clip_id.encode()).hexdigest()}.npz"
        signal, sample_rate = read_wav(path)
        pooled = extractor.extract(signal, sample_rate, cache_path, use_cache=True,
                                   speech_segments=segments[clip_id], cache_variant=variant)
        hidden.append(pooled)
        if row_index % 50 == 0 or row_index == len(rows):
            print(f"[{encoder}][{split}] cached speech embeddings {row_index}/{len(rows)}", flush=True)
    means = np.stack([x[:, :x.shape[-1] // 2] for x in hidden]).astype(np.float32)
    stds = np.stack([x[:, x.shape[-1] // 2:] for x in hidden]).astype(np.float32)
    return {"ids": ids, "means": means, "stds": stds}


def select_layer_info(path):
    source = json.loads(path.read_text(encoding="utf-8"))
    info = {}
    for encoder in ENCODERS:
        for target in TARGETS:
            candidate = source.get(f"{encoder}/{target}/current_mean_std", {})
            layer = candidate.get("layer_index", 0)
            info[(encoder, target)] = int(layer) if str(layer).isdigit() else 0
    return info


def paired_metric_deltas(predictions, replicates, seed):
    """Paired participant-cluster intervals for the planned ablations."""
    rng = np.random.default_rng(seed)
    records = []
    metric_names = ("rmse", "r2", "pearson", "spearman")
    for (encoder, target), frame in predictions.groupby(["encoder", "target"]):
        by_method = {name: part.set_index("id") for name, part in frame.groupby("method")}
        comparisons = (("speech_only_mean_std", "full_audio_mean_std"),
                       ("speech_only_plus_pause", "speech_only_mean_std"),
                       ("speech_only_plus_pause", "speech_only_plus_duration"))
        for method, reference in comparisons:
            if method not in by_method or reference not in by_method:
                continue
            a = by_method[method][["user_id", "actual", "prediction"]].rename(columns={"prediction": "pred_a"})
            b = by_method[reference][["prediction"]].rename(columns={"prediction": "pred_b"})
            paired = a.join(b, how="inner")
            users = paired.user_id.astype(str).unique()
            grouped = {user: paired[paired.user_id.astype(str).eq(user)] for user in users}
            for metric_name in metric_names:
                observed = metrics(paired.actual, paired.pred_a)[metric_name] - metrics(paired.actual, paired.pred_b)[metric_name]
                draws = []
                for _ in range(replicates):
                    sample = rng.choice(users, len(users), replace=True)
                    boot = pd.concat([grouped[user] for user in sample], ignore_index=True)
                    value = metrics(boot.actual, boot.pred_a)[metric_name] - metrics(boot.actual, boot.pred_b)[metric_name]
                    if np.isfinite(value):
                        draws.append(value)
                records.append({"encoder": encoder, "target": target, "method": method,
                    "reference": reference, "metric": metric_name, "delta": observed,
                    "lower_95": float(np.quantile(draws, .025)) if draws else np.nan,
                    "upper_95": float(np.quantile(draws, .975)) if draws else np.nan,
                    "bootstrap_unit": "participant"})
    return pd.DataFrame(records)


def evaluate(Xtr, ytr, Xval, yval, groups, config):
    from sklearn.model_selection import GroupKFold
    folds = min(int(config["modeling"].get("cv_folds", 5)), len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    best = None
    for alpha in config["modeling"]["embedding_ridge_alphas"]:
        fold_scores, oof = [], np.full(len(ytr), np.nan)
        for train_idx, val_idx in splitter.split(Xtr, ytr, groups):
            if set(groups[train_idx]) & set(groups[val_idx]):
                raise AssertionError("Participant overlap in GroupKFold")
            model = build_pipeline(Xtr.iloc[train_idx], "Ridge", {"alpha": float(alpha)},
                                   int(config.get("seed", 42)), standardize=True, stable_ridge=True)
            model.fit(Xtr.iloc[train_idx], ytr.iloc[train_idx])
            oof[val_idx] = model.predict(Xtr.iloc[val_idx])
            fold_scores.append(metrics(ytr.iloc[val_idx], oof[val_idx])["rmse"])
        result = {"alpha": float(alpha), "cv_fold_rmse_mean": float(np.mean(fold_scores)),
                  "cv_fold_rmse_std": float(np.std(fold_scores)), "oof": oof}
        if best is None or result["cv_fold_rmse_mean"] < best["cv_fold_rmse_mean"]:
            best = result
    model = build_pipeline(Xtr, "Ridge", {"alpha": best["alpha"]}, int(config.get("seed", 42)),
                           standardize=True, stable_ridge=True)
    model.fit(Xtr, ytr)
    pred = model.predict(Xval)
    return model, pred, best


def pause_associations(train_pause):
    rows = []
    for target in TARGETS:
        for col in ("duration_seconds", "pause_ratio", "internal_pause_ratio", "number_of_pauses",
                    "mean_pause_duration_seconds", "maximum_pause_duration_seconds", "mean_rms"):
            x = pd.to_numeric(train_pause[col], errors="coerce")
            y = pd.to_numeric(train_pause[target], errors="coerce")
            corr = float(x.corr(y, method="spearman"))
            partial = np.nan
            if col != "duration_seconds" and x.notna().all() and y.notna().all():
                xr = x.rank(method="average").to_numpy(float)
                yr = y.rank(method="average").to_numpy(float)
                dr = train_pause.duration_seconds.rank(method="average").to_numpy(float)
                design = np.column_stack([np.ones(len(dr)), dr])
                rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
                ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
                partial = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) and np.std(ry) else np.nan
            rows.append({"target": target, "feature": col, "spearman_train": corr,
                         "partial_spearman_controlling_duration_train": partial})
    return pd.DataFrame(rows)


def write_report(result_dir, result_frame, delta_frame, associations):
    report = [
        "# VAD and pause pooling diagnostic", "",
        "Validation uses the existing participant-disjoint split. Ridge alpha was selected using training-only GroupKFold by participant (`user_id`); test participants were not scored. Each encoder/target uses its hidden layer selected by the prior train-only SSL sweep. Full-audio rows reproduce the existing baseline.", "",
        "Silero VAD regions measure speech and non-speech acoustically. Leading/trailing non-speech contributes to total non-speech duration and `pause_ratio`; pause counts/duration summaries use internal gaps between speech regions. Clips with no detected speech use a zero speech embedding and have zero speech duration.", "",
        "## Validation metrics", "",
        markdown_table(result_frame[["encoder", "target", "method", "mae", "rmse", "r2", "pearson", "spearman"]]), "",
        "## Paired participant bootstrap", "",
        "RMSE deltas are method minus reference; negative values favor the method. Intervals resample whole validation participants.", "",
        markdown_table(delta_frame[delta_frame.metric.eq("rmse")][["encoder", "target", "method", "reference", "delta", "lower_95", "upper_95"]]), "",
        "## Pause association after duration control", "",
        "Associations below use training data only. Partial Spearman is computed on rank residuals after controlling for duration.", "",
        markdown_table(associations, ".3f"), "",
        "## Interpretation", "",
        "Compare `speech_only_plus_pause` with `speech_only_mean_std` to assess whether adding pauses helps. Compare `speech_only_plus_pause` with `speech_only_plus_duration` to assess whether pause summaries contribute beyond a duration control. Treat these as diagnostic comparisons; small point-estimate differences do not establish a general model winner.", "",
    ]
    (result_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def write_plots(result_dir, result_frame):
    """Write small dependency-free SVG RMSE charts for the validation ablations."""
    from html import escape
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "PLOTS_UNAVAILABLE.txt").unlink(missing_ok=True)
    colors = {"full_audio_mean_std": "#64748b", "speech_only_mean_std": "#2563eb",
              "speech_only_plus_pause": "#0f766e", "speech_only_plus_duration": "#d97706"}
    for target in TARGETS:
        frame = result_frame[result_frame.target.eq(target)]
        width, height, left, right, top, bottom = 1080, 560, 75, 25, 40, 160
        plot_h, max_value = height - top - bottom, max(1.2, float(frame.rmse.max()) * 1.08)
        group_w = (width - left - right) / max(1, len(frame))
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="{left}" y="25" font-family="Arial" font-size="19">Validation RMSE: {escape(target)}</text>']
        for tick in np.linspace(0, max_value, 7):
            y = top + plot_h * (1 - tick / max_value)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#475569">{tick:.2f}</text>')
        bar_w = group_w * .62
        for i, row in enumerate(frame.itertuples(index=False)):
            x = left + i * group_w + (group_w - bar_w) / 2
            bar_h = float(row.rmse) / max_value * plot_h
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{colors[row.method]}"/>')
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{float(row.rmse):.3f}</text>')
            label = f"{row.encoder}: {row.method.replace('_', ' ')}"
            parts.append(f'<text transform="translate({x+bar_w/2:.1f},{top+plot_h+12}) rotate(55)" text-anchor="start" font-family="Arial" font-size="12" fill="#334155">{escape(label)}</text>')
        parts.append('</svg>')
        (plot_dir / f"{target}_rmse.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--encoders", default="wavlm,hubert", help="Comma-separated: wavlm,hubert")
    parser.add_argument("--limit", type=int, default=None, help="Debug only; truncates each split")
    parser.add_argument("--allow-model-download", action="store_true", help="Allow downloading missing SSL model weights")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild pause associations/report from cached experiment results")
    args = parser.parse_args()
    config = load_config(args.config)
    VAD_OPTIONS.update({key: config.get("vad_pause", {}).get(key, value) for key, value in VAD_OPTIONS.items()})
    data = load_split_data(config, args.limit)
    result_dir = ROOT / "results" / "vad_pause"
    if args.finalize_only:
        train_features = pd.read_csv(result_dir / "features" / "train_pause_features.csv", dtype={"id": str})
        train_targets = data["train"][["id", *TARGETS]].copy()
        train_pause = train_features.merge(train_targets, on="id", validate="one_to_one")
        associations = pause_associations(train_pause)
        atomic_csv(associations, result_dir / "pause_target_associations_train.csv")
        results = pd.read_csv(result_dir / "experiment_results.csv")
        deltas = pd.read_csv(result_dir / "paired_participant_deltas.csv")
        write_report(result_dir, results, deltas, associations)
        write_plots(result_dir, results)
        print(f"Updated report and duration-controlled associations in {result_dir}")
        return
    encoders = tuple(value.strip().lower() for value in args.encoders.split(",") if value.strip())
    unknown = set(encoders) - set(ENCODERS)
    if unknown:
        raise ValueError(f"Unknown encoder(s): {sorted(unknown)}")

    import torch
    try:
        from silero_vad import load_silero_vad
    except ImportError as exc:
        raise RuntimeError("Install the lightweight VAD dependency with: python -m pip install silero-vad") from exc
    vad_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vad_model = load_silero_vad(onnx=False).to(vad_device).eval()
    (result_dir / "features").mkdir(parents=True, exist_ok=True)
    vad_segments, pause_frames = {}, {}
    for split in ("train", "val", "test"):
        vad_segments[split], pause_frames[split] = create_vad_cache(config, data, split, vad_model, torch, vad_device, result_dir)
    del vad_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # SSL feature extraction uses CUDA when available; regressions and grouped CV are CPU sklearn pipelines.
    full, speech = {}, {}
    layer_info = select_layer_info(result_dir.parent / "ssl_layer_sweep" / "selected_layers.json")
    for encoder in encoders:
        model_cache = project_path(config["paths"]["cache_dir"]) / "huggingface"
        extractor = TransformerEmbeddingExtractor(encoder, config, model_cache,
                                                   allow_download=args.allow_model_download)
        for split in ("train", "val"):
            full[(encoder, split)] = load_full_cache(config, data, encoder, split)
            cache_root = project_path(config["paths"]["cache_dir"]) / "vad_pause" / "embeddings" / encoder / split
            speech[(encoder, split)] = extract_speech_cache(config, data, split, encoder,
                                                            vad_segments[split], extractor, cache_root)
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    train_pause = pause_frames["train"].set_index("id").loc[data["train"].id.astype(str)].reset_index()
    val_pause = pause_frames["val"].set_index("id").loc[data["val"].id.astype(str)].reset_index()
    # Save split-aligned cached speech embeddings in compact matrices as an indexable aggregate.
    for encoder in encoders:
        for split in ("train", "val"):
            item = speech[(encoder, split)]
            aggregate = project_path(config["paths"]["cache_dir"]) / "vad_pause" / "embeddings" / encoder / f"{split}_speech_mean_std.npz"
            atomic_npz(aggregate, ids=item["ids"], means=item["means"], stds=item["stds"],
                       model_name=np.asarray(config["extractors"][encoder]["model_name"]),
                       metadata=np.asarray(json.dumps({"vad": VAD_OPTIONS, "selected_layer_indices": sorted(set(layer_info[(encoder, target)] for target in TARGETS)), "pooling": "all-layer-mean+population-std"}, sort_keys=True)))

    results, predictions, bootstrap_rows = [], [], []
    groups_train = data["train"].user_id.astype(str).to_numpy()
    val_users = data["val"].user_id.astype(str).to_numpy()
    metrics_to_ci = ("pearson", "spearman", "r2")
    for encoder in encoders:
        for target in TARGETS:
            layer = layer_info[(encoder, target)]
            ytr = pd.to_numeric(data["train"][target], errors="coerce").astype(float)
            yval = pd.to_numeric(data["val"][target], errors="coerce").astype(float)
            names = ("full_audio_mean_std", "speech_only_mean_std", "speech_only_plus_pause",
                     "speech_only_plus_duration")
            feature_matrices = {}
            for split, y in (("train", ytr), ("val", yval)):
                pool = full[(encoder, split)]
                sp = speech[(encoder, split)]
                idx = layer
                a = np.concatenate((pool["means"][:, idx, :], pool["stds"][:, idx, :]), axis=1)
                b = np.concatenate((sp["means"][:, idx, :], sp["stds"][:, idx, :]), axis=1)
                pause = train_pause if split == "train" else val_pause
                pause_cols = [c for c in pause if c in {
                    "speech_duration_seconds", "non_speech_duration_seconds", "pause_ratio",
                    "number_of_pauses", "mean_pause_duration_seconds", "median_pause_duration_seconds",
                    "maximum_pause_duration_seconds", "pauses_over_0_5_seconds", "pauses_over_1_second"}]
                duration = pause[["duration_seconds"]].to_numpy(dtype=float)
                pause_values = pause[pause_cols].to_numpy(dtype=float)
                for name, matrix in ((names[0], a), (names[1], b),
                                     (names[2], np.concatenate((b, pause_values), axis=1)),
                                     (names[3], np.concatenate((b, duration), axis=1))):
                    feature_matrices[(split, name)] = pd.DataFrame(matrix)
            for method in names:
                model, pred, cv = evaluate(feature_matrices[("train", method)], ytr,
                                           feature_matrices[("val", method)], yval, groups_train, config)
                row = {"encoder": encoder, "target": target, "method": method, "layer_index": layer,
                       "alpha": cv["alpha"], "cv_fold_rmse_mean": cv["cv_fold_rmse_mean"],
                       "cv_fold_rmse_std": cv["cv_fold_rmse_std"], **metrics(yval, pred)}
                results.append(row)
                pred_frame = pd.DataFrame({"id": data["val"].id.astype(str), "user_id": val_users,
                                           "actual": yval, "prediction": pred})
                predictions.append(pred_frame.assign(encoder=encoder, target=target, method=method))
                ci = bootstrap_by_participant(yval, pred, val_users,
                    replicates=int(config.get("vad_pause", {}).get("bootstrap_replicates", 1000)),
                    seed=int(config.get("modeling", {}).get("bootstrap_seed", 43)))
                for metric_name in metrics_to_ci:
                    interval = ci.get(metric_name, {})
                    bootstrap_rows.append({"encoder": encoder, "target": target, "method": method,
                                           "metric": metric_name,
                                           "lower_95": interval.get("lower_95", np.nan),
                                           "upper_95": interval.get("upper_95", np.nan)})
                print(f"[{encoder}][{target}] {method} complete: MAE={row['mae']:.4f} RMSE={row['rmse']:.4f} R2={row['r2']:.4f}", flush=True)

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    boot_frame = pd.DataFrame(bootstrap_rows)
    atomic_csv(result_frame, result_dir / "experiment_results.csv")
    atomic_csv(prediction_frame, result_dir / "validation_predictions.csv")
    atomic_csv(boot_frame, result_dir / "participant_bootstrap.csv")
    delta_frame = paired_metric_deltas(prediction_frame,
        int(config.get("vad_pause", {}).get("bootstrap_replicates", 1000)),
        int(config.get("modeling", {}).get("bootstrap_seed", 43)))
    atomic_csv(delta_frame, result_dir / "paired_participant_deltas.csv")
    all_pause = pd.concat([pause_frames[s] for s in ("train", "val", "test")], ignore_index=True)
    atomic_csv(all_pause.drop(columns=list(TARGETS)), result_dir / "features" / "pause_features_all_splits.csv")

    # Correlations use training-only labels; partial duration control is descriptive only.
    associations = pause_associations(train_pause)
    atomic_csv(pd.DataFrame(associations), result_dir / "pause_target_associations_train.csv")

    write_plots(result_dir, result_frame)

    configuration = {"task": "VAD / pause pooling diagnostic", "branch": "audio-diagnostics",
        "vad": VAD_OPTIONS, "vad_library": "silero-vad", "encoders": {e: config["extractors"][e]["model_name"] for e in encoders},
        "targets": list(TARGETS), "fixed_split_method": "authoritative train/val/test participant-disjoint CSVs; evaluation only on held-out val",
        "alpha_selection": "GroupKFold(user_id) on train only; Ridge solver=lsqr; scaler fit inside fold",
        "alpha_grid": config["modeling"]["embedding_ridge_alphas"], "pooling": ["full_audio mean+std", "speech-only mean+std", "speech-only mean+std + requested pause statistics", "speech-only mean+std + duration control"],
        "layer_indices": {f"{e}/{t}": layer_info[(e,t)] for e in encoders for t in TARGETS},
        "cache": "Per-clip VAD and speech embeddings keyed by audio stat, VAD options/segments and encoder cache fingerprint; existing full-audio SSL caches reused",
        "test_used_for_model_fitting_or_scoring": False, "gpu": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "unavailable (CPU SSL extraction)"}
    atomic_json(result_dir / "configuration.json", configuration)
    write_report(result_dir, result_frame, delta_frame, associations)
    print("\nVAD / pause validation comparison\n", result_frame.to_string(index=False), flush=True)
    print(f"Results saved under {result_dir}", flush=True)


if __name__ == "__main__":
    main()
