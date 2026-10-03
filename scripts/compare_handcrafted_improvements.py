"""Train-only grouped-CV comparison of original and enhanced handcrafted features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_audio_models import (  # noqa: E402
    atomic_csv, atomic_json, bootstrap_by_participant, build_pipeline,
    load_config, load_split_data, metrics, project_path, search_model,
)
from scripts.extract_audio_features import read_wav  # noqa: E402

TARGETS = ("confidence_score", "speaking_skills", "overall_performance")
SETS = ("A_original", "B_improved_only", "C_original_plus_new")
MODELS = ("Ridge", "SVR", "ExtraTrees", "RandomForest")
EXCLUDED = {"id", "user_id", "user_no", "participant_id", "question_id", "split", "file_name",
            "audio_path", "original_video_path", "confidence", "confidence_score", "speaking_skills",
            "overall_performance", "interview_score", "answer_score", "openness", "conscientiousness",
            "extraversion", "agreeableness", "neuroticism", "overall_personality", "facial_expression",
            "feature_status", "feature_error", "status", "sample_rate_hz"}
TIME_SEGMENT = re.compile(r"\[(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\]\s*(.*?)(?=\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]|$)", re.S)
WORD = re.compile(r"\b[\w']+\b", re.UNICODE)
FEATURE_VERSION = "handcrafted-improvement-v1.0"


def time_seconds(minutes, seconds):
    return int(minutes) * 60 + int(seconds)


def parse_timed_transcript(transcript):
    """Return token count and coarse timed speech duration; no syllables inferred."""
    segments = []
    for match in TIME_SEGMENT.finditer(str(transcript or "")):
        start = time_seconds(match.group(1), match.group(2))
        end = time_seconds(match.group(3), match.group(4))
        tokens = WORD.findall(match.group(5))
        if end > start and tokens:
            segments.append((float(start), float(end), len(tokens)))
    if not segments:
        return {"timed_transcript_words": np.nan, "timed_speech_seconds": np.nan,
                "transcript_speech_rate_wpm": np.nan, "articulation_rate_wpm": np.nan}
    words = sum(item[2] for item in segments)
    speaking_seconds = sum(item[1] - item[0] for item in segments)
    return {"timed_transcript_words": float(words), "timed_speech_seconds": float(speaking_seconds),
            "transcript_speech_rate_wpm": np.nan, "articulation_rate_wpm": 60.0 * words / speaking_seconds}


def frame_activity(times, regions):
    active = np.zeros(len(times), dtype=bool)
    for start, end in regions:
        active |= (times >= float(start)) & (times < float(end))
    return active


def f0_track(signal, sample_rate, duration, regions):
    """Autocorrelation F0 with lag interpolation, restricted to VAD speech frames."""
    size, hop = int(.040 * sample_rate), int(.010 * sample_rate)
    if len(signal) < size:
        signal = np.pad(signal, (0, size - len(signal)))
    starts = np.arange(0, max(1, len(signal) - size + 1), hop)
    windows = np.lib.stride_tricks.sliding_window_view(signal, size)[::hop][:len(starts)]
    times = (starts + size / 2) / sample_rate
    speech = frame_activity(times, regions)
    f0 = np.full(len(windows), np.nan, dtype=float)
    low = max(2, int(sample_rate / 400.0))
    high = min(size - 2, int(sample_rate / 65.0))
    taper = np.hanning(size)
    fft_size = 1 << (2 * size - 1).bit_length()
    for index in np.flatnonzero(speech):
        x = windows[index].astype(float)
        x = (x - x.mean()) * taper
        energy = float(x @ x)
        if energy < 1e-9:
            continue
        spectrum = np.fft.rfft(x, n=fft_size)
        correlation = np.fft.irfft(spectrum * spectrum.conjugate(), n=fft_size)[:high + 2]
        if correlation[0] <= 0:
            continue
        normalized = correlation / correlation[0]
        candidates = np.flatnonzero((normalized[low + 1:high] >= normalized[low:high - 1]) &
                                    (normalized[low + 1:high] > normalized[low + 2:high + 1])) + low + 1
        candidates = candidates[normalized[candidates] >= .35]
        if not len(candidates):
            continue
        lag = float(candidates[0])
        if 1 <= int(lag) < len(normalized) - 1:
            left, center, right = normalized[int(lag)-1:int(lag)+2]
            denominator = left - 2 * center + right
            if abs(denominator) > 1e-12:
                lag += .5 * (left - right) / denominator
        estimate = sample_rate / lag
        if 65 <= estimate <= 400:
            f0[index] = estimate
    return times, speech, f0


def slope(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2 or np.ptp(x[valid]) <= 0:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], 1)[0])


def acoustic_features(signal, sample_rate, regions):
    duration = len(signal) / sample_rate
    rms_size, rms_hop = int(.025 * sample_rate), int(.010 * sample_rate)
    if len(signal) < rms_size:
        signal = np.pad(signal, (0, rms_size - len(signal)))
    starts = np.arange(0, max(1, len(signal) - rms_size + 1), rms_hop)
    windows = np.lib.stride_tricks.sliding_window_view(signal, rms_size)[::rms_hop][:len(starts)]
    times = (starts + rms_size / 2) / sample_rate
    activity = frame_activity(times, regions)
    rms = np.sqrt(np.mean(windows.astype(float) ** 2, axis=1))
    voiced_rms = rms[activity]
    t_f0, f0_speech, f0 = f0_track(signal, sample_rate, duration, regions)
    voiced = np.isfinite(f0)
    median_f0 = float(np.median(f0[voiced])) if voiced.any() else np.nan
    relative = np.full(len(f0), np.nan)
    if np.isfinite(median_f0) and median_f0 > 0:
        relative[voiced] = 12 * np.log2(f0[voiced] / median_f0)
    f0_values = f0[voiced]
    rms_mean = float(np.mean(voiced_rms)) if len(voiced_rms) else np.nan
    rms_std = float(np.std(voiced_rms)) if len(voiced_rms) else np.nan
    q05, q95 = np.quantile(voiced_rms, [.05, .95]) if len(voiced_rms) else (np.nan, np.nan)
    rms_db = 20 * np.log10(np.maximum(voiced_rms, 1e-8)) if len(voiced_rms) else np.asarray([])
    relative_pitch = relative[voiced]
    valid_rate = int(voiced.sum()) / max(1, int(f0_speech.sum()))
    features = {
        "speech_duration_seconds": float(sum(max(0.0, b-a) for a, b in regions)),
        "non_speech_duration_seconds": max(0.0, duration - sum(max(0.0, b-a) for a, b in regions)),
        "f0_mean_hz": float(np.mean(f0_values)) if len(f0_values) else np.nan,
        "f0_std_hz": float(np.std(f0_values)) if len(f0_values) else np.nan,
        "f0_min_hz": float(np.min(f0_values)) if len(f0_values) else np.nan,
        "f0_max_hz": float(np.max(f0_values)) if len(f0_values) else np.nan,
        "f0_range_hz": float(np.ptp(f0_values)) if len(f0_values) else np.nan,
        "voiced_percentage_of_speech": float(100 * valid_rate),
        "pitch_slope_semitones_per_second": slope(t_f0[voiced], relative[voiced]),
        "final_contour_slope_semitones_per_second": slope(t_f0[voiced & (t_f0 >= 2*duration/3)], relative[voiced & (t_f0 >= 2*duration/3)]),
        "relative_pitch_mean_semitones": float(np.mean(relative_pitch)) if len(relative_pitch) else np.nan,
        "relative_pitch_std_semitones": float(np.std(relative_pitch)) if len(relative_pitch) else np.nan,
        "mean_rms": rms_mean, "rms_std": rms_std,
        "rms_dynamic_range_db_p95_p05": float(20*np.log10(max(q95, 1e-8)/max(q05, 1e-8))) if len(voiced_rms) else np.nan,
        "energy_slope_db_per_second": slope(times[activity], rms_db) if len(rms_db) == int(activity.sum()) else np.nan,
    }
    # Delivery dynamics by thirds: relative pitch, voiced fraction, and speech RMS only.
    for part, (lo, hi) in enumerate(((0.0, 1/3), (1/3, 2/3), (2/3, 1.000001)), start=1):
        in_third = (t_f0 / max(duration, 1e-9) >= lo) & (t_f0 / max(duration, 1e-9) < hi)
        third_voice = in_third & f0_speech
        third_rms = activity & (times / max(duration, 1e-9) >= lo) & (times / max(duration, 1e-9) < hi)
        third_pitch = in_third & voiced
        features[f"third_{part}_relative_pitch_mean_st"] = float(np.mean(relative[third_pitch])) if third_pitch.any() else np.nan
        features[f"third_{part}_voiced_percentage"] = float(100 * voiced[third_pitch].sum() / max(1, third_voice.sum())) if third_voice.any() else np.nan
        features[f"third_{part}_mean_rms"] = float(np.mean(rms[third_rms])) if third_rms.any() else np.nan
    return features


def new_features_for_clip(wave_path, transcript, vad_item):
    signal, sample_rate = read_wav(wave_path)
    duration = len(signal) / sample_rate
    regions = [(float(a), float(b)) for a, b in vad_item["segments"] if b > a]
    acoustic = acoustic_features(signal, sample_rate, regions)
    timed = parse_timed_transcript(transcript)
    words = timed["timed_transcript_words"]
    timed["transcript_speech_rate_wpm"] = 60 * words / duration if np.isfinite(words) and duration > 0 else np.nan
    # Pause gaps are only internal non-speech intervals bounded by detected speech.
    pauses = [max(0.0, regions[i+1][0] - regions[i][1]) for i in range(len(regions)-1)]
    pauses = [value for value in pauses if value > 0]
    internal_pause = float(sum(pauses))
    denominator = max(0.0, duration - internal_pause)
    timed["speech_rate_excluding_pauses_wpm"] = 60 * words / denominator if np.isfinite(words) and denominator > 0 else np.nan
    internal_ratio = internal_pause / duration if duration > 0 else np.nan
    features = {**acoustic,
        "transcript_speech_rate_wpm": timed["transcript_speech_rate_wpm"],
        "articulation_rate_wpm": timed["articulation_rate_wpm"],
        "speech_rate_excluding_pauses_wpm": timed["speech_rate_excluding_pauses_wpm"],
        "pause_count": len(pauses), "pause_ratio": internal_ratio,
        "mean_pause_duration_seconds": float(np.mean(pauses)) if pauses else 0.0,
        "median_pause_duration_seconds": float(np.median(pauses)) if pauses else 0.0,
        "longest_pause_seconds": float(max(pauses)) if pauses else 0.0,
        "pauses_over_0_5_seconds": int(sum(value > .5 for value in pauses)),
        "pauses_over_1_second": int(sum(value > 1.0 for value in pauses)),
        "non_speech_ratio_including_edges": acoustic["non_speech_duration_seconds"] / duration if duration > 0 else np.nan,
    }
    return {f"new_{key}": value for key, value in features.items()}


def timed_transcripts(split):
    path = ROOT / "Datasets" / "Split Dataset" / f"{split}.csv"
    table = pd.read_csv(path, dtype={"id": str, "user_no": str})
    if "transcript" not in table:
        return {}
    return dict(zip(table.id.astype(str), table.transcript.fillna("")))


def cache_signature(audio_path, transcript, vad_item):
    stat = audio_path.stat()
    payload = {"version": FEATURE_VERSION, "audio_size": stat.st_size, "audio_mtime_ns": stat.st_mtime_ns,
        "transcript_hash": hashlib.sha256(str(transcript or "").encode()).hexdigest(),
        "vad_segments": vad_item.get("segments", [])}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_feature_table(config, split, metadata, result_dir):
    vad_path = project_path(config["paths"]["cache_dir"]) / "vad_pause" / "vad" / f"{split}.json"
    if not vad_path.is_file():
        raise FileNotFoundError(f"Missing cached Silero VAD regions: {vad_path}. Run scripts/run_vad_pause.py first.")
    vad_payload = json.loads(vad_path.read_text(encoding="utf-8"))
    vad_clips = vad_payload["clips"]
    transcripts = timed_transcripts(split)
    cache_dir = project_path(config["paths"]["cache_dir"]) / "handcrafted_improvements"
    cache_path = cache_dir / f"{split}.json"
    previous = {}
    if cache_path.is_file():
        try:
            previous = json.loads(cache_path.read_text(encoding="utf-8")).get("clips", {})
        except (OSError, json.JSONDecodeError):
            previous = {}
    output = []
    cached_items = dict(previous)
    for row_index, row in enumerate(metadata.itertuples(index=False), start=1):
        clip_id = str(row.id)
        audio_path = project_path(row.audio_path)
        if clip_id not in vad_clips:
            raise KeyError(f"VAD cache has no clip {clip_id} in {split}")
        transcript = transcripts.get(clip_id, "")
        signature = cache_signature(audio_path, transcript, vad_clips[clip_id])
        cached = previous.get(clip_id, {})
        if cached.get("signature") == signature:
            features = cached["features"]
        else:
            features = new_features_for_clip(audio_path, transcript, vad_clips[clip_id])
            cached_items[clip_id] = {"signature": signature, "features": features}
        output.append({"id": clip_id, "user_id": str(row.user_id), "question_id": str(row.question_id), **features})
        if row_index % 100 == 0 or row_index == len(metadata):
            cache_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(cache_path, {"schema_version": 1, "feature_version": FEATURE_VERSION,
                                     "vad_options": vad_payload.get("options"), "clips": cached_items})
            print(f"[features][{split}] {row_index}/{len(metadata)} clips", flush=True)
    frame = pd.DataFrame(output)
    # Outputs are identifiers plus predictors only; targets are joined only in-memory for modeling.
    atomic_csv(frame.drop(columns=["user_id", "question_id"]), result_dir / "features" / f"{split}_new_features.csv")
    return frame


def load_original(split, config, metadata):
    path = project_path(config["paths"]["legacy_features_dir"]) / f"{split}_features.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Original handcrafted baseline table missing: {path}")
    frame = pd.read_csv(path, dtype={"id": str})
    if frame.id.duplicated().any():
        raise ValueError(f"Duplicate clip IDs in original feature table {path}")
    expected = metadata.id.astype(str).tolist()
    index = frame.set_index(frame.id.astype(str))
    if any(clip not in index.index for clip in expected):
        raise ValueError(f"Original feature table {split} does not cover fixed split clips")
    aligned = index.loc[expected].reset_index(drop=True)
    columns = [col for col in aligned.columns if col.lower() not in EXCLUDED and pd.api.types.is_numeric_dtype(aligned[col])]
    x = aligned[columns].apply(pd.to_numeric, errors="coerce")
    return x, columns


def feature_sets(original, improved):
    a = original.reset_index(drop=True)
    b = improved.reset_index(drop=True)
    c = pd.concat([a, b], axis=1)
    return {"A_original": a, "B_improved_only": b, "C_original_plus_new": c}


def assert_predictor_safety(frame, config, label):
    forbidden = set(EXCLUDED) | set(config["targets"].keys()) | set(config["targets"].values())
    bad = sorted(set(frame.columns) & {str(value).lower() for value in forbidden})
    if bad:
        raise AssertionError(f"Target/identity/meta columns present in {label}: {bad}")
    if not all(pd.api.types.is_numeric_dtype(frame[col]) for col in frame.columns):
        raise AssertionError(f"Non-numeric input columns present in {label}")


def paired_rmse_intervals(predictions, replicates, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for (target, model), frame in predictions.groupby(["target", "model"]):
        by_set = {name: part.set_index("id") for name, part in frame.groupby("feature_set")}
        for left, right in (("B_improved_only", "A_original"), ("C_original_plus_new", "A_original"),
                            ("C_original_plus_new", "B_improved_only")):
            if left not in by_set or right not in by_set:
                continue
            a = by_set[left][["user_id", "actual", "prediction"]].rename(columns={"prediction": "pred_a"})
            b = by_set[right][["prediction"]].rename(columns={"prediction": "pred_b"})
            paired = a.join(b, how="inner")
            participants = paired.user_id.astype(str).unique()
            grouped = {user: paired[paired.user_id.astype(str).eq(user)] for user in participants}
            delta = metrics(paired.actual, paired.pred_a)["rmse"] - metrics(paired.actual, paired.pred_b)["rmse"]
            draws = []
            for _ in range(replicates):
                chosen = rng.choice(participants, len(participants), replace=True)
                boot = pd.concat([grouped[user] for user in chosen], ignore_index=True)
                value = metrics(boot.actual, boot.pred_a)["rmse"] - metrics(boot.actual, boot.pred_b)["rmse"]
                if np.isfinite(value):
                    draws.append(value)
            rows.append({"target": target, "model": model, "feature_set": left, "reference": right,
                         "rmse_delta": delta, "lower_95": np.quantile(draws, .025),
                         "upper_95": np.quantile(draws, .975), "bootstrap_unit": "participant"})
    return pd.DataFrame(rows)


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(map(str, columns)) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [format(float(v), ".4f") if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_plots(result_dir, results):
    from html import escape
    path = result_dir / "plots"
    path.mkdir(parents=True, exist_ok=True)
    colors = {"A_original": "#64748b", "B_improved_only": "#2563eb", "C_original_plus_new": "#0f766e"}
    for target in TARGETS:
        frame = results[results.target.eq(target)]
        width, height, left, right, top, bottom = 1150, 600, 70, 25, 40, 165
        max_y = max(float(frame.rmse.max()) * 1.08, .1)
        ph = height - top - bottom
        gw = (width-left-right) / max(1, len(frame))
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
                 '<rect width="100%" height="100%" fill="white"/>',
                 f'<text x="{left}" y="24" font-family="Arial" font-size="18">Validation RMSE: {escape(target)}</text>']
        for tick in np.linspace(0, max_y, 7):
            y = top + ph * (1-tick/max_y)
            parts += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>',
                      f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>']
        bw = gw * .62
        for i, row in enumerate(frame.itertuples(index=False)):
            x = left + i*gw + (gw-bw)/2
            bh = row.rmse/max_y*ph
            y = top + ph-bh
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{colors[row.feature_set]}"/>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{row.rmse:.3f}</text>')
            label = f"{row.model}: {row.feature_set}"
            parts.append(f'<text transform="translate({x+bw/2:.1f},{top+ph+10}) rotate(55)" font-family="Arial" font-size="11">{escape(label)}</text>')
        parts.append("</svg>")
        (path / f"{target}_rmse.svg").write_text("\n".join(parts), encoding="utf-8")


def write_report(result_dir, results, deltas, dimensions):
    report = ["# Handcrafted feature improvement comparison", "",
        "All validation results use the existing participant-disjoint split. Hyperparameters were chosen by train-only `GroupKFold(user_id)`; the test split was not scored. Participant-cluster bootstrap intervals and paired RMSE deltas are saved separately.", "",
        "## Feature sets", "",
        markdown_table(pd.DataFrame([{"set": name, "dimensions": count} for name, count in dimensions.items()])), "",
        "## Validation results", "",
        markdown_table(results[["feature_set", "model", "target", "mae", "rmse", "r2", "pearson", "spearman"]].sort_values(["target", "model", "feature_set"])), "",
        "## Paired RMSE deltas", "",
        "Delta is feature-set RMSE minus reference RMSE; negative favors the first set. Confidence intervals resample participants.", "",
        markdown_table(deltas.sort_values(["target", "model", "feature_set"])), "",
        "## Feature definitions and limitations", "",
        "New features include transcript word rates only when timed segments are present; acoustic pause count/ratio/duration summaries from internal Silero speech gaps; F0 level/range, voiced percentage, semitone-relative contour slopes; speech RMS level/variability/dynamic range/trend; and mean relative pitch, voiced percentage, and RMS over each answer third. No syllable counts are derived. Semitone normalization uses each recording’s median voiced F0, not participant identity or a cross-clip identity baseline.", "",
        "All original baseline columns remain in A and C, including jitter/shimmer proxies. There was no HNR feature in the existing table. No existing features were dropped. New exports contain predictor values and clip IDs only; identifiers and target values are excluded from all model matrices.", "",
        "Limitations: transcript times are coarse `[MM:SS - MM:SS]` segments and unavailable for some recordings; F0/autocorrelation can produce errors on noisy or low-volume speech; VAD can miss very quiet speech; jitter/shimmer remain uncalibrated proxies; the conclusions are limited to this validation cohort.", ""]
    (result_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Debug only: truncate each fixed split")
    parser.add_argument("--finalize-only", action="store_true", help="Rebuild SVG plots and report from saved experiment results")
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_split_data(config, args.limit)
    result_dir = ROOT / "results" / "handcrafted_improvements"
    if args.finalize_only:
        results = pd.read_csv(result_dir / "experiment_results.csv")
        deltas = pd.read_csv(result_dir / "paired_participant_deltas.csv")
        dimension_table = pd.read_csv(result_dir / "feature_dimensions.csv")
        dimensions = dict(zip(dimension_table.feature_set, dimension_table.dimensions.astype(int)))
        write_plots(result_dir, results)
        write_report(result_dir, results, deltas, dimensions)
        print(f"Updated report and plots in {result_dir}")
        return
    (result_dir / "features").mkdir(parents=True, exist_ok=True)

    new_tables, original_tables, original_columns = {}, {}, {}
    for split in ("train", "val", "test"):
        metadata = data[split]
        new_tables[split] = make_feature_table(config, split, metadata, result_dir)
        original_tables[split], original_columns[split] = load_original(split, config, metadata)
    dimensions = {"A_original": len(original_columns["train"]),
                  "B_improved_only": len([col for col in new_tables["train"].columns if col.startswith("new_")]),
                  "C_original_plus_new": len(original_columns["train"]) + len([col for col in new_tables["train"].columns if col.startswith("new_")])}

    results, predictions, bootstrap_rows, cv_rows = [], [], [], []
    groups = data["train"].user_id.astype(str).to_numpy()
    val_users = data["val"].user_id.astype(str).to_numpy()
    experiment_sets = {}
    for split in ("train", "val"):
        improved = new_tables[split].filter(regex="^new_").reset_index(drop=True)
        experiment_sets[split] = feature_sets(original_tables[split], improved)

    for target_index, target in enumerate(TARGETS):
        ytrain = pd.to_numeric(data["train"][target], errors="coerce").astype(float)
        yval = pd.to_numeric(data["val"][target], errors="coerce").astype(float)
        train_valid, val_valid = ytrain.notna(), yval.notna()
        ytrain, yval = ytrain[train_valid].reset_index(drop=True), yval[val_valid].reset_index(drop=True)
        train_groups = groups[train_valid.to_numpy()]
        validation_users = val_users[val_valid.to_numpy()]
        for set_index, feature_set in enumerate(SETS):
            Xtrain = experiment_sets["train"][feature_set].loc[train_valid].reset_index(drop=True)
            Xval = experiment_sets["val"][feature_set].loc[val_valid].reset_index(drop=True)
            assert_predictor_safety(Xtrain, config, f"{feature_set} training X for {target}")
            assert_predictor_safety(Xval, config, f"{feature_set} validation X for {target}")
            if list(Xtrain.columns) != list(Xval.columns):
                raise AssertionError(f"Feature schema mismatch for {feature_set}")
            for model_index, model_name in enumerate(MODELS):
                seed = int(config.get("seed", 42)) + target_index * 100 + model_index
                record, _, trials = search_model(Xtrain, ytrain, train_groups, model_name, config,
                    int(config["modeling"].get("cv_folds", 5)), seed, standardize=True,
                    n_jobs=4, stable_ridge=(model_name == "Ridge"))
                for trial in trials:
                    cv_rows.append({"target": target, "feature_set": feature_set, **trial})
                if record is None:
                    results.append({"target": target, "feature_set": feature_set, "model": model_name,
                                    "feature_dimensions": Xtrain.shape[1], "error": "all CV candidates failed"})
                    print(f"[{feature_set}][{target}] {model_name} failed all CV candidates", flush=True)
                    continue
                params = record["params"]
                pipeline = build_pipeline(Xtrain, model_name, params, seed, standardize=True,
                                          n_jobs=4, stable_ridge=(model_name == "Ridge"))
                pipeline.fit(Xtrain, ytrain)
                pred = pipeline.predict(Xval)
                scores = metrics(yval, pred)
                row = {"target": target, "feature_set": feature_set, "model": model_name,
                       "feature_dimensions": Xtrain.shape[1], "params": json.dumps(params, sort_keys=True),
                       "cv_rmse_fold_mean": record["cv_rmse_fold_mean"],
                       "cv_rmse_fold_std": record["cv_rmse_fold_std"], "error": "", **scores}
                results.append(row)
                pred_frame = pd.DataFrame({"id": data["val"].loc[val_valid, "id"].astype(str).to_numpy(),
                    "user_id": validation_users, "actual": yval, "prediction": pred})
                predictions.append(pred_frame.assign(target=target, feature_set=feature_set, model=model_name))
                ci = bootstrap_by_participant(yval, pred, validation_users,
                    replicates=int(config["modeling"].get("bootstrap_replicates", 1000)),
                    seed=int(config["modeling"].get("bootstrap_seed", 43)) + seed)
                for metric_name, interval in ci.items():
                    bootstrap_rows.append({"target": target, "feature_set": feature_set,
                        "model": model_name, "metric": metric_name,
                        "lower_95": interval["lower_95"], "upper_95": interval["upper_95"]})
                print(f"[{feature_set}][{target}] {model_name} complete: MAE={scores['mae']:.4f} RMSE={scores['rmse']:.4f} R2={scores['r2']:.4f}", flush=True)

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    delta_frame = paired_rmse_intervals(prediction_frame,
        int(config["modeling"].get("bootstrap_replicates", 1000)),
        int(config["modeling"].get("bootstrap_seed", 43))) if len(prediction_frame) else pd.DataFrame()
    atomic_csv(result_frame, result_dir / "experiment_results.csv")
    atomic_csv(prediction_frame, result_dir / "validation_predictions.csv")
    atomic_csv(pd.DataFrame(bootstrap_rows), result_dir / "participant_bootstrap.csv")
    atomic_csv(delta_frame, result_dir / "paired_participant_deltas.csv")
    atomic_csv(pd.DataFrame([{"feature_set": name, "dimensions": count,
        "feature_names": json.dumps(list(experiment_sets["train"][name].columns))} for name, count in dimensions.items()]),
        result_dir / "feature_dimensions.csv")
    atomic_csv(pd.DataFrame(cv_rows), result_dir / "grouped_cv_trials.csv")
    metadata = {"task": "handcrafted feature improvement comparison", "feature_version": FEATURE_VERSION,
        "targets": list(TARGETS), "feature_sets": dimensions,
        "original_features": list(original_columns["train"]),
        "new_features": [col for col in experiment_sets["train"]["B_improved_only"].columns],
        "models": list(MODELS), "fixed_splits": "existing participant-disjoint train/validation/test split",
        "selection": "GroupKFold(user_id), training only; validation used only for final scores and participant bootstrap",
        "test_used_for_model_fitting_or_scoring": False,
        "transcript_timing": "Only parses supplied [MM:SS - MM:SS] segments; rates are word-based; no syllable counts invented",
        "pitch_normalization": "Semitone relative to within-recording median F0; participant ID is excluded from every predictor",
        "retained_voice_quality_features": [x for x in original_columns["train"] if "jitter" in x.lower() or "shimmer" in x.lower() or "hnr" in x.lower()],
        "hnr_note": "No HNR column existed in the original handcrafted feature table; none was removed or fabricated",
        "limitations": ["Timed transcript segments are coarse and missing for some clips", "F0 and voiced fraction depend on Silero segmentation and autocorrelation pitch tracking", "Jitter/shimmer retained as existing proxies, not clinical cycle-level estimates", "Only one participant-disjoint validation cohort; tiny metric deltas are not general evidence"]}
    atomic_json(result_dir / "configuration.json", metadata)
    write_plots(result_dir, result_frame)
    write_report(result_dir, result_frame, delta_frame, dimensions)
    print("\nHandcrafted validation comparison\n", result_frame.to_string(index=False), flush=True)
    print(f"Results saved under {result_dir}", flush=True)


if __name__ == "__main__":
    main()
