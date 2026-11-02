"""Shared aggregation: turns frame-level (or 1 Hz) time series into one feature
row per video, following the paper's visual features (Sec. 5.3):
  - expression features: proportion of time active, number of active segments,
    variance of the signal
  - head pose: variance / max / min of yaw, pitch, roll
  - head motion: displacement statistics (here: frame-to-frame change)
We additionally add a slope (trend over time) per signal, which the paper did
not use but is cheap and captures "building/fading" behaviour.

Length note: `*_max`, `*_min` and `*_active_segments` grow with video length (more frames =
more chances for an extreme value / another episode). `*_p90`, `*_p10` and `*_seg_rate`
are length-insensitive equivalents; train_v2.py can compare the two feature families.
"""
import re
import numpy as np
import pandas as pd


def _runs_above(x: np.ndarray, thr: float) -> int:
    """Number of contiguous segments where x > thr (paper's 'active segments')."""
    active = x > thr
    if active.size == 0:
        return 0
    starts = active & ~np.concatenate([[False], active[:-1]])
    return int(starts.sum())


def _slope(t, x) -> float:
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    if ok.sum() < 3 or np.nanstd(t[ok]) == 0:
        return np.nan
    return float(np.polyfit(t[ok], x[ok], 1)[0])


def summarize_activation(name: str, t, x, thr: float) -> dict:
    """For emotion/AU-like signals."""
    x = np.asarray(x, dtype=float)
    return {
        f"{name}_mean": np.nanmean(x),
        f"{name}_var": np.nanvar(x),
        f"{name}_max": np.nanmax(x),
        f"{name}_p90": np.nanpercentile(x, 90),            # robust to video length
        f"{name}_active_frac": np.nanmean(x > thr),
        f"{name}_active_segments": _runs_above(np.nan_to_num(x), thr),
        f"{name}_seg_rate": _runs_above(np.nan_to_num(x), thr) / max(len(x), 1),  # per sample
        f"{name}_slope": _slope(t, x),
    }


def summarize_pose(name: str, t, x) -> dict:
    """For head pose angles (yaw/pitch/roll)."""
    x = np.asarray(x, dtype=float)
    return {
        f"{name}_mean": np.nanmean(x),
        f"{name}_var": np.nanvar(x),
        f"{name}_max": np.nanmax(x),
        f"{name}_min": np.nanmin(x),
        f"{name}_p90": np.nanpercentile(x, 90),
        f"{name}_p10": np.nanpercentile(x, 10),
        f"{name}_motion": np.nanmean(np.abs(np.diff(x))) if len(x) > 1 else np.nan,
    }


def summarize_continuous(name: str, t, x) -> dict:
    """For valence/arousal/gaze/other continuous signals."""
    x = np.asarray(x, dtype=float)
    return {
        f"{name}_mean": np.nanmean(x),
        f"{name}_std": np.nanstd(x),
        f"{name}_slope": _slope(t, x),
        f"{name}_motion": np.nanmean(np.abs(np.diff(x))) if len(x) > 1 else np.nan,
    }


# --- column classification for Py-Feat Detectorv2 output --------------------
EMOTION_NAMES = {"anger", "disgust", "fear", "happiness", "happy", "sadness",
                 "sad", "surprise", "neutral", "contempt"}


def classify_columns(df: pd.DataFrame):
    """Split numeric columns into groups by name. Prints the groups so you can
    verify them against your installed Py-Feat version."""
    emo, aus, pose, cont = [], [], [], []
    for c in df.select_dtypes("number").columns:
        lc = c.lower()
        if lc in ("frame", "time_sec", "facescore"):
            continue
        if lc in EMOTION_NAMES:
            emo.append(c)
        elif re.match(r"^au\d+", lc):
            aus.append(c)
        elif lc in ("pitch", "yaw", "roll"):
            pose.append(c)
        elif lc in ("valence", "arousal") or "gaze" in lc:
            cont.append(c)
        # everything else (face box, mesh, blendshapes, identity) is ignored on purpose
    return emo, aus, pose, cont


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """True for sampled frames where the detector produced real values. Py-Feat returns an
    all-NaN row for a frame with no face, so len(df) over-counts usable frames."""
    emo, aus, pose, _ = classify_columns(df)
    cols = emo + aus + pose
    if not cols:
        return pd.Series(True, index=df.index)
    return df[cols].notna().any(axis=1)


def aggregate_video(df: pd.DataFrame, act_thr: float = 0.5, verbose: bool = False) -> dict:
    """df must have a `time_sec` column plus Py-Feat output columns."""
    t = df["time_sec"].to_numpy(dtype=float)
    emo, aus, pose, cont = classify_columns(df)
    if verbose:
        print(f"  emotions={emo}\n  AUs={aus}\n  pose={pose}\n  continuous={cont}")
    row = {}
    for c in emo + aus:
        row.update(summarize_activation(c, t, df[c], act_thr))
    for c in pose:
        row.update(summarize_pose(c, t, df[c]))
    for c in cont:
        row.update(summarize_continuous(c, t, df[c]))
    # frame-level emotion variance, as in the paper (dominant-emotion intensity)
    if emo:
        dom = df[emo].max(axis=1).to_numpy(dtype=float)
        row["dominant_emotion_var"] = np.nanvar(dom)
    # fraction of time roughly frontal (|yaw|,|pitch| small) - paper's fracFrontal
    lower = {c.lower(): c for c in pose}
    if "yaw" in lower and "pitch" in lower:
        yaw, pitch = df[lower["yaw"]], df[lower["pitch"]]
        # Py-Feat versions differ: values this small mean radians, so convert to degrees
        if np.nanpercentile(np.abs(np.r_[yaw.to_numpy(), pitch.to_numpy()]), 99) < 2.0:
            yaw, pitch = np.degrees(yaw), np.degrees(pitch)
        row["frac_frontal"] = float(((yaw.abs() < 15) & (pitch.abs() < 15)).mean())
    row["n_samples"] = len(df)
    return row
