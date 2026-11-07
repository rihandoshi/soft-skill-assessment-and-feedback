"""
build_features_v3.py  -  richer per-video features from the EXISTING Py-Feat frame cache.

No re-extraction needed: reads frame_features/recruitview/<video>.csv (already on disk) and writes
features_recruitview_v3.csv (one row per video, keyed by file_name).

What is new compared with features_recruitview.csv (aggregate.py):
  * 52 MediaPipe/ARKit blendshapes that Detectorv2 already outputs but were never used
    (eyeLookDown/Up/In/Out = eye-gaze direction, eyeBlink = blinks, mouthSmile, browInnerUp, jawOpen ...).
  * Framing / proximity: face width / frame width, face centre offset, their stability.
  * Head-motion energy from the 3D head position (X, Y, Z) and pose (Pitch/Yaw/Roll).
  * Robust stats for every channel: mean, std, p10, p50, p90, mean |first difference| (movement),
    plus share of frames above a channel-specific "active" level (train-free: uses fixed thresholds).
  * Blink / gaze-aversion rates per second (length-normalised, not counts).
Identity embeddings and raw landmark/mesh coordinates are deliberately NOT used (appearance / fairness risk).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "frame_features" / "recruitview"
OUT = ROOT / "features_recruitview_v3.csv"

AUS = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12", "AU14",
       "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"]
EMO = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
VA = ["valence", "arousal"]
POSE = ["Pitch", "Roll", "Yaw"]
GAZE = ["gaze_pitch", "gaze_yaw", "gaze_angle"]
BLEND = ['browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff',
         'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
         'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
         'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight',
         'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
         'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
         'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
         'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft',
         'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
         'noseSneerLeft', 'noseSneerRight']
OTHER = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore", "X", "Y", "Z",
         "FrameHeight", "FrameWidth", "time_sec"]
USECOLS = AUS + EMO + VA + POSE + GAZE + BLEND + OTHER


def stats(prefix, x, out, active_thr=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return
    out[f"{prefix}_mean"] = x.mean()
    out[f"{prefix}_std"] = x.std() if len(x) > 1 else 0.0
    out[f"{prefix}_p10"], out[f"{prefix}_p50"], out[f"{prefix}_p90"] = np.percentile(x, [10, 50, 90])
    out[f"{prefix}_move"] = np.abs(np.diff(x)).mean() if len(x) > 1 else 0.0
    if active_thr is not None:
        out[f"{prefix}_act"] = (x > active_thr).mean()


def video_features(df):
    out = {}
    df = df[[c for c in USECOLS if c in df.columns]].copy()
    valid = df[AUS].notna().any(axis=1)
    out["v3_n_frames_total"] = len(df)
    out["v3_face_rate"] = valid.mean() if len(df) else np.nan
    df = df[valid]
    out["v3_n_valid"] = len(df)
    if len(df) == 0:
        return out
    for c in AUS:
        stats(c, df[c], out, active_thr=0.5)
    for c in EMO:
        stats(c, df[c], out, active_thr=0.5)
    for c in VA + POSE + GAZE:
        stats(c, df[c], out)
    for c in BLEND:
        if c in df:
            stats("bs_" + c, df[c], out, active_thr=0.3)
    # symmetric combos (less noisy than left/right separately)
    if "eyeBlinkLeft" in df:
        blink = (df["eyeBlinkLeft"] + df["eyeBlinkRight"]) / 2
        smile = (df["mouthSmileLeft"] + df["mouthSmileRight"]) / 2
        look_down = (df["eyeLookDownLeft"] + df["eyeLookDownRight"]) / 2
        look_up = (df["eyeLookUpLeft"] + df["eyeLookUpRight"]) / 2
        look_side = (df["eyeLookOutLeft"] + df["eyeLookInRight"] + df["eyeLookInLeft"] + df["eyeLookOutRight"]) / 4
        averted = (look_down > 0.4) | (look_up > 0.4) | (look_side > 0.4)
        stats("blink", blink, out, 0.4)
        stats("smile_bs", smile, out, 0.3)
        stats("look_down", look_down, out, 0.4)
        stats("look_up", look_up, out, 0.4)
        stats("look_side", look_side, out, 0.4)
        out["gaze_averted_frac"] = averted.mean()
        out["gaze_aversion_switch_rate"] = np.abs(np.diff(averted.astype(int))).mean() if len(df) > 1 else 0.0
    # framing / proximity (normalised by frame size -> camera-independent)
    fw = df["FrameWidth"].replace(0, np.nan)
    fh = df["FrameHeight"].replace(0, np.nan)
    face_w = df["FaceRectWidth"] / fw
    cx = (df["FaceRectX"] + df["FaceRectWidth"] / 2) / fw - 0.5
    cy = (df["FaceRectY"] + df["FaceRectHeight"] / 2) / fh - 0.5
    stats("face_size", face_w, out)
    stats("face_cx", cx, out)
    stats("face_cy", cy, out)
    out["face_off_center"] = np.sqrt(cx ** 2 + cy ** 2).mean()
    out["landscape"] = float((fw / fh).median() > 1)
    stats("FaceScore", df["FaceScore"], out)
    # head translation energy (3D)
    for c in ["X", "Y", "Z"]:
        stats("head_" + c, df[c] / df["FaceRectWidth"].replace(0, np.nan), out)
    if len(df) > 1:
        d = df[["Pitch", "Yaw", "Roll"]].diff().abs().sum(axis=1)
        out["head_rot_energy"] = d.mean()
        out["head_rot_still_frac"] = (d < np.nanpercentile(d, 25) + 1e-9).mean()
        # expressivity: mean frame-to-frame change across all AUs
        out["au_change_energy"] = df[AUS].diff().abs().mean(axis=1).mean()
        out["emo_change_energy"] = df[EMO].diff().abs().mean(axis=1).mean()
    # thirds of the answer (opening / middle / closing) for a few key channels
    n = len(df)
    if n >= 6:
        idx = np.array_split(np.arange(n), 3)
        for name, ch in [("happy", "Happy"), ("valence", "valence"), ("arousal", "arousal"),
                         ("AU12", "AU12"), ("gaze", "gaze_angle")]:
            vals = [df[ch].iloc[i].mean() for i in idx]
            out[f"{name}_open"], out[f"{name}_mid"], out[f"{name}_close"] = vals
            out[f"{name}_close_minus_open"] = vals[2] - vals[0]
    return out


def main():
    files = sorted(CACHE.glob("*.csv"))
    print(f"{len(files)} cached frame files")
    rows = []
    for i, f in enumerate(files):
        try:
            df = pd.read_csv(f, usecols=lambda c: c in set(USECOLS))
        except Exception as e:  # noqa
            print("  skip", f.name, e)
            continue
        r = video_features(df)
        r["file_name"] = f.stem + ".mp4"
        rows.append(r)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(files)}")
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT.name}: {out.shape}")


if __name__ == "__main__":
    main()
