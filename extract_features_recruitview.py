"""Extract per-video visual features for RecruitView2.

Uses OpenCV to sample one frame per second (avoids the torchcodec/FFmpeg issue
you hit earlier), runs Py-Feat on the frames, caches the raw per-frame output
per video, then aggregates to one row per video.

Outputs:
  frame_features/recruitview/<video>.csv   raw per-second Py-Feat output (cache)
  features_recruitview.csv                 one row per video (input to train.py)
"""
import tempfile
from pathlib import Path

import cv2
import pandas as pd
import feat

from aggregate import aggregate_video, valid_mask

VIDEO_DIR = Path("RecruitView2/videos")
CACHE_DIR = Path("frame_features/recruitview")
OUT_CSV = Path("features_recruitview.csv")
INTERVAL_SEC = 1.0
LIMIT = None            # set e.g. 5 to test on the first 5 videos only
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

DetCls = getattr(feat, "Detectorv2", None) or feat.Detector
try:
    detector = DetCls(device="auto")
except Exception as e:
    print(f"device='auto' failed ({e}); using default device")
    detector = DetCls()


def best_face_per_frame(fex: pd.DataFrame) -> pd.DataFrame:
    """Keep the most confident face for each frame (multi-face frames)."""
    if "frame" not in fex.columns:
        return fex
    if "FaceScore" in fex.columns:
        fex = fex.sort_values("FaceScore")
    return fex.groupby("frame", as_index=False).tail(1).sort_values("frame")


def video_stats(video_path: Path):
    """Duration and how many 1 Hz samples we expect (cheap: no decoding)."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    step = max(1, round(fps * INTERVAL_SEC))
    return n_frames / fps, len(range(0, n_frames, step))


def extract_video(video_path: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps * INTERVAL_SEC))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(0, n_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                break
            p = Path(tmp) / f"{f:07d}.jpg"
            cv2.imwrite(str(p), img)
            try:
                fex = pd.DataFrame(detector.detect(str(p), data_type="image", progress_bar=False))
            except Exception as e:
                print(f"    frame {f}: detect failed ({e})")
                continue
            if fex.empty:
                continue
            if "FaceScore" in fex.columns:
                fex = fex.sort_values("FaceScore").tail(1)
            else:
                fex = fex.head(1)
            row = fex.iloc[0].to_dict()
            row["frame"] = f
            row["time_sec"] = f / fps
            rows.append(row)
    cap.release()
    return pd.DataFrame(rows)


def main():
    videos = sorted(p for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if LIMIT:
        videos = videos[:LIMIT]
    print(f"Processing {len(videos)} videos from {VIDEO_DIR}")
    agg_rows = []
    for i, v in enumerate(videos, 1):
        cache = CACHE_DIR / f"{v.stem}.csv"
        try:
            if cache.exists():
                frames_all = pd.read_csv(cache)
            else:
                frames_all = extract_video(v)
                if frames_all.empty:
                    print(f"[{i}/{len(videos)}] {v.name}: no faces detected, skipping")
                    continue
                frames_all.to_csv(cache, index=False)
            # Py-Feat returns an all-NaN row for frames without a face: aggregate real detections only
            frames = frames_all[valid_mask(frames_all)]
            dur, expected = video_stats(v)
            if frames.empty:
                print(f"[{i}/{len(videos)}] {v.name}: no valid detections, skipping")
                continue
            row = aggregate_video(frames, verbose=(i == 1))   # prints column groups once
            row["video_id"] = v.stem
            row["file_name"] = v.name
            row["duration_sec"] = dur
            row["expected_samples"] = expected
            row["face_rate"] = len(frames) / max(expected, 1)   # share of sampled frames with a detected face
            agg_rows.append(row)
            print(f"[{i}/{len(videos)}] {v.name}: {len(frames)}/{expected} samples with a detected face")
        except Exception as e:
            print(f"[{i}/{len(videos)}] {v.name}: FAILED -> {e}")

    pd.DataFrame(agg_rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(agg_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
