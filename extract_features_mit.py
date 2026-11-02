"""Aggregate MIT Interview Dataset's PRECOMPUTED visual features per video.

No detector is run here: the dataset already ships frame-level face tracking
(Facial_Features/*.csv) and SHORE smile/nod/shake output (SmileData/{pre,post}).

Facial_Features first 12 columns (per the dataset README):
  Pitch, Yaw, Roll, inBrL, otBrL, inBrR, otBrR, EyeOL, EyeOR, oLipH, iLipH, LipCDt
SmileData first 3 columns: smile intensity, head nod, head shake.

Output: features_mit.csv  (one row per interview: video_id [lower-case, e.g. "p12"/"pp12"],
        participant, phase, ...)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate import summarize_activation, summarize_pose, summarize_continuous

MIT_ROOT = Path("MIT_INTERVIEW_DATASET")
OUT_CSV = Path("features_mit.csv")

FACE_COLS = ["Pitch", "Yaw", "Roll", "inBrL", "otBrL", "inBrR", "otBrR",
             "EyeOL", "EyeOR", "oLipH", "iLipH", "LipCDt"]
POSE_COLS = ["Pitch", "Yaw", "Roll"]
SMILE_COLS = ["smile_intensity", "head_nod", "head_shake"]


def parse_id(stem: str):
    """P12 -> (12, 'pre'); PP12 -> (12, 'post')."""
    m = re.match(r"^(PP|P)(\d+)", stem, flags=re.I)
    if not m:
        return None, None
    return int(m.group(2)), ("post" if m.group(1).upper() == "PP" else "pre")


def read_table(path: Path, names=None, ncols=None) -> pd.DataFrame:
    """Read csv/txt that may or may not have a header and may be comma/space separated."""
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    # If the first row is non-numeric it's a header row: re-read with header
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().mean() > 0.5:
        df = pd.read_csv(path, sep=None, engine="python", header=0)
    df = df.apply(pd.to_numeric, errors="coerce")
    if ncols is not None:
        df = df.iloc[:, :ncols]
    if names is not None:
        df.columns = names[: df.shape[1]]
    return df


def parse_smile_path(p: Path):
    """Return (participant, phase) for a SmileData file. Phase comes from a parent
    folder named pre/post when present, otherwise from the P/PP prefix."""
    parts = [x.lower() for x in p.parts]
    after = parts[parts.index("smiledata") + 1:-1] if "smiledata" in parts else parts[:-1]
    phase = None
    for d in after:
        if "post" in d:
            phase = "post"
        elif "pre" in d and phase is None:
            phase = "pre"
    pid, name_phase = parse_id(p.stem)
    if pid is None:                                    # e.g. "12.txt"
        m = re.search(r"(\d+)", p.stem)
        pid = int(m.group(1)) if m else None
    return pid, (phase or name_phase)


def index_smile_files() -> dict:
    idx = {}
    for p in MIT_ROOT.rglob("*"):
        if p.is_file() and "smiledata" in [x.lower() for x in p.parts]:
            pid, phase = parse_smile_path(p)
            if pid is not None and phase in ("pre", "post"):
                idx[(pid, phase)] = p
    return idx


def print_smile_diagnostics():
    sd = [d for d in MIT_ROOT.rglob("*") if d.is_dir() and d.name.lower() == "smiledata"]
    print("  !! No SmileData files could be indexed.")
    if not sd:
        print("  No folder named SmileData exists under", MIT_ROOT)
        return
    files = [p for p in sd[0].rglob("*") if p.is_file()]
    print(f"  {sd[0]} contains {len(files)} files; first few:")
    for p in files[:12]:
        print("   ", p.relative_to(sd[0]))
    print("  Send me this listing if the names look unfamiliar.")


def face_features(path: Path) -> dict:
    df = read_table(path, names=FACE_COLS, ncols=len(FACE_COLS)).dropna(how="all")
    t = np.arange(len(df), dtype=float)          # frame index (fps not given)
    row = {}
    for c in df.columns:
        if c in POSE_COLS:
            row.update(summarize_pose(c, t, df[c]))
        else:
            row.update(summarize_continuous(c, t, df[c]))
    # derived: eyebrow height (mean of the 4 brow distances)
    brow = None
    if {"inBrL", "otBrL", "inBrR", "otBrR"}.issubset(df.columns):
        brow = df[["inBrL", "otBrL", "inBrR", "otBrR"]].mean(axis=1)
        row.update(summarize_continuous("brow_height", t, brow.to_numpy()))

    # --- scale-free features -------------------------------------------------------------
    # The raw distances are in tracker units and, unless the dataset already normalised them
    # (the README does not say), scale with face size / camera distance. Coefficient of
    # variation, relative motion and ratios between distances do not.
    for c in [c for c in FACE_COLS if c not in POSE_COLS and c in df.columns]:
        x = df[c].to_numpy(dtype=float)
        m = np.nanmean(x)
        if np.isfinite(m) and abs(m) > 1e-9:
            row[f"{c}_cv"] = np.nanstd(x) / abs(m)
            row[f"{c}_relmotion"] = (np.nanmean(np.abs(np.diff(x))) / abs(m)) if len(x) > 1 else np.nan

    def ratio(num, den):
        num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
        out = np.full(num.shape, np.nan)
        ok = den > 1e-9
        out[ok] = num[ok] / den[ok]
        return out

    if {"oLipH", "iLipH", "LipCDt"}.issubset(df.columns):
        row.update(summarize_continuous("mouth_open_ratio", t, ratio(df["oLipH"] + df["iLipH"], df["LipCDt"])))
    if {"EyeOL", "EyeOR"}.issubset(df.columns):
        eye = df[["EyeOL", "EyeOR"]].mean(axis=1)
        if "LipCDt" in df.columns:
            row.update(summarize_continuous("smile_width_ratio", t, ratio(df["LipCDt"], eye)))
        if brow is not None:
            row.update(summarize_continuous("brow_raise_ratio", t, ratio(brow, eye)))
    row["n_frames"] = len(df)
    return row


def smile_features(path: Path) -> dict:
    df = read_table(path, names=SMILE_COLS, ncols=3).dropna(how="all")
    t = np.arange(len(df), dtype=float)
    row = {}
    for c in df.columns:
        x = df[c].to_numpy(dtype=float)
        # SHORE scales differ: treat as 0-100 if values exceed 1.5, else 0-1
        thr = 50.0 if np.nanmax(x) > 1.5 else 0.5
        row.update(summarize_activation(c, t, x, thr))
    return row


def main():
    face_files = sorted(MIT_ROOT.rglob("Facial_Features/*.csv"))
    print(f"Found {len(face_files)} Facial_Features files")

    # index SmileData by (participant, phase), tolerant of different naming layouts
    smile_idx = index_smile_files()
    print(f"Indexed {len(smile_idx)} SmileData files")
    if not smile_idx:
        print_smile_diagnostics()

    rows = []
    for i, f in enumerate(face_files, 1):
        pid, phase = parse_id(f.stem)
        if pid is None:
            print(f"  skipping unrecognised file name {f.name}")
            continue
        try:
            row = face_features(f)
            sp = smile_idx.get((pid, phase))
            if sp is not None:
                row.update(smile_features(sp))
            else:
                print(f"  no SmileData for {f.stem}")
            row.update({"video_id": f.stem.lower(), "participant": pid, "phase": phase})
            rows.append(row)
            print(f"[{i}/{len(face_files)}] {f.stem}")
        except Exception as e:
            print(f"[{i}/{len(face_files)}] {f.name}: FAILED -> {e}")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"Saved {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
