"""Stage 6, step 1: per-clip recording-quality measurements (measurement layer only; the gate is set in step 2).

Two kinds of measurement:
  SQUIM (torchaudio.pipelines.SQUIM_OBJECTIVE, trained on DNS 2020; no clean reference needed)
    squim_stoi    estimated intelligibility (0-1)
    squim_pesq    estimated perceptual quality (1-4.5)
    squim_si_sdr  estimated signal-to-distortion ratio (dB)
    Run on 10-s chunks with >= 50% VAD speech (whole clip if none qualifies); median over chunks.
  Signal checks (frame RMS, 25 ms / 10 ms; VAD speech regions from the vad_pause features)
    speech_level_dbfs   median frame level inside speech
    noise_floor_dbfs    median frame level in non-speech (>= 0.1 s from speech; digital silence excluded);
                        falls back to the 10th percentile of all frames when < 0.5 s of non-speech exists
    snr_vad_db          speech_level - noise_floor
    clip_frac           share of samples with |x| >= 0.99 (clipping)
    peak_dbfs, digital_silence_frac

Output   stage6_quality/quality_scores.csv   one row per clip (all splits; no labels used)
Cache    outputs/audio/cache/stage6_quality/{id}.json

Run:  python scripts/explainability/run_quality_scores.py [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402

SR = 16000
CHUNK_S = 10.0
MIN_CHUNK_SPEECH = 0.5
FRAME, HOP = int(0.025 * SR), int(0.010 * SR)
EDGE_GUARD_S = 0.1
DIGITAL_SILENCE_DB = -90.0
CLIP_LEVEL = 0.99


def frame_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < FRAME:
        return np.empty(0), np.empty(0)
    idx = np.arange(0, len(x) - FRAME + 1, HOP)
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME)[idx]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-10)), (idx + FRAME / 2) / SR


def in_regions(t: np.ndarray, regions: np.ndarray, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(len(t), bool)
    for a, b in regions:
        m |= (t >= a - pad) & (t < b + pad)
    return m


def signal_checks(x: np.ndarray, regions: np.ndarray) -> dict:
    db, t = frame_db(x)
    out = {"peak_dbfs": float(20 * np.log10(max(np.abs(x).max(), 1e-10))) if len(x) else np.nan,
           "clip_frac": float((np.abs(x) >= CLIP_LEVEL).mean()) if len(x) else np.nan,
           "digital_silence_frac": float((db < DIGITAL_SILENCE_DB).mean()) if len(db) else np.nan}
    live = db >= DIGITAL_SILENCE_DB
    speech = in_regions(t, regions) & live
    quiet = ~in_regions(t, regions, EDGE_GUARD_S) & live
    out["speech_level_dbfs"] = float(np.median(db[speech])) if speech.sum() >= 50 else np.nan
    if quiet.sum() >= 50:
        out["noise_floor_dbfs"], out["noise_floor_source"] = float(np.median(db[quiet])), "non-speech"
    elif live.sum() >= 50:
        out["noise_floor_dbfs"], out["noise_floor_source"] = float(np.percentile(db[live], 10)), "p10-all"
    else:
        out["noise_floor_dbfs"], out["noise_floor_source"] = np.nan, "none"
    out["snr_vad_db"] = out["speech_level_dbfs"] - out["noise_floor_dbfs"]
    return out


def squim_chunks(x: np.ndarray, regions: np.ndarray) -> list[tuple[int, int]]:
    n = int(CHUNK_S * SR)
    chunks = []
    for a in range(0, len(x) - n + 1, n):
        t0, t1 = a / SR, (a + n) / SR
        sp = np.clip(np.minimum(regions[:, 1], t1) - np.maximum(regions[:, 0], t0), 0, None).sum() if len(regions) else 0
        if sp / CHUNK_S >= MIN_CHUNK_SPEECH:
            chunks.append((a, a + n))
    if not chunks and len(x) >= 2 * SR:
        chunks = [(0, len(x))]
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import soundfile as sf
    import torch
    from torchaudio.pipelines import SQUIM_OBJECTIVE

    seg = pd.read_csv(P.SEGMENTS, dtype={"id": str}, usecols=["id"])
    clips = X.load_clips(None).merge(seg, on="id")
    if args.limit:
        clips = clips.head(args.limit)
    vad, _ = X.load_vad(set(clips["id"]))
    P.CACHE_QUALITY.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, rows, t0 = None, [], time.time()
    for k, (cid, path) in enumerate(zip(clips["id"], clips["audio_path"]), 1):
        cache = P.CACHE_QUALITY / f"{cid}.json"
        if cache.exists():
            rows.append(json.loads(cache.read_text()))
            continue
        if model is None:
            model = SQUIM_OBJECTIVE.get_model().to(device).eval()
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        assert sr == SR, f"{cid}: expected 16 kHz, got {sr}"
        regions = vad.get(cid, np.empty((0, 2)))
        row = {"id": cid, **signal_checks(x, regions)}
        chunks = squim_chunks(x, regions)
        row["squim_n_chunks"] = len(chunks)
        if chunks:
            with torch.no_grad():
                if len(chunks) == 1 and chunks[0][1] - chunks[0][0] != int(CHUNK_S * SR):
                    batch = torch.from_numpy(x[None, :]).to(device)
                else:
                    batch = torch.from_numpy(np.stack([x[a:b] for a, b in chunks])).to(device)
                stoi, pesq, sdr = (v.float().cpu().numpy() for v in model(batch))
            row.update(squim_stoi=float(np.median(stoi)), squim_pesq=float(np.median(pesq)),
                       squim_si_sdr=float(np.median(sdr)))
        else:
            row.update(squim_stoi=np.nan, squim_pesq=np.nan, squim_si_sdr=np.nan)
        row = {k2: (None if isinstance(v, float) and not np.isfinite(v) else v) for k2, v in row.items()}
        cache.write_text(json.dumps(row))
        rows.append(row)
        if k % 200 == 0 or k == len(clips):
            print(f"  {k}/{len(clips)} clips {(time.time() - t0) / 60:.1f} min", flush=True)
    table = pd.DataFrame(rows)
    if args.limit:
        print(table.describe().T.round(3).to_string())
        return
    P.S6.mkdir(parents=True, exist_ok=True)
    table.to_csv(P.QUALITY_SCORES, index=False)
    print(f"wrote {P.QUALITY_SCORES}: {len(table)} clips")


if __name__ == "__main__":
    main()
