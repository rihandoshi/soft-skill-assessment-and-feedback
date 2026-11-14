"""Stage 5, step 1: run a pretrained dimensional speech-emotion model on 4-s windows (2-s hop).

Raw model outputs only (measurement layer). Nothing is normalised or judged here; that is step 2
(`extract_ser_descriptors.py`) and Stage 2.

Output per model:
  results/explainability/stage5_ser/tables/windows_{model}.csv   id, t_start, t_end, speech_frac, arousal, dominance, valence
  outputs/audio/cache/stage5_ser/{model}/{id}.npz                per-clip cache (outputs + mean hidden state for Stage 10)

Clips shorter than one window are scored as a single shorter window. Works on any list of clips, so Stage 7 can
run it on the MIT segments with the same settings.

Run:  python scripts/explainability/run_ser_windows.py [--model audeering] [--batch 16] [--limit 20]
      python scripts/explainability/run_ser_windows.py --model odyssey --dataset mit --batch 8
        (Phase 1: the Odyssey cross-check on MIT -> stage7_mit_validation/tables/mit_windows_odyssey.csv)
      --max-gpu-frac caps PyTorch's share of GPU memory (default 0.8).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402  (clip list + VAD loading)

SR = 16000
WIN_S, HOP_S = 4.0, 2.0
MIN_WIN_S = 1.0


def windows_for(n_samples: int) -> list[tuple[int, int]]:
    win, hop = int(WIN_S * SR), int(HOP_S * SR)
    if n_samples < int(MIN_WIN_S * SR):
        return []
    if n_samples <= win:
        return [(0, n_samples)]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples - hop // 2:  # cover the tail
        starts.append(n_samples - win)
    return [(s, s + win) for s in starts]


def speech_fraction(regions: np.ndarray, t0: float, t1: float) -> float:
    if not len(regions):
        return 0.0
    ov = np.clip(np.minimum(regions[:, 1], t1) - np.maximum(regions[:, 0], t0), 0, None).sum()
    return float(ov / (t1 - t0))


def run(model_name: str, clips: pd.DataFrame, batch: int) -> pd.DataFrame:
    import soundfile as sf
    import torch
    import ser_models as M
    cache_dir = P.CACHE_SER / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    vad, _ = X.load_vad(set(clips["id"]))
    model, t0, done = None, time.time(), 0
    rows = []
    for k, (cid, path) in enumerate(zip(clips["id"], clips["audio_path"]), 1):
        cache = cache_dir / f"{cid}.npz"
        if cache.exists():
            with np.load(cache) as z:
                spans, out = z["spans"], z["out"]
        else:
            if model is None:
                model = M.load(model_name, "cuda" if torch.cuda.is_available() else "cpu")
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == SR, f"{cid}: expected 16 kHz, got {sr}"
            spans = np.array(windows_for(len(audio)), dtype=np.int64).reshape(-1, 2)
            outs = []
            full = [(a, b) for a, b in spans if b - a == int(WIN_S * SR)]
            for i in range(0, len(full), batch):
                outs.append(M.predict(model, np.stack([audio[a:b] for a, b in full[i:i + batch]])))
            for a, b in spans:
                if b - a != int(WIN_S * SR):  # short clip: one shorter window
                    outs.append(M.predict(model, audio[a:b][None, :]))
            out = np.concatenate(outs) if outs else np.empty((0, 3), np.float32)
            np.savez(cache, spans=spans, out=out)
            done += 1
        reg = vad.get(cid, np.empty((0, 2)))
        for (a, b), (ar, do, va) in zip(spans, out):
            t_start, t_end = a / SR, b / SR
            rows.append({"id": cid, "t_start": round(t_start, 3), "t_end": round(t_end, 3),
                         "speech_frac": round(speech_fraction(reg, t_start, t_end), 3),
                         "arousal": float(ar), "dominance": float(do), "valence": float(va)})
        if k % 200 == 0 or k == len(clips):
            print(f"  {model_name} {k}/{len(clips)} clips ({done} computed) {(time.time() - t0) / 60:.1f} min", flush=True)
    return pd.DataFrame(rows)


def cap_gpu_memory(frac: float) -> None:
    """Make PyTorch fail with an out-of-memory error inside this process instead of filling the whole card
    (a 6-GB laptop GPU also drives the display)."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        print(f"GPU {torch.cuda.get_device_name(0)}: {total:.1f} GB, capped at {frac:.0%}", flush=True)


def report_gpu_peak() -> None:
    import torch
    if torch.cuda.is_available():
        print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)


def mit_clips() -> pd.DataFrame:
    """MIT answer clips (Stage 7) with their cached Silero VAD regions plugged into the VAD loader."""
    import json
    clips = pd.read_csv(P.S7_TABLES / "mit_answers.csv", dtype={"id": str})[["id", "audio_path"]]
    X.load_vad = lambda ids: ({i: np.asarray(json.loads((P.CACHE_MIT / "vad" / f"{i}.json").read_text()),
                                             float).reshape(-1, 2) for i in ids}, {})
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="audeering")
    parser.add_argument("--dataset", choices=["recruitview", "mit"], default="recruitview")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-gpu-frac", type=float, default=0.8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cap_gpu_memory(args.max_gpu_frac)
    clips = mit_clips() if args.dataset == "mit" else X.load_clips(None)[["id", "audio_path"]]
    if args.limit:
        clips = clips.sample(args.limit, random_state=0)
    table = run(args.model, clips, args.batch)
    report_gpu_peak()
    if args.limit:
        print(table.describe())
        return
    out = P.S7_TABLES / f"mit_windows_{args.model}.csv" if args.dataset == "mit" else P.s5_windows(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "mit" and args.model == "audeering" and out.exists():
        raise SystemExit(f"{out} exists (Stage 7, with extra columns); not overwriting")
    table.to_csv(out, index=False)
    print(f"wrote {out}: {len(table)} windows, {table['id'].nunique()} clips")


if __name__ == "__main__":
    main()
