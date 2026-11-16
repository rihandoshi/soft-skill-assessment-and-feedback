"""Stage 11 (Phase 3, Part B, variant B3): stronger pooled embeddings for the confidence retraining (GPU, inference
only; no fine-tuning).

  wavlm_large  microsoft/wavlm-large, frame-mean of the hidden states of layers 6, 12, 18 and 24 per window
               (the layer used is chosen later by grouped CV on train people only)
  audeering    audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim, its own pooled hidden state (the 1024-d vector
               its arousal / dominance / valence head reads)

Windows: the style tagger's (10 s, 5-s hop, >= 50% VAD speech; answers under 10 s as one window, >= 3 s), inputs
normalised per window. Per answer: mean and standard deviation over the windows (std = 0 for one window), i.e.
2 x 1024 values per layer.

Cache   outputs/audio/cache/stage11_style_tagger/embeddings/{model}/{id}.npz  (mean: (layers, 1024), std: same)
Run:    python scripts/explainability/extract_partb_embeddings.py --model wavlm_large --limit 4   (smoke test)
        python scripts/explainability/extract_partb_embeddings.py --model wavlm_large
        python scripts/explainability/extract_partb_embeddings.py --model audeering
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
import run_disfluency_classifier as D  # noqa: E402
import run_style_tagger as ST  # noqa: E402  (windows)
import voxprofile_models as VP  # noqa: E402  (normalise)
from run_ser_windows import cap_gpu_memory, report_gpu_peak  # noqa: E402

SR = 16000
WAVLM_LAYERS = [6, 12, 18, 24]
CACHE = P.ROOT / "outputs" / "audio" / "cache" / "stage11_style_tagger" / "embeddings"


def load(name: str, device: str):
    import torch
    if name == "wavlm_large":
        from transformers import WavLMModel
        m = WavLMModel.from_pretrained("microsoft/wavlm-large").to(device).eval()

        def f(x):
            hs = m(x, attention_mask=torch.ones_like(x, dtype=torch.long), output_hidden_states=True).hidden_states
            return torch.stack([hs[i].mean(1) for i in WAVLM_LAYERS], 1)          # (b, 4, 1024)
        return f
    import ser_models as M
    m = M.load("audeering", device)
    return lambda x: m.model(x)[0][:, None, :]                                    # (b, 1, 1024)


def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["wavlm_large", "audeering"], required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-gpu-frac", type=float, default=0.8)
    args = ap.parse_args()
    cap_gpu_memory(args.max_gpu_frac)
    clips = D.all_clips()
    if args.limit:
        clips = pd.concat([g.sample(max(1, args.limit // 2), random_state=0) for _, g in clips.groupby("dataset")])
    regions = D.speech_regions(clips)
    out_dir = CACHE / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f, t0, done = None, time.time(), 0
    for k, c in enumerate(clips.itertuples(), 1):
        cache = out_dir / f"{c.id}.npz"
        if cache.exists() and not args.limit:
            continue
        if f is None:
            f = load(args.model, device)
        x, sr = sf.read(c.audio_path, dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        assert sr == SR, f"{c.id}: {sr} Hz"
        w = ST.windows(len(x), regions.get(c.id, np.empty((0, 2))))
        if not w:
            continue
        groups: dict[int, list[int]] = {}
        for i, (s, e, _) in enumerate(w):
            groups.setdefault(int(e * SR) - int(s * SR), []).append(i)
        feats = [None] * len(w)
        for n, idx in groups.items():
            chunks = VP.normalise(np.stack([x[int(w[i][0] * SR): int(w[i][0] * SR) + n] for i in idx]))
            for b in range(0, len(idx), args.batch):
                with torch.no_grad():
                    h = f(torch.from_numpy(chunks[b:b + args.batch]).to(device)).float().cpu().numpy()
                for j, i in enumerate(idx[b:b + args.batch]):
                    feats[i] = h[j]
        a = np.stack(feats)                                                   # (windows, layers, 1024)
        if not args.limit:
            np.savez(cache, mean=a.mean(0).astype(np.float32), std=a.std(0).astype(np.float32))
        done += 1
        if k % 100 == 0 or k == len(clips) or args.limit:
            print(f"  {args.model} {k}/{len(clips)} clips ({done} computed, {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    report_gpu_peak()
    print(f"cached in {out_dir}: {len(list(out_dir.glob('*.npz')))} answers")


if __name__ == "__main__":
    main()
