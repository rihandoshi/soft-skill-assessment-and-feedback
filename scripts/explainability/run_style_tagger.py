"""Stage 11 (Phase 3, Part A), step 2: the Vox-Profile speaking-style tagger on every answer.

Model: tiantiaf/wavlm-large-voice-quality (revision 5c0df771; OpenRAIL, no commercial use), loaded weights-only by
voxprofile_models.py. 25 independent tags (sigmoid), trained on ParaSpeechCaps with 3-15-s utterances.

Windows (pre-registered, stage11_style_tagger/preregistration.md): 10 s with a 5-s hop over the whole answer, the
tail covered by a last window ending at the clip end; a window is kept when >= 50% of it is VAD speech. An answer
shorter than 10 s with >= 3 s of audio is scored as one window of its own length.

Per answer (style_answers.csv): n_windows and the mean probability of each tag (columns tag_<name>, "-" -> "_").

Output  stage11_style_tagger/tables/style_windows.csv, style_answers.csv (RecruitView + MIT)
Cache   outputs/audio/cache/stage11_style_tagger/{id}.npz
Run:    python scripts/explainability/run_style_tagger.py --limit 6     (smoke test: memory + speed, no tables)
        python scripts/explainability/run_style_tagger.py               (all answers; resumes from the cache)
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
import run_disfluency_classifier as D  # noqa: E402  (clip list, VAD regions)
import voxprofile_models as VP  # noqa: E402
from run_ser_windows import cap_gpu_memory, report_gpu_peak  # noqa: E402

SR = 16000
WIN_S, HOP_S, MIN_SPEECH, MIN_CLIP_S = 10.0, 5.0, 0.5, 3.0
CACHE = P.ROOT / "outputs" / "audio" / "cache" / "stage11_style_tagger"
TAGS = VP.VOICE_QUALITY_TAGS
COLS = [f"tag_{t.replace('-', '_')}" for t in TAGS]


def windows(n_samples: int, regions: np.ndarray) -> list[tuple[float, float, float]]:
    dur = n_samples / SR
    if dur < MIN_CLIP_S:
        return []
    if dur <= WIN_S:
        spans = [(0.0, dur)]
    else:
        starts = list(np.arange(0.0, dur - WIN_S + 1e-9, HOP_S))
        if starts[-1] + WIN_S < dur - HOP_S / 2:
            starts.append(dur - WIN_S)
        spans = [(float(s), float(s + WIN_S)) for s in starts]
    out = []
    for s, e in spans:
        speech = sum(max(0.0, min(e, b) - max(s, a)) for a, b in regions) / max(e - s, 1e-6)
        if speech >= MIN_SPEECH:
            out.append((s, e, float(speech)))
    return out


def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-gpu-frac", type=float, default=0.8)
    args = ap.parse_args()
    cap_gpu_memory(args.max_gpu_frac)
    clips = D.all_clips()
    if args.limit:
        clips = pd.concat([g.sample(max(1, args.limit // 2), random_state=0) for _, g in clips.groupby("dataset")])
    regions = D.speech_regions(clips)
    CACHE.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None

    t0, done, rows = time.time(), 0, []
    for k, c in enumerate(clips.itertuples(), 1):
        cache = CACHE / f"{c.id}.npz"
        if cache.exists() and not args.limit:
            z = np.load(cache)
            spans, probs = z["spans"], z["probs"]
        else:
            if model is None:
                model = VP.load("voice_quality", device)
            x, sr = sf.read(c.audio_path, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            assert sr == SR, f"{c.id}: {sr} Hz"
            w = windows(len(x), regions.get(c.id, np.empty((0, 2))))
            spans = np.asarray(w, float).reshape(-1, 3)
            groups: dict[int, list[int]] = {}               # equal-length windows are batched together
            for i, (s, e, _) in enumerate(w):
                groups.setdefault(int(e * SR) - int(s * SR), []).append(i)
            probs = np.zeros((len(w), len(TAGS)), np.float32)
            for n, idx in groups.items():
                chunks = VP.normalise(np.stack([x[int(w[i][0] * SR): int(w[i][0] * SR) + n] for i in idx]))
                for b in range(0, len(idx), args.batch):
                    with torch.no_grad():
                        (logits,) = model(torch.from_numpy(chunks[b:b + args.batch]).to(device))
                    probs[idx[b:b + args.batch]] = torch.sigmoid(logits).cpu().numpy()
            if not args.limit:
                np.savez(cache, spans=spans, probs=probs)
            done += 1
        for (s, e, sp), pr in zip(spans, probs):
            rows.append({"id": c.id, "dataset": c.dataset, "start": s, "end": e, "speech_frac": sp,
                         **dict(zip(COLS, map(float, pr)))})
        if k % 100 == 0 or k == len(clips) or args.limit:
            print(f"  {k}/{len(clips)} clips ({done} computed, {(time.time() - t0) / 60:.1f} min)", flush=True)

    report_gpu_peak()
    win = pd.DataFrame(rows, columns=["id", "dataset", "start", "end", "speech_frac", *COLS])
    ans = win.groupby("id")[COLS].mean()
    ans.insert(0, "n_windows", win.groupby("id").size())
    ans = clips[["id", "dataset"]].merge(ans.reset_index(), on="id", how="left")
    ans["n_windows"] = ans["n_windows"].fillna(0).astype(int)
    if args.limit:
        print(ans.drop(columns=["id", "dataset"]).describe().T.round(3).to_string())
        return
    P.S11_TABLES.mkdir(parents=True, exist_ok=True)
    win.to_csv(P.STYLE_WINDOWS, index=False)
    ans.to_csv(P.STYLE_ANSWERS, index=False)
    print(f"wrote {P.STYLE_ANSWERS}: {len(ans)} answers ({int((ans['n_windows'] == 0).sum())} without a window), "
          f"{len(win)} windows")
    print(ans.groupby("dataset")[COLS].median().T.round(3).to_string())


if __name__ == "__main__":
    main()
