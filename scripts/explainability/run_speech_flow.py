"""Phase 1, fluency second opinion: Vox-Profile speech-flow classifier per 3-second window.

Model: tiantiaf/wavlm-large-speech-flow (Vox-Profile, arXiv 2505.14648; OpenRAIL, no commercial use), revision
aea5a480. WavLM-large fine-tuned with LoRA (rank 16) on the feed-forward layers of the upper half (layers 13-23),
a softmax-weighted sum of all 25 hidden states, three 1x1 convolutions (1024 -> 256), mean pooling, and two heads:
  fluency      softmax over (fluent, disfluent)
  disfluency   sigmoid over (Block, Prolongation, Sound Repetition, Word Repetition, Interjection), multi-label

Loading: voxprofile_models.py (weights only, LoRA folded into transformers' WavLMModel; no code from the model repo
is executed; every weight in the file must be used and none may be missing).

Windows: the same as the disfluency classifier (run_disfluency_classifier.py): 3 s, 1.5-s hop, kept when >= 50% VAD
speech, so the two models are compared on identical audio. The model card uses a 1-s hop; the window length (3 s)
is what matters for the classifier.

Per answer (speech_flow_answers.csv):
  n_windows           speech windows scored
  sf_p_disfluent      mean "disfluent" probability over the windows
  sf_share_disfluent  share of windows with p(disfluent) > 0.5
  sf_<type>           mean sigmoid probability of each disfluency type

Output  stage8_detectors/tables/speech_flow_windows.csv, speech_flow_answers.csv (RecruitView + MIT)
Cache   outputs/audio/cache/stage8_detectors/speech_flow/{id}.npz
Run:    python scripts/explainability/run_speech_flow.py --limit 10      (smoke test: memory + speed, no tables)
        python scripts/explainability/run_speech_flow.py                 (all answers; resumes from the cache)
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
import run_disfluency_classifier as D  # noqa: E402  (clip list, VAD regions, windows)
from run_ser_windows import cap_gpu_memory, report_gpu_peak  # noqa: E402
import voxprofile_models as VP  # noqa: E402  (weights-only loader, shared with the style tagger)

SR = D.SR
CACHE = P.CACHE_S8 / "speech_flow"
TYPES = VP.SPEECH_FLOW_TYPES
OUT_WINDOWS = P.S8_TABLES / "speech_flow_windows.csv"
OUT_ANSWERS = P.S8_TABLES / "speech_flow_answers.csv"


def load_model(device: str):
    return VP.load("speech_flow", device)


def main() -> None:
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=8)
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
                model = load_model(device)
            x, sr = sf.read(c.audio_path, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            assert sr == SR, f"{c.id}: {sr} Hz"
            w = D.windows(len(x), regions.get(c.id, np.empty((0, 2))))
            spans = np.asarray(w, float).reshape(-1, 3)
            n = int(D.WIN_S * SR)
            chunks = np.stack([np.pad(x[int(s * SR):int(e * SR)], (0, max(0, n - (int(e * SR) - int(s * SR)))))[:n]
                               for s, e, _ in w]) if w else np.empty((0, n), np.float32)
            chunks = VP.normalise(chunks)
            out = []
            for b in range(0, len(chunks), args.batch):
                with torch.no_grad():
                    fl, ty = model(torch.from_numpy(chunks[b:b + args.batch]).to(device))
                out.append(np.concatenate([torch.softmax(fl, -1).cpu().numpy(), torch.sigmoid(ty).cpu().numpy()], 1))
            probs = np.concatenate(out).astype(np.float32) if out else np.empty((0, 7), np.float32)
            if not args.limit:
                np.savez(cache, spans=spans, probs=probs)
            done += 1
        for (s, e, sp), pr in zip(spans, probs):
            rows.append({"id": c.id, "dataset": c.dataset, "start": s, "end": e, "speech_frac": sp,
                         "p_fluent": float(pr[0]), "p_disfluent": float(pr[1]),
                         **{f"p_{t.replace(' ', '')}": float(v) for t, v in zip(TYPES, pr[2:])}})
        if k % 100 == 0 or k == len(clips) or args.limit:
            print(f"  {k}/{len(clips)} clips ({done} computed, {(time.time() - t0) / 60:.1f} min)", flush=True)

    report_gpu_peak()
    win = pd.DataFrame(rows)
    tcols = [f"p_{t.replace(' ', '')}" for t in TYPES]
    g = win.groupby("id")
    ans = pd.DataFrame({"n_windows": g.size(), "sf_p_disfluent": g["p_disfluent"].mean(),
                        "sf_share_disfluent": g["p_disfluent"].apply(lambda v: float((v > 0.5).mean())),
                        **{f"sf_{c[2:]}": g[c].mean() for c in tcols}}).reset_index()
    ans = clips[["id", "dataset"]].merge(ans, on="id", how="left")
    ans["n_windows"] = ans["n_windows"].fillna(0).astype(int)
    if args.limit:
        print(ans.describe().T.round(3).to_string())
        return
    win.to_csv(OUT_WINDOWS, index=False)
    ans.to_csv(OUT_ANSWERS, index=False)
    print(f"wrote {OUT_ANSWERS}: {len(ans)} answers, {len(win)} windows")
    print(ans.groupby("dataset")[["sf_p_disfluent", "sf_share_disfluent"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
