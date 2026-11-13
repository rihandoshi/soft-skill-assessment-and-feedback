"""Stage 8, detector 2: disfluency classifier per 3-second window.

Model: vocametrix/wav2vec2-xlsr-53-stuttering-classification (Apache-2.0; Wav2Vec2ForSequenceClassification,
standard transformers class, weights from safetensors in outputs/audio/cache/hf_vocametrix_stutter, no remote code).
Trained on SEP-28k-Extended (3-s podcast clips). Classes: Soundrepetition, Wordrepetition, block, fluent,
interjection ("um/uh"), prolongation. Reported F1: interjection 0.77, fluent 0.82, word repetition 0.56, block 0.32.

Windows: 3 s (the training clip length), hop 1.5 s, over the whole answer; a window is kept when >= 50% of it is
VAD speech. The feature extractor normalises each window (do_normalize, as in training).

Per answer (disfluency_answers.csv):
  n_windows                       speech windows scored
  p_<class>                       mean class probability over the windows
  share_<class>                   share of windows whose top class is <class>
  share_disfluent                 share of windows whose top class is not "fluent"

Output  stage8_detectors/tables/disfluency_windows.csv, disfluency_answers.csv (RecruitView + MIT)
Cache   outputs/audio/cache/stage8_detectors/disfluency/{id}.npz
Run:    python scripts/explainability/run_disfluency_classifier.py [--limit 20]
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
import run_verbatim_asr as ASR  # noqa: E402

SR = 16000
WIN_S, HOP_S, MIN_SPEECH = 3.0, 1.5, 0.5
BATCH = 16
CACHE = P.CACHE_S8 / "disfluency"


def all_clips() -> pd.DataFrame:
    rv = ASR.load_clips().assign(dataset="recruitview")
    mit = pd.read_csv(P.S7_TABLES / "mit_answers.csv", dtype={"id": str})[["id", "audio_path"]].assign(dataset="mit")
    return pd.concat([rv, mit], ignore_index=True)


def speech_regions(clips: pd.DataFrame) -> dict[str, np.ndarray]:
    reg = ASR.load_vad_speech()
    for cid in clips.loc[clips["dataset"] == "mit", "id"]:
        reg[cid] = np.asarray(json.loads((P.CACHE_MIT / "vad" / f"{cid}.json").read_text()), float).reshape(-1, 2)
    return reg


def windows(n_samples: int, regions: np.ndarray) -> list[tuple[float, float, float]]:
    dur = n_samples / SR
    starts = np.arange(0.0, max(dur - WIN_S, 0.0) + 1e-9, HOP_S) if dur >= WIN_S else np.array([0.0])
    out = []
    for s in starts:
        e = min(s + WIN_S, dur)
        speech = sum(max(0.0, min(e, b) - max(s, a)) for a, b in regions) / max(e - s, 1e-6)
        if speech >= MIN_SPEECH:
            out.append((float(s), float(e), float(speech)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    clips = all_clips()
    if args.limit:
        clips = pd.concat([g.sample(args.limit // 2, random_state=0) for _, g in clips.groupby("dataset")])
    regions = speech_regions(clips)
    CACHE.mkdir(parents=True, exist_ok=True)

    import soundfile as sf
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = AutoFeatureExtractor.from_pretrained(P.HF_DISFLUENCY, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(P.HF_DISFLUENCY, local_files_only=True,
                                                              use_safetensors=True).eval().to(device)
    if device == "cuda":
        model.half()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    t0, done = time.time(), 0
    win_rows = []
    for k, r in enumerate(clips.itertuples(), 1):
        cache = CACHE / f"{r.id}.npz"
        if cache.exists():
            z = np.load(cache)
            spans, probs = z["spans"], z["probs"]
        else:
            x, sr = sf.read(r.audio_path, dtype="float32")
            x = x.mean(axis=1) if x.ndim > 1 else x
            assert sr == SR, f"{r.id}: {sr} Hz"
            w = windows(len(x), regions.get(r.id, np.empty((0, 2))))
            spans = np.asarray(w, float).reshape(-1, 3)
            probs = np.empty((0, len(labels)), np.float32)
            chunks = [np.pad(x[int(s * SR):int(e * SR)], (0, max(0, int(WIN_S * SR) - (int(e * SR) - int(s * SR)))))
                      for s, e, _ in w]
            out = []
            for b in range(0, len(chunks), BATCH):
                inp = fe(chunks[b:b + BATCH], sampling_rate=SR, return_tensors="pt")
                with torch.no_grad():
                    logits = model(inp["input_values"].to(device, dtype=model.dtype)).logits
                out.append(torch.softmax(logits.float(), -1).cpu().numpy())
            if out:
                probs = np.concatenate(out).astype(np.float32)
            np.savez(cache, spans=spans, probs=probs)
            done += 1
        for (s, e, sp), pr in zip(spans, probs):
            win_rows.append({"id": r.id, "dataset": r.dataset, "start": s, "end": e, "speech_frac": sp,
                             "top": labels[int(pr.argmax())], **{f"p_{c}": float(v) for c, v in zip(labels, pr)}})
        if k % 200 == 0 or k == len(clips):
            print(f"  {k}/{len(clips)} clips ({done} new, {(time.time() - t0) / 60:.1f} min)", flush=True)

    win = pd.DataFrame(win_rows)
    pcols = [f"p_{c}" for c in labels]
    agg = win.groupby(["id", "dataset"])[pcols].mean()
    share = pd.crosstab(win["id"], win["top"], normalize="index").reindex(columns=labels, fill_value=0.0)
    share.columns = [f"share_{c}" for c in labels]
    ans = agg.join(share, on="id").reset_index()
    ans.insert(2, "n_windows", ans["id"].map(win.groupby("id").size()))
    ans["share_disfluent"] = 1.0 - ans["share_fluent"]
    ans = clips[["id", "dataset"]].merge(ans, on=["id", "dataset"], how="left")   # answers with no speech windows
    ans["n_windows"] = ans["n_windows"].fillna(0).astype(int)
    if args.limit:
        print(ans.describe().T.round(3).to_string())
        return
    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    win.to_csv(P.DISFLUENCY_WINDOWS, index=False)
    ans.to_csv(P.DISFLUENCY_ANSWERS, index=False)
    print(f"wrote {P.DISFLUENCY_ANSWERS}: {len(ans)} answers, {len(win)} windows")
    print(win.groupby("dataset")["top"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
