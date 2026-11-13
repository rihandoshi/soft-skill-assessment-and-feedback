"""Stage 8, detector 3: verbatim ASR with CrisperWhisper (fillers, repetitions, false starts).

Why: Whisper (Stage 4, even with a filler prompt) drops ~93% of fillers on MIT (0.26 vs 3.6 per 100 words in the
Turker transcripts; per-interview rho 0.16). CrisperWhisper (nyralabs/CrisperWhisper, INTERSPEECH 2024,
CC-BY-NC-4.0: research use) is Whisper large-v3 fine-tuned to transcribe verbatim: it emits fillers as the tokens
[UH] / [UM] and keeps repetitions and false starts. Loaded with the standard WhisperForConditionalGeneration from
safetensors in outputs/audio/cache/hf_crisperwhisper (no remote code; the pickle training_args.bin not downloaded).

Implementation notes
  * transformers 5 does not register the model's added tokens ([UH] = 35007, [UM] = 35481): the tokenizer decodes
    them to "". Tokens are therefore decoded one by one with an explicit map for the added tokens.
  * The config's forced_decoder_ids contain a None (decoded as token 0, "!"); they are cleared and language / task
    are passed explicitly. Greedy decoding (the config's num_beams=5 is ~5x slower; fillers are kept either way).
  * No word timestamps (they need eager attention and run at ~1x realtime on a 6 GB GPU). Answers are cut at VAD
    pauses into chunks <= 28 s, decoded in batches; each word keeps its chunk's start / end time. Timing-based
    measures (pauses) come from Stage 4; this table only counts.

Per word (crisper_words.csv): id, dataset, chunk, chunk_start, chunk_end, word_idx, word,
  is_filler       [UH] / [UM] / um / uh / erm / hmm ...
  is_repetition   same word as the previous non-filler word in the chunk ("we we", "the the")
  is_fragment     cut-off word: ends with "-", or a single letter other than "I" / "a" ("I s once I set up")
Per answer (crisper_answers.csv): n_words (non-filler), n_fillers, n_repetitions, n_fragments, speech_s, rates per
  100 words and fillers per minute of speech.

Output  stage8_detectors/tables/crisper_words.csv, crisper_answers.csv (RecruitView + MIT)
Cache   outputs/audio/cache/stage8_detectors/crisper/{id}.json
Run:    python scripts/explainability/run_crisper_asr.py [--limit 20] [--dataset mit|recruitview]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import run_disfluency_classifier as DC  # noqa: E402  (clip list + VAD regions for both datasets)

SR = 16000
MAX_CHUNK_S = 28.0
CACHE = P.CACHE_S8 / "crisper"
FILLERS = {"uh", "um", "umm", "uhm", "uhh", "erm", "er", "ah", "hmm", "hm", "mm", "mhm"}
NORM_RE = re.compile(r"[^a-z']")


def norm(w: str) -> str:
    return NORM_RE.sub("", w.lower())


def chunks_for(regions: np.ndarray, dur: float) -> list[tuple[float, float]]:
    """Cut [0, dur] into pieces <= MAX_CHUNK_S, cutting in the middle of VAD pauses where possible."""
    if dur <= MAX_CHUNK_S:
        return [(0.0, dur)]
    gaps = [(a_end + b_start) / 2 for (_, a_end), (b_start, _) in zip(regions[:-1], regions[1:])]
    out, start = [], 0.0
    while dur - start > MAX_CHUNK_S:
        cands = [g for g in gaps if start + 5.0 < g <= start + MAX_CHUNK_S]
        cut = max(cands) if cands else start + MAX_CHUNK_S
        out.append((start, cut))
        start = cut
    out.append((start, dur))
    return out


class Decoder:
    def __init__(self, tokenizer, model_dir: Path):
        added = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))["added_tokens_decoder"]
        self.added = {int(k): v["content"] for k, v in added.items()}
        self.tok = tokenizer
        self.eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    def __call__(self, ids: list[int]) -> str:
        parts = []
        for i in ids:
            if i == self.eot:
                break
            if i in self.added:
                c = self.added[i]
                if c.startswith("<|"):            # language / task / timestamp / notimestamps tokens
                    continue
                parts.append(f" {c} ")
            else:
                parts.append(self.tok.decode([i]))
        return re.sub(r"\s+", " ", "".join(parts)).strip()


def transcribe(clips: pd.DataFrame, batch_size: int) -> None:
    todo = [r for r in clips.itertuples() if not (CACHE / f"{r.id}.json").exists()]
    if not todo:
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    regions = DC.speech_regions(clips)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(P.HF_CRISPER, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(P.HF_CRISPER, local_files_only=True,
                                                            use_safetensors=True, dtype=dtype).to(device).eval()
    model.generation_config.forced_decoder_ids = None
    decode = Decoder(proc.tokenizer, P.HF_CRISPER)

    pending: list[tuple[str, int, float, float, np.ndarray]] = []
    results: dict[str, list] = {}
    n_chunks: dict[str, int] = {}
    t0, audio_s, done = time.time(), 0.0, 0

    def flush():
        nonlocal done
        if not pending:
            return
        feats = proc.feature_extractor([p[4] for p in pending], sampling_rate=SR, return_tensors="pt").input_features
        with torch.no_grad():
            ids = model.generate(feats.to(device, dtype=dtype), language="en", task="transcribe", num_beams=1)
        for (cid, k, s, e, _), seq in zip(pending, ids.tolist()):
            results.setdefault(cid, []).append({"chunk": k, "start": round(s, 3), "end": round(e, 3),
                                                "text": decode(seq)})
            if len(results[cid]) == n_chunks[cid]:
                chunks = sorted(results.pop(cid), key=lambda c: c["chunk"])
                tmp = CACHE / f"{cid}.json.tmp"           # write then rename: a crash never leaves a half file
                tmp.write_text(json.dumps({"chunks": chunks}), encoding="utf-8")
                tmp.replace(CACHE / f"{cid}.json")
                done += 1
        pending.clear()

    for r in todo:
        x, sr = sf.read(r.audio_path, dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        assert sr == SR, f"{r.id}: {sr} Hz"
        dur = len(x) / SR
        audio_s += dur
        cuts = chunks_for(regions.get(r.id, np.empty((0, 2))), dur)
        n_chunks[r.id] = len(cuts)
        for k, (s, e) in enumerate(cuts):
            pending.append((r.id, k, s, e, x[int(s * SR):int(e * SR)]))
            if len(pending) == batch_size:
                before = done
                flush()
                if done // 25 > before // 25:
                    el = time.time() - t0
                    print(f"  crisper {done}/{len(todo)} clips  {el / 60:.1f} min  "
                          f"{audio_s / max(el, 1e-6):.1f}x realtime", flush=True)
    flush()
    print(f"  crisper {done}/{len(todo)} clips  {(time.time() - t0) / 60:.1f} min", flush=True)


def tag(clips: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, ans = [], []
    for r in clips.itertuples():
        f = CACHE / f"{r.id}.json"
        if not f.exists():
            continue
        i = 0
        for c in json.loads(f.read_text(encoding="utf-8"))["chunks"]:
            prev = None
            for raw in c["text"].split():
                n = norm(raw)
                is_filler = raw.upper() in {"[UH]", "[UM]"} or n in FILLERS
                is_fragment = not is_filler and n != "" and (raw.endswith("-") or (len(n) == 1 and n not in {"i", "a"}))
                is_rep = not is_filler and n != "" and n == prev
                rows.append({"id": r.id, "dataset": r.dataset, "chunk": c["chunk"], "chunk_start": c["start"],
                             "chunk_end": c["end"], "word_idx": i, "word": raw, "is_filler": is_filler,
                             "is_repetition": is_rep, "is_fragment": is_fragment})
                i += 1
                if not is_filler and n:
                    prev = n
    words = pd.DataFrame(rows)
    speech = {cid: float(sum(b - a for a, b in reg)) for cid, reg in DC.speech_regions(clips).items()}
    for (cid, ds), g in words.groupby(["id", "dataset"], sort=False):
        n_words = int((~g["is_filler"]).sum())
        nf, nr, nfr = int(g["is_filler"].sum()), int(g["is_repetition"].sum()), int(g["is_fragment"].sum())
        per100 = 100.0 / max(n_words, 1)
        sp = speech.get(cid, np.nan)
        ans.append({"id": cid, "dataset": ds, "n_words": n_words, "n_fillers": nf, "n_repetitions": nr,
                    "n_fragments": nfr, "speech_s": sp,
                    "fillers_per_100w": nf * per100, "repetitions_per_100w": nr * per100,
                    "restarts_per_100w": (nr + nfr) * per100,
                    "fillers_per_min": nf / (sp / 60) if sp and sp > 0 else np.nan})
    return words, pd.DataFrame(ans)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dataset", choices=["mit", "recruitview"])
    ap.add_argument("--batch-size", type=int, default=4)   # 8 overflows a 6 GB GPU into shared memory (~10x slower)
    args = ap.parse_args()
    clips = DC.all_clips()
    clips = pd.concat([clips[clips["dataset"] == "mit"], clips[clips["dataset"] != "mit"]])   # MIT first
    if args.dataset:
        clips = clips[clips["dataset"] == args.dataset]
    if args.limit:
        clips = clips.sample(args.limit, random_state=0)
    transcribe(clips, args.batch_size)
    words, ans = tag(clips)
    if args.limit or args.dataset:
        print(ans.describe().T.round(2).to_string())
        if args.limit:
            print(words.groupby("id")["word"].apply(lambda s: " ".join(s)[:300]).head(3).to_string())
        return
    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    words.to_csv(P.CRISPER_WORDS, index=False)
    ans.to_csv(P.CRISPER_ANSWERS, index=False)
    print(f"wrote {P.CRISPER_ANSWERS}: {len(ans)} answers; {words['is_filler'].sum()} fillers, "
          f"{words['is_repetition'].sum()} repetitions, {words['is_fragment'].sum()} fragments")


if __name__ == "__main__":
    main()
