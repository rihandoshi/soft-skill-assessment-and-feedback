"""Stage 4, step 1: verbatim ASR with word timestamps (shared upstream step for the audio and text branches).

Why: the RecruitView transcripts are cleaned. They drop fillers and, in slow sentences, real words (Stage 1
verification: one line kept 21 of ~43 spoken words), which corrupts every speech-rate value. A new interview
has no transcript either, so the pipeline needs its own ASR anyway.

What it does: faster-whisper (default medium.en) on every clip, with word timestamps and word probabilities.
A filler-rich initial prompt makes Whisper keep "um / uh" instead of silently cleaning them up. Each word is
tagged:
  is_filled_pause   um, uh, erm, hmm, ... (vocal hesitations; excluded from word counts / speech rate)
  in_vad_speech     the word overlaps Silero VAD speech (False = possible hallucination in silence)
Lexical fillers ("like", "basically", "you know") are NOT tagged here: whether "like" is a filler depends on
the sentence, and filler counting is the text branch's job (it receives this word table).

Output
  results/explainability/stage4_asr/words.csv     id, word_idx, word, start, end, prob, is_filled_pause, in_vad_speech
  outputs/audio/cache/stage4_asr/{id}_{model}.json (per-clip cache; re-runs only transcribe missing clips)

Run:  python scripts/explainability/run_verbatim_asr.py [--model medium.en] [--limit 20]
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

ROOT = P.ROOT
AUDIO_META_DIR = ROOT / "Datasets" / "RecruitView Audio"
SPLITS = ["train", "val", "test"]
INITIAL_PROMPT = "Umm, so, uh, I think, hmm, I mean, like, uh, we did the, um, the project."
FILLED_PAUSES = {"um", "umm", "ummm", "uh", "uhh", "uhm", "erm", "er", "err", "ah", "ahh", "hmm", "hm", "mm", "mmm",
                 "eh", "uhhh"}
NORM_RE = re.compile(r"[^a-z']")


def norm(word: str) -> str:
    return NORM_RE.sub("", word.lower())


def load_clips() -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        meta = pd.read_csv(AUDIO_META_DIR / f"{split}_audio_metadata.csv", dtype={"id": str})
        frames.append(meta.assign(split=split))
    meta = pd.concat(frames, ignore_index=True)
    meta = meta[meta["status"] == "extracted"]
    return pd.DataFrame({"id": meta["id"].to_numpy(),
                         "audio_path": meta["audio_path"].map(lambda p: str(ROOT / Path(str(p).replace("\\", "/"))))})


def load_vad_speech() -> dict[str, np.ndarray]:
    regions = pd.concat([pd.read_csv(P.VAD_FEATURES / f"{s}_vad_regions.csv", dtype={"id": str}) for s in SPLITS])
    speech = regions[regions["region_type"] == "speech"]
    return {cid: g.sort_values("start_seconds")[["start_seconds", "end_seconds"]].to_numpy(float)
            for cid, g in speech.groupby("id")}


def transcribe_all(clips: pd.DataFrame, model_name: str) -> dict[str, list[dict]]:
    P.CACHE_ASR.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for cid, path in zip(clips["id"], clips["audio_path"]):
        cache = P.CACHE_ASR / f"{cid}_{model_name}.json"
        if cache.exists():
            out[cid] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            todo.append((cid, path))
    if not todo:
        return out
    import soundfile as sf  # pass samples, not a path: the installed PyAV is too new for faster-whisper's decoder
    import torch
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    t0, audio_s = time.time(), 0.0
    for k, (cid, path) in enumerate(todo, 1):
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        assert sr == 16000, f"{cid}: expected 16 kHz, got {sr}"
        audio_s += len(audio) / sr
        segments, _ = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5,
                                       initial_prompt=INITIAL_PROMPT, condition_on_previous_text=False,
                                       vad_filter=False, temperature=0.0)
        words = [{"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                  "p": round(float(w.probability), 4)} for s in segments for w in (s.words or [])]
        (P.CACHE_ASR / f"{cid}_{model_name}.json").write_text(json.dumps(words), encoding="utf-8")
        out[cid] = words
        if k % 25 == 0 or k == len(todo):
            el = time.time() - t0
            print(f"  asr {k}/{len(todo)}  {el / 60:.1f} min  {audio_s / max(el, 1e-6):.1f}x realtime", flush=True)
    return out


def word_table(words: dict[str, list[dict]], vad: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for cid, ws in words.items():
        reg = vad.get(cid, np.empty((0, 2)))
        for i, w in enumerate(ws):
            in_vad = bool(((reg[:, 0] < w["end"] + 0.1) & (reg[:, 1] > w["start"] - 0.1)).any()) if len(reg) else False
            rows.append({"id": cid, "word_idx": i, "word": w["w"], "start": w["start"], "end": w["end"], "prob": w["p"],
                         "is_filled_pause": norm(w["w"]) in FILLED_PAUSES, "in_vad_speech": in_vad})
    return pd.DataFrame(rows).sort_values(["id", "word_idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="medium.en")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    clips = load_clips()
    if args.limit:
        clips = clips.sample(args.limit, random_state=0)
    words = transcribe_all(clips, args.model)
    table = word_table(words, load_vad_speech())
    if args.limit:
        print(table.groupby("id").size().describe())
        return
    P.S4.mkdir(parents=True, exist_ok=True)
    table.to_csv(P.WORDS, index=False)
    print(f"wrote {P.WORDS}: {len(table)} words, {table['id'].nunique()} clips, "
          f"{table['is_filled_pause'].sum()} filled pauses, {(~table['in_vad_speech']).sum()} words outside VAD speech")


if __name__ == "__main__":
    main()
