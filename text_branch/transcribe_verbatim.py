"""Verbatim speech-to-text with word timestamps, keeping fillers ([um], [uh]), repetitions
and cut-off words (th-).

Standard Whisper is trained on cleaned subtitles and drops most fillers. CrisperWhisper 2.0
(nyralabs, non-commercial research licence) is built for verbatim transcription. Runs
locally; audio never leaves the machine.

Reads  outputs/text/wav/<id>.wav          (from delivery_features.py --step wav)
Writes outputs/text/verbatim/<id>.json    {"text": ..., "words": [{"text", "start", "end"}]}
Finished clips are skipped, so the run can be stopped and resumed.

Usage:
  python text_branch/transcribe_verbatim.py [--limit 10] [--model nyralabs/CrisperWhisper2.0_large]
Requires: pip install "crisperwhisper[transformers]"
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from delivery_features import VERBATIM_DIR, WAV_DIR, all_responses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nyralabs/CrisperWhisper2.0_large")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=str(VERBATIM_DIR), help="where to write <id>.json")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    from crisperwhisper import CrisperWhisperModel
    model = CrisperWhisperModel(args.model, backend="transformers", compute_type="float16")

    ids = all_responses()["id"].tolist()
    if args.limit:
        ids = ids[: args.limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i in ids if not (out_dir / f"{i}.json").exists()
            and (WAV_DIR / f"{i}.wav").exists()]
    print(f"{len(ids)} clips, {len(todo)} to transcribe", flush=True)

    t0 = time.time()
    for k, rid in enumerate(todo):
        audio, sr = sf.read(WAV_DIR / f"{rid}.wav", dtype="float32")
        if audio.size == 0:
            out = {"text": "", "words": []}
        else:
            r = model.transcribe(audio, sr=sr, language="en", mode="verbatim", word_timestamps=True)
            out = {"text": r.text.strip(),
                   "words": [{"text": w.word.strip(), "start": w.start, "end": w.end}
                             for w in (r.words or [])]}
        out["model"] = args.model
        (out_dir / f"{rid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        if (k + 1) % 25 == 0 or k + 1 == len(todo):
            rate = (time.time() - t0) / (k + 1)
            print(f"{k + 1}/{len(todo)}  {rate:.1f}s/clip  ~{rate * (len(todo) - k - 1) / 60:.0f} min left",
                  flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
