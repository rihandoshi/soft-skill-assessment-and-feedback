"""Phase 4, step 3: cut the candidate's answers out of a full interview recording, without a human transcript and
without a gated model (no Hugging Face token).

  1. Whisper words for the whole recording (faster-whisper medium.en, Stage 4 / 7 settings).
  2. Words are grouped into chunks at sentence ends (. ? !) and at gaps >= 0.5 s.
  3. Speaker embeddings from microsoft/wavlm-base-plus-sv (an x-vector speaker-verification model, MIT licence,
     ungated) on 1.5-s windows with a 0.5-s hop over the whole recording. A chunk's embedding is the mean of the
     windows that lie inside it (or whose centre does, for short chunks). K-means (k = 2) on the unit-length chunk
     embeddings splits the two speakers; chunks with no window take the speaker of the nearest chunk in time.
  4. The interviewer is the speaker who speaks less. Then every chunk that ends with "?" and matches one of the
     five question patterns is given to the interviewer, whatever its voice cluster.
     History (Stage 12 notes): attempt 1 embedded each whole chunk and had no question rule. It first picked the
     interviewer by question keywords, which swapped the speakers in most interviews (candidates say "team" and
     "leadership" more often than the interviewer does). Attempt 2 (this version) was developed on odd-numbered
     MIT students only and tested once on the even-numbered ones.
  5. Consecutive chunks of one speaker form a turn. From here the Stage 7 rules are reused unchanged
     (mit_segment_answers.find_questions / answer_span): the five questions in order, the answer from the first
     candidate word after the question to the last candidate word before the next substantive interviewer turn
     (> 12 tokens), backchannels and short prompts kept inside, clips padded by 0.2 s before and 0.5 s after.

Assumes two speakers (one interviewer, one candidate) and the five standard questions; recordings with other
questions need their own patterns. This is the shared upstream step (README section 11); if the team settles on a
diarization model (e.g. pyannote with a token), only steps 2-4 change.

GPU: Whisper medium (about 1.5 GB) and a 100M-parameter speaker model (under 1 GB), one after the other.

Run:  python scripts/explainability/segment_interview.py interview.wav [--out answers.csv]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mit_segment_answers as M  # noqa: E402  (question patterns, answer rules)

SR = 16000
SV_MODEL = "microsoft/wavlm-base-plus-sv"
CHUNK_GAP_S = 0.5
END = re.compile(r"[.?!][\"')\]]*$")


def transcribe(audio: np.ndarray) -> list[dict]:
    import analyze_answer as AA
    from faster_whisper import WhisperModel
    import run_verbatim_asr as A
    dev = AA.device()
    model = WhisperModel(AA.ASR_MODEL, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segments, _ = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5,
                                   initial_prompt=A.INITIAL_PROMPT, condition_on_previous_text=False,
                                   vad_filter=False, temperature=0.0)
    words = [{"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
              "p": round(float(w.probability), 4)} for s in segments for w in (s.words or [])]
    del model
    AA.free_gpu()
    return words


def chunk_words(words: list[dict]) -> list[list[int]]:
    out, cur = [], []
    for i, w in enumerate(words):
        if cur and (w["start"] - words[cur[-1]]["end"] >= CHUNK_GAP_S or END.search(words[cur[-1]]["w"])):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


WIN_S, HOP_S = 1.5, 0.5


class SpeakerEmbedder:
    def __init__(self, device: str):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
        self.model = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()
        self.device = device

    def windows(self, audio: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """(window start times in s, unit-length embeddings) for 1.5-s windows with a 0.5-s hop."""
        import torch
        w, h = int(WIN_S * SR), int(HOP_S * SR)
        starts = np.arange(0, max(len(audio) - w, 1), h)
        out = []
        for i in range(0, len(starts), batch):
            x = [np.pad(audio[s:s + w], (0, max(0, w - len(audio[s:s + w])))) for s in starts[i:i + batch]]
            inp = self.fe(x, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                e = self.model(inp["input_values"].to(self.device)).embeddings.float().cpu().numpy()
            out.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return starts / SR, np.concatenate(out) if out else np.empty((0, 512), np.float32)


def is_question(words: list[dict], chunk: list[int]) -> bool:
    t = " ".join(words[i]["w"] for i in chunk).rstrip()
    return t.endswith("?") and any(re.search(p, t, flags=re.I) for _, p in M.QUESTIONS)


def label_speakers(audio: np.ndarray, words: list[dict], chunks: list[list[int]], embed) -> np.ndarray:
    """Per chunk: 1 = interviewer, 0 = candidate."""
    from sklearn.cluster import KMeans
    ws, E = embed.windows(audio)
    span = [(words[c[0]]["start"], words[c[-1]]["end"]) for c in chunks]
    ce = []
    for a, b in span:
        inside = (ws >= a - 0.25) & (ws + WIN_S <= b + 0.25)
        if not inside.any():
            inside = (ws + WIN_S / 2 >= a) & (ws + WIN_S / 2 <= b)
        ce.append(E[inside].mean(0) if inside.any() else None)
    have = np.flatnonzero([e is not None for e in ce])
    if len(have) < 2:
        return np.zeros(len(chunks), int)
    Z = np.stack([ce[i] / np.linalg.norm(ce[i]) for i in have])
    lab = np.full(len(chunks), -1)
    lab[have] = KMeans(2, n_init=10, random_state=0).fit_predict(Z)
    mids = np.array([(a + b) / 2 for a, b in span])
    for k in np.flatnonzero(lab < 0):
        lab[k] = lab[have[int(np.argmin(np.abs(mids[have] - mids[k])))]]
    secs = np.bincount(lab, weights=[b - a for a, b in span], minlength=2)
    lab = (lab == int(np.argmin(secs))).astype(int)         # the speaker who speaks less = interviewer
    lab[[is_question(words, c) for c in chunks]] = 1
    return lab


def turns_from(words: list[dict], chunks: list[list[int]], lab: np.ndarray) -> tuple[list[dict], list[list[int]]]:
    """Turns with speaker names (lab: 1 = interviewer), and per turn its word indices."""
    text = lambda ws: " ".join(words[i]["w"] for i in ws)  # noqa: E731
    turns, per_turn = [], []
    for c, l in zip(chunks, lab):
        spk = "interviewer" if l == 1 else "interviewee"
        if turns and turns[-1]["speaker"] == spk:
            per_turn[-1] += c
            turns[-1]["text"] = text(per_turn[-1])
        else:
            turns.append({"speaker": spk, "text": text(c)})
            per_turn.append(list(c))
    return turns, per_turn


def segment(audio: np.ndarray, words: list[dict] | None = None, embed=None) -> tuple[pd.DataFrame, dict]:
    """Answers found in one interview: question, question text, answer start / end and clip start / end (s)."""
    import analyze_answer as AA
    words = words if words is not None else transcribe(audio)
    chunks = chunk_words(words)
    own = embed is None
    embed = embed or SpeakerEmbedder(AA.device())
    lab = label_speakers(audio, words, chunks, embed)
    if own:
        del embed
        AA.free_gpu()
    turns, per_turn = turns_from(words, chunks, lab)
    qs = M.find_questions(turns)
    order = [q for q, _ in M.QUESTIONS if q in qs]
    rows = []
    dur = len(audio) / SR
    for k, q in enumerate(order):
        span = M.answer_span(turns, per_turn, words, qs[q], qs[order[k + 1]] if k + 1 < len(order) else None)
        if span is None or span["end_s"] - span["start_s"] < M.MIN_ANSWER_S:
            continue
        rows.append({"question": q, "question_text": turns[qs[q]]["text"][:160].strip(),
                     "answer_start_s": span["start_s"], "answer_end_s": span["end_s"],
                     "clip_start_s": max(span["start_s"] - 0.2, 0.0), "clip_end_s": min(span["end_s"] + 0.5, dur),
                     "n_backchannels": span["n_backchannels"], "n_prompts": span["n_prompts"]})
    word_speaker = [""] * len(words)
    for t, ws in zip(turns, per_turn):
        for i in ws:
            word_speaker[i] = t["speaker"]
    info = {"n_words": len(words), "n_chunks": len(chunks), "n_turns": len(turns), "word_speaker": word_speaker,
            "questions_found": "".join(q[1] for q in order)}
    return pd.DataFrame(rows), {"words": words, "turns": turns, **info}


def main() -> None:
    import analyze_answer as AA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--out")
    args = ap.parse_args()
    ans, info = segment(AA.load_audio(args.audio))
    print({k: v for k, v in info.items() if k not in ("words", "turns", "word_speaker")})
    print(ans.to_string(index=False))
    if args.out:
        ans.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
