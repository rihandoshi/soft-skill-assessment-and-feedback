"""Stage 7, step 1: cut the interviewee's answers to the five main questions out of each MIT interview.

MIT audio is one microphone for both people (the two stereo channels are identical), so the speakers must be
separated. Instead of a gated diarization model we use what the dataset already has: the Turker transcripts mark
every turn as "Interviewer:" or "Interviewee:" (but carry no times). Verbatim Whisper (same settings as Stage 4) gives
word timestamps for the full interview; aligning its words to the transcript words (difflib on normalised tokens)
gives each timestamped word a speaker.

Answers: the five standard questions are found in the interviewer's turns by keyword, in order:
  Q1 about yourself / your background | Q2 leadership | Q3 team / conflict | Q4 weakness / change one thing about
  yourself | Q5 why hire you / why should you get the job      (pre and post interviews word them differently)
The answer = the interviewee's speech from the first interviewee word after the question to the last interviewee
word before the next *substantive* interviewer turn (> PROMPT_MAX tokens). Short interviewer backchannels ("mhhmm")
and prompts ("You can pick anything.") stay inside the clip; they are counted (`n_backchannels`, `n_prompts`) and
their aligned duration is recorded (`interviewer_s_inside`) so clips with much interviewer speech can be excluded.

Steps (cached, re-runnable):
  --step asr       16-kHz mono copy of each interview + Whisper words  -> outputs/audio/cache/stage7_mit/
  --step segment   alignment + answer clips (16-kHz mono WAV)          -> outputs/audio/cache/stage7_mit/answers/
                   + stage7_mit_validation/tables/mit_answers.csv, mit_alignment.csv

Run:  python scripts/explainability/mit_segment_answers.py --step asr      (~40-60 min GPU)
      python scripts/explainability/mit_segment_answers.py --step segment
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import run_verbatim_asr as A  # noqa: E402

SR = 16000
ASR_MODEL = "medium.en"
BACKCHANNEL_MAX = 3      # 'mhhmm', 'uh huh'
PROMPT_MAX = 12          # short nudges ('You can pick anything.') stay inside the answer; longer turns end it
# Pre (P#) and post (PP#) interviews word the five questions differently.
# Variants seen in the transcripts: "your backround", "a little bit about you", "your life", "change on thing",
# "why do you think you should get into medical school / go to law school / get this job".
QUESTIONS = [("Q1", r"about yourself|your backg?round|introduce yourself|about you\b|your life"),
             ("Q2", r"leader"),
             ("Q3", r"team|challeng|conflict"),
             ("Q4", r"weakness|change one? thing"),
             ("Q5", r"\bhire\b|why should|why do you think (?:we|you) should|you should (?:get|go)|give you th")]
MIN_ANSWER_S = 3.0


def interview_ids() -> list[str]:
    return sorted((p.stem for p in P.MIT_AUDIO.glob("*.wav")), key=lambda s: (s.startswith("PP"), int(re.sub(r"\D", "", s))))


def mono16(iid: str) -> np.ndarray:
    cache = P.CACHE_MIT / "audio16k" / f"{iid}.wav"
    import soundfile as sf
    if cache.exists():
        return sf.read(cache, dtype="float32")[0]
    from scipy.signal import resample_poly
    x, sr = sf.read(P.MIT_AUDIO / f"{iid}.wav", dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    assert sr == 48000, f"{iid}: expected 48 kHz, got {sr}"
    y = resample_poly(x, 1, 3).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    sf.write(cache, y, SR, subtype="PCM_16")
    return y


def step_asr(limit: int | None) -> None:
    ids = interview_ids()[:limit] if limit else interview_ids()
    todo = [i for i in ids if not (P.CACHE_MIT / "asr" / f"{i}_{ASR_MODEL}.json").exists()]
    print(f"{len(ids)} interviews, {len(todo)} to transcribe", flush=True)
    if not todo:
        return
    import torch
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WhisperModel(ASR_MODEL, device=device, compute_type="float16" if device == "cuda" else "int8")
    (P.CACHE_MIT / "asr").mkdir(parents=True, exist_ok=True)
    t0, audio_s = time.time(), 0.0
    for k, iid in enumerate(todo, 1):
        audio = mono16(iid)
        audio_s += len(audio) / SR
        segments, _ = model.transcribe(audio, language="en", word_timestamps=True, beam_size=5,
                                       initial_prompt=A.INITIAL_PROMPT, condition_on_previous_text=False,
                                       vad_filter=False, temperature=0.0)
        words = [{"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                  "p": round(float(w.probability), 4)} for s in segments for w in (s.words or [])]
        (P.CACHE_MIT / "asr" / f"{iid}_{ASR_MODEL}.json").write_text(json.dumps(words), encoding="utf-8")
        el = time.time() - t0
        print(f"  asr {k}/{len(todo)} {iid}  {el / 60:.1f} min  {audio_s / max(el, 1e-6):.1f}x realtime", flush=True)


# ============================================================================================
def tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9' ]", " ", text.lower().replace("’", "'")).split() if t.strip("'")]


def load_turns() -> dict[str, list[dict]]:
    t = pd.read_csv(P.MIT_TRANSCRIPTS, header=None, names=["id", "text"])  # the file has no header row
    out = {}
    for iid, text in zip(t["id"].str.strip().str.upper(), t["text"].astype(str)):
        turns = []
        for raw in text.split("|"):
            m = re.match(r"\s*(Interviewer|Interviewee)\s*:\s*(.*)", raw, flags=re.I | re.S)
            if m:
                turns.append({"speaker": "interviewer" if m.group(1).lower() == "interviewer" else "interviewee",
                              "text": m.group(2).strip()})
        out[iid] = turns
    return out


def align(turns: list[dict], words: list[dict]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Returns per ASR word: speaker label (object array, None = unknown), matched transcript turn index (-1), and
    per turn the list of matched ASR word indices."""
    t_tok, t_turn = [], []
    for ti, turn in enumerate(turns):
        for tok in tokens(turn["text"]):
            t_tok.append(tok)
            t_turn.append(ti)
    a_tok = [(tokens(w["w"]) or [""])[0] for w in words]
    sm = difflib.SequenceMatcher(None, t_tok, a_tok, autojunk=False)
    turn_of = np.full(len(words), -1)
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            turn_of[blk.b + j] = t_turn[blk.a + j]
    # unmatched ASR words: take the turn of the neighbours when both sides agree
    filled = turn_of.copy()
    idx = np.flatnonzero(turn_of >= 0)
    for i in np.flatnonzero(turn_of < 0):
        pos = np.searchsorted(idx, i)
        left = turn_of[idx[pos - 1]] if pos > 0 else None
        right = turn_of[idx[pos]] if pos < len(idx) else None
        if left is not None and right is not None and left == right:
            filled[i] = left
    speaker = np.array([turns[t]["speaker"] if t >= 0 else None for t in filled], dtype=object)
    per_turn = [[] for _ in turns]
    for i, t in enumerate(turn_of):
        if t >= 0:
            per_turn[t].append(i)
    return speaker, filled, per_turn


def find_questions(turns: list[dict]) -> dict[str, int]:
    found, start = {}, 0
    for q, pat in QUESTIONS:
        for ti in range(start, len(turns)):
            if turns[ti]["speaker"] == "interviewer" and re.search(pat, turns[ti]["text"], flags=re.I):
                found[q] = ti
                start = ti + 1
                break
    return found


def answer_span(turns, per_turn, words, q_turn: int, next_q_turn: int | None):
    """(start_s, end_s, n_backchannels, first/last answer turn) for the interviewee block after q_turn."""
    stop = len(turns) if next_q_turn is None else next_q_turn
    ans_turns, inside, n_back, n_prompt = [], [], 0, 0
    for ti in range(q_turn + 1, stop):
        tr = turns[ti]
        if tr["speaker"] == "interviewer":
            n = len(tokens(tr["text"]))
            if n <= PROMPT_MAX:
                n_back += n <= BACKCHANNEL_MAX
                n_prompt += n > BACKCHANNEL_MAX
                if ans_turns:
                    inside.append(ti)
                continue
            if ans_turns:
                break
            continue  # interviewer elaborates the question before the answer starts
        ans_turns.append(ti)
    if ans_turns:  # prompts after the last answer turn are not inside the clip
        inside = [ti for ti in inside if ti < ans_turns[-1]]
    w_idx = [i for ti in ans_turns for i in per_turn[ti]]
    if not w_idx:
        return None
    start = min(words[i]["start"] for i in w_idx)
    end = max(words[i]["end"] for i in w_idx)
    q_words = per_turn[q_turn]
    q_end = max(words[i]["end"] for i in q_words) if q_words else np.nan
    n_tok = sum(len(tokens(turns[t]["text"])) for t in ans_turns)
    iv = [i for ti in inside for i in per_turn[ti]]
    iv_s = float(sum(words[i]["end"] - words[i]["start"] for i in iv))
    return {"start_s": start, "end_s": end, "question_end_s": q_end, "n_backchannels": n_back, "n_prompts": n_prompt,
            "interviewer_s_inside": iv_s,
            "answer_turns": len(ans_turns), "transcript_tokens": n_tok, "matched_tokens": len(w_idx)}


def step_segment() -> None:
    import soundfile as sf
    turns_all = load_turns()
    P.MIT_CLIPS.mkdir(parents=True, exist_ok=True)
    P.S7_TABLES.mkdir(parents=True, exist_ok=True)
    rows, align_rows = [], []
    for iid in interview_ids():
        f = P.CACHE_MIT / "asr" / f"{iid}_{ASR_MODEL}.json"
        if not f.exists() or iid not in turns_all:
            align_rows.append({"interview": iid, "status": "no asr" if not f.exists() else "no transcript"})
            continue
        words = json.loads(f.read_text(encoding="utf-8"))
        turns = turns_all[iid]
        speaker, _, per_turn = align(turns, words)
        t_tokens = sum(len(tokens(t["text"])) for t in turns)
        qs = find_questions(turns)
        align_rows.append({"interview": iid, "status": "ok", "asr_words": len(words), "transcript_tokens": t_tokens,
                           "asr_words_labelled": float(np.mean([s is not None for s in speaker])) if len(words) else 0,
                           "transcript_tokens_matched": sum(len(x) for x in per_turn) / max(t_tokens, 1),
                           "questions_found": "".join(q[1] for q in qs)})
        audio = mono16(iid)
        order = [q for q, _ in QUESTIONS if q in qs]
        for k, q in enumerate(order):
            nxt = qs[order[k + 1]] if k + 1 < len(order) else None
            span = answer_span(turns, per_turn, words, qs[q], nxt)
            if span is None or span["end_s"] - span["start_s"] < MIN_ANSWER_S:
                continue
            cid = f"MIT_{iid}_{q}"
            a, b = int(max(span["start_s"] - 0.2, 0) * SR), int(min(span["end_s"] + 0.5, len(audio) / SR) * SR)
            sf.write(P.MIT_CLIPS / f"{cid}.wav", audio[a:b], SR, subtype="PCM_16")
            pid = re.sub(r"\D", "", iid)
            rows.append({"id": cid, "interview": iid, "participant": f"MIT{pid}",
                         "session": "post" if iid.startswith("PP") else "pre", "question": q,
                         "question_text": turns[qs[q]]["text"][:160], "clip_start_s": a / SR, "clip_end_s": b / SR,
                         "audio_duration_s": (b - a) / SR, "audio_path": str(P.MIT_CLIPS / f"{cid}.wav"),
                         **{k2: v for k2, v in span.items() if k2 not in ("start_s", "end_s")},
                         "token_match_rate": span["matched_tokens"] / max(span["transcript_tokens"], 1)})
    pd.DataFrame(rows).to_csv(P.S7_TABLES / "mit_answers.csv", index=False)
    al = pd.DataFrame(align_rows)
    al.to_csv(P.S7_TABLES / "mit_alignment.csv", index=False)
    ans = pd.DataFrame(rows)
    print(f"{len(ans)} answers from {ans['interview'].nunique()} interviews; questions found per interview:")
    print(al["questions_found"].value_counts().head(8).to_string())
    print(ans[["audio_duration_s", "token_match_rate", "n_backchannels"]].describe().round(2).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=["asr", "segment"], required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    step_asr(args.limit) if args.step == "asr" else step_segment()


if __name__ == "__main__":
    main()
