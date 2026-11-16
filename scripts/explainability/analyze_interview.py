"""Phase 4, step 2: one full interview recording in, the evidence JSON of every answer and the interview summary
(Phase 5, interview_summary.py) out.

  1. load the recording (any format) -> 16-kHz mono
  2. segment_interview.py: Whisper over the whole recording, speaker split, the five questions, answer clips
  3. analyze_answer.py on every answer clip, with
       - the candidate's Whisper words from step 2 (shifted to clip time; interviewer words dropped), so Whisper runs
         only once, as for the MIT dataset answers
       - one speaker context for the interview: pitch range and typical pitch from ALL the candidate's answers
         (the expressiveness voice-pitch adjustment needs this)
  4. the interview_summary block (interview_summary.summarize): per dimension an interview level against
     interview_cuts.json, the answers worth pointing at, the trend word where Phase 5 allows it, and the response
     latency (live_interview only; time from the interviewer's last word to the answer's first word)
  5. writes {out}/{name}.json: interview id, setting, answers (question, clip times, evidence) and the interview
     summary; and {out}/{name}_measures.csv (measurements, percentiles, levels, latency; internal)

Setting: live_interview by default (level_cuts.json "default_setting"); recorded_answer for self-recorded clips.

GPU: about 2 GB peak (models one at a time). A 5-minute interview takes about 3-4 minutes on the laptop GPU.

Run:  python scripts/explainability/analyze_interview.py interview.wav [--name P1] [--out DIR]
      [--setting live_interview|recorded_answer]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import analyze_answer as AA  # noqa: E402
import interview_summary as IS  # noqa: E402
import segment_interview as SI  # noqa: E402
from run_ser_windows import cap_gpu_memory, report_gpu_peak  # noqa: E402

SR = AA.SR


def clip_words(words: list[dict], word_speaker: list[str], start: float, end: float) -> list[dict]:
    """The candidate's words inside [start, end], in clip time (as mit_run_pipeline.clip_words)."""
    return [{**w, "start": round(w["start"] - start, 3), "end": round(w["end"] - start, 3)}
            for w, s in zip(words, word_speaker)
            if w["start"] >= start and w["end"] <= end and s != "interviewer"]


def interview_block(seg, table, info, setting: str) -> dict:
    """The Phase 5 interview_summary for a new recording, against interview_cuts.json."""
    import numpy as np
    import pandas as pd
    cuts = IS.load_cuts(setting)
    t = pd.DataFrame({"question_id": seg["question"].to_numpy()})
    for dim in IS.DIMS:
        idx = table[f"{dim}_index"].to_numpy(float).copy()
        if setting == "recorded_answer":          # the quality gate withholds the level in this setting
            idx[~table["acoustic_ok"].astype(bool).to_numpy()] = np.nan
        t[f"{dim}_index"] = idx
        t[f"{dim}_level"] = np.where(np.isfinite(idx), table[f"{dim}_level"].to_numpy(object), "missing")
    if setting == "live_interview":
        t["latency_s"] = [IS.response_latency(info["words"], info["word_speaker"], st)
                          for st in seg["answer_start_s"]]
        table["latency_s"] = t["latency_s"].to_numpy()
    use = dict(cuts["use"])
    if setting != "live_interview":             # trends and latency were tested on live interviews only
        use = {k: (False if k == "latency" else "not_used") for k in use}
    q = cuts["quantiles"]
    summary = IS.summarize(t, setting, use, cuts["question_means"],
                           lambda name, v: AA.pct_from_quantiles(q[name], v) if name in q else float("nan"))
    summary["status"] = "Phase 5; answer boundaries from segment_interview.py are not validated (Phase 4, test 2)"
    return summary


def run(path: str, name: str | None = None, setting: str | None = None, out_dir: Path | None = None,
        words: list[dict] | None = None) -> dict:
    t0 = time.time()
    name = name or Path(path).stem
    audio = AA.load_audio(path)
    print(f"{name}: {len(audio) / SR / 60:.1f} min of audio", flush=True)
    seg, info = SI.segment(audio, words)
    print(f"  segmentation: questions found {info['questions_found'] or 'none'}, {len(seg)} answers "
          f"({time.time() - t0:.0f} s)", flush=True)
    answers, cw = [], {}
    for r in seg.itertuples():
        aid = f"{name}_{r.question}"
        a, b = int(r.clip_start_s * SR), int(r.clip_end_s * SR)
        answers.append(AA.Answer(id=aid, audio=audio[a:b], question=r.question_text, offset_s=float(r.clip_start_s),
                                 speaker_key="candidate"))
        cw[aid] = clip_words(info["words"], info["word_speaker"], r.clip_start_s, r.clip_end_s)
    result = {"interview_id": name, "setting": setting or json.loads(P.LEVEL_CUTS.read_text())["default_setting"],
              "modality": "audio", "answers": [], "interview_summary": None}
    if answers:
        evs, table, speakers = AA.analyze(answers, setting, words=cw)
        result["setting"] = evs[0]["setting"]
        for r, ev in zip(seg.itertuples(), evs):
            result["answers"].append({"question_id": r.question, "clip": {"start": AA.BAE.mmss(r.clip_start_s),
                                                                          "end": AA.BAE.mmss(r.clip_end_s)},
                                      "evidence": ev})
        result["interview_summary"] = interview_block(seg, table, info, result["setting"])
        sp = speakers["candidate"]
        result["internal"] = {"speaker_pitch_range_hz": [sp.pitch_floor_hz, sp.pitch_ceiling_hz],
                              "segmentation": {k: info[k] for k in ("n_words", "n_chunks", "n_turns", "questions_found")},
                              "runtime_s": round(time.time() - t0, 1)}
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_dir / f"{name}_measures.csv", index=False)
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                              encoding="utf-8")
        print(f"wrote {out_dir / f'{name}.json'}", flush=True)
    print(f"  total {time.time() - t0:.0f} s", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--name")
    ap.add_argument("--setting", choices=["live_interview", "recorded_answer"])
    ap.add_argument("--out", default=str(P.S12 / "examples"))
    ap.add_argument("--max-gpu-frac", type=float, default=0.8)
    args = ap.parse_args()
    cap_gpu_memory(args.max_gpu_frac)
    res = run(args.audio, args.name, args.setting, Path(args.out))
    report_gpu_peak()
    for a in res["answers"]:
        obs = "; ".join(f"{o['dimension']}: {o['say']}" for o in a["evidence"]["observations"]) or "nothing to mention"
        print(f"  {a['question_id']} [{a['clip']['start']}-{a['clip']['end']}] {obs}")


if __name__ == "__main__":
    main()
