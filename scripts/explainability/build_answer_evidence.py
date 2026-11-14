"""Stage 9 input: evidence JSON v2 (Phase 1 version), one object per answer, for the LLM that writes coach-style
feedback and for fusion.

Built from Stage 8's answer levels (compared with other people's answers only; no personal baseline). Audio only:
no words, quotes or filler counts (those belong to the text branch), no answer length judgement (text branch).
What the LLM gets per answer:
  question            the question text (so it can say "when you were asked ...")
  setting, modality   recorded_answer (RecruitView) / live_interview (MIT); always "audio"
  observations        only dimensions whose level is clear (low / high) AND that passed the MIT validation, each
                      with: level, issue or strength, what to say in plain words, the reasons in words, where in the
                      answer it was strongest (start / end time), and a tip for issues
  not_assessed        dimensions withheld (recording quality) or not validated, so the LLM does not guess them
  internal            the 0-100 indices and the speaking time, for fusion and the faithfulness check only; never
                      quoted to the user

Where:
  hesitation      the run of 3 consecutive disfluency-classifier windows (3 s, 1.5-s hop, about 6 s) with the
                  highest mean non-fluent probability (vocametrix 1 - p(fluent) and speech-flow p(disfluent),
                  averaged)
  expressiveness  the run of 3 consecutive emotion-model windows (4 s, 2-s hop, about 8 s, >= 50% speech) with the
                  lowest mean activation (for "flat") or the highest (for "lively")

Which dimensions may be used is read from stage8_detectors/tables/validation_mit.csv (the key test of each dimension
must be "supported").

Output  stage9_feedback/evidence_v2.jsonl (all answers), stage9_feedback/evidence_v2_examples/*.json (a few)
Run:    python scripts/explainability/build_answer_evidence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

KEY_TEST = {"hesitation": ("hesitation_index", "NoFillers"),
            "confidence": ("pred_confidence", "Overall"),
            "expressiveness": ("expressiveness_index", "EngagingTone")}
# level -> (kind, plain words). Levels not listed are not mentioned.
SAY = {"hesitation": {"high": ("issue", "hesitated a lot"), "low": ("strength", "spoke smoothly, with very little hesitation")},
       "confidence": {"low": ("issue", "didn't sound very confident"), "high": ("strength", "sounded confident")},
       "expressiveness": {"low": ("issue", "sounded quite flat"), "high": ("strength", "sounded lively and engaged")}}
TIPS = {"hesitation": "plan the first sentence before starting, and when you need a moment, pause silently",
        "confidence": "slow down slightly, finish sentences with a falling tone, and avoid trailing off",
        "expressiveness": "stress the key words and let the voice rise and fall more"}
SETTING = {"recruitview": "recorded_answer", "mit": "live_interview"}
RUN = 3  # consecutive windows in a "where" stretch
EXAMPLES = ["MIT_P8_Q3", "MIT_P1_Q1"]


def usable_dimensions() -> dict[str, bool]:
    v = pd.read_csv(P.S8_TABLES / "validation_mit.csv")
    out = {}
    for dim, (m, t) in KEY_TEST.items():
        row = v[(v["measure"] == m) & (v["target"] == t)]
        out[dim] = bool(len(row)) and row["verdict"].iloc[0] == "supported"
    return out


def mmss(s: float) -> str:
    return f"{int(s // 60)}:{int(s % 60):02d}"


def stretch(w: pd.DataFrame, value: str, highest: bool) -> dict | None:
    """Start / end of the run of RUN consecutive windows with the highest (or lowest) mean `value`."""
    if w is None or w.empty:
        return None
    w = w.sort_values("start").reset_index(drop=True)
    k = min(RUN, len(w))
    m = w[value].rolling(k).mean().iloc[k - 1:]
    i = int(m.idxmax() if highest else m.idxmin())
    return {"start": mmss(w.loc[i - k + 1, "start"]), "end": mmss(w.loc[i, "end"])}


def load_windows() -> tuple[dict, dict]:
    dis = pd.read_csv(P.DISFLUENCY_WINDOWS, dtype={"id": str})
    sf = pd.read_csv(P.S8_TABLES / "speech_flow_windows.csv", dtype={"id": str})[["id", "start", "p_disfluent"]]
    dis = dis.merge(sf.round({"start": 3}), on=["id", "start"], how="left")   # identical windows
    # the two classifiers averaged, as in the hesitation index; vocametrix alone where speech-flow is missing
    dis["non_fluent"] = ((1.0 - dis["p_fluent"]) + dis["p_disfluent"].fillna(1.0 - dis["p_fluent"])) / 2.0
    ser = pd.concat([pd.read_csv(P.s5_windows("audeering"), dtype={"id": str}),
                     pd.read_csv(P.S7_TABLES / "mit_windows_audeering.csv", dtype={"id": str})], ignore_index=True)
    ser = ser[ser["speech_frac"] >= 0.5].rename(columns={"t_start": "start", "t_end": "end"})
    ser["activation"] = (ser["arousal"] + ser["dominance"]) / 2.0
    return dict(tuple(dis.groupby("id"))), dict(tuple(ser.groupby("id")))


def questions() -> dict[str, str]:
    q = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(P.ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str}, usecols=["id", "question"])
        q.update(dict(zip(d["id"], d["question"])))
    m = pd.read_csv(P.S7_TABLES / "mit_answers.csv", dtype={"id": str})
    q.update(dict(zip(m["id"], m["question_text"].str.strip())))
    return q


def num(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 1)


def main() -> None:
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str, "duplicate_of": str})
    dis_w, ser_w = load_windows()
    use = usable_dimensions()
    qtext = questions()
    P.S9.mkdir(parents=True, exist_ok=True)
    P.EVIDENCE_V2_EXAMPLES.mkdir(parents=True, exist_ok=True)
    for f in P.EVIDENCE_V2_EXAMPLES.glob("*.json"):
        f.unlink()
    n_obs, n_where = [], [0, 0]
    with P.EVIDENCE_V2.open("w", encoding="utf-8") as fh:
        for r in lv.itertuples():
            src = r.id if (r.id in dis_w or r.id in ser_w) or pd.isna(r.duplicate_of) else r.duplicate_of
            obs, not_assessed = [], []
            for dim in KEY_TEST:
                level = getattr(r, f"{dim}_level")
                if not use[dim]:
                    not_assessed.append({"dimension": dim, "why": "not validated against human ratings"})
                    continue
                if getattr(r, f"{dim}_withheld"):
                    not_assessed.append({"dimension": dim, "why": "recording quality too low to judge"})
                    continue
                if level not in SAY[dim]:
                    continue
                kind, words_say = SAY[dim][level]
                o = {"dimension": dim, "level": level, "kind": kind, "say": words_say}
                reasons = str(getattr(r, f"{dim}_reasons", "") or "")
                o["reasons"] = [x for x in reasons.split("; ") if x and x != "nan"]
                if dim == "hesitation":
                    where = stretch(dis_w.get(src), "non_fluent", highest=(level == "high"))
                else:
                    where = stretch(ser_w.get(src), "activation", highest=(level == "high"))
                if where:
                    o["where"] = where
                n_where[0] += 1
                n_where[1] += bool(where)
                if kind == "issue":
                    o["tip"] = TIPS[dim]
                obs.append(o)
            ev = {"answer_id": r.id, "question": qtext.get(r.id, ""), "setting": SETTING[r.dataset],
                  "modality": "audio", "observations": obs, "not_assessed": not_assessed,
                  "rules": "compared with other candidates' answers only; mention only the observations listed",
                  "internal": {"hesitation": num(r.hesitation_index), "expressiveness": num(r.expressiveness_index),
                               "confidence": num(r.confidence_index), "speech_span_s": num(r.speech_span_s)}}
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            n_obs.append(len(obs))
            if r.id in EXAMPLES or (r.dataset == "recruitview" and r.split == "test" and len(obs) >= 2
                                    and len(list(P.EVIDENCE_V2_EXAMPLES.glob("*.json"))) < 6):
                (P.EVIDENCE_V2_EXAMPLES / f"{r.id}.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8")
    n_obs = np.asarray(n_obs)
    print(f"wrote {P.EVIDENCE_V2}: {len(n_obs)} answers; usable dimensions {use}")
    print(f"observations per answer: none {np.mean(n_obs == 0):.0%}, 1 {np.mean(n_obs == 1):.0%}, "
          f"2+ {np.mean(n_obs >= 2):.0%}; observations with a time span {n_where[1]}/{n_where[0]}")


if __name__ == "__main__":
    main()
