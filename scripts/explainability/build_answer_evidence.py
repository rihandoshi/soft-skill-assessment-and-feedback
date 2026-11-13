"""Stage 9 input: evidence JSON v2, one object per answer, for the LLM that writes coach-style feedback.

Built from Stage 8's answer levels (compared with other people's answers only; no personal baseline). What the
LLM gets per answer:
  question            the question text (so it can say "when you were asked ...")
  observations        only dimensions whose level is clear (low / high) AND that passed the MIT validation, each
                      with: what to say in plain words, issue or strength, the reasons in words, and for hesitation
                      an example excerpt (verbatim, fillers written as "uh" / "um") with its rough time
  not_assessed        dimensions withheld (recording quality) or not validated, so the LLM does not guess them
No numbers, percentiles or "than usual" comparisons are included in the wording fields; the numeric levels stay in
`_debug` for the faithfulness check and are not to be quoted.

Which dimensions may be used is read from stage8_detectors/tables/validation_mit.csv (the key test of each dimension
must be "supported"); confidence additionally relies on RecruitView (it is the confidence-score model).

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
            "expressiveness": ("expressiveness_index", "EngagingTone"),
            "length": ("length_index", "StructuredAnswers")}
# level -> (kind, plain words). Levels not listed are not mentioned.
SAY = {"hesitation": {"high": ("issue", "hesitated a lot"), "low": ("strength", "spoke smoothly, with very little hesitation")},
       "confidence": {"low": ("issue", "didn't sound very confident"), "high": ("strength", "sounded confident")},
       "expressiveness": {"low": ("issue", "sounded quite flat (little variation in tone)"),
                          "high": ("strength", "sounded lively and engaging")},
       "length": {"high": ("issue", "the answer ran long")}}
TIPS = {"hesitation": "pause silently instead of saying 'um', and plan the first sentence before starting",
        "confidence": "slow down slightly, finish sentences with a falling tone, and avoid trailing off",
        "expressiveness": "stress the key words and let the pitch rise and fall more",
        "length": "use a short structure (situation, action, result) and stop after the result"}
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


def hesitation_example(words: pd.DataFrame) -> dict | None:
    """The chunk with the most fillers + repetitions; a short verbatim excerpt around the first of them."""
    if words.empty:
        return None
    marks = words["is_filler"] | words["is_repetition"] | words["is_fragment"]
    per_chunk = marks.groupby(words["chunk"]).sum()
    if per_chunk.max() < 2:
        return None
    c = words[words["chunk"] == per_chunk.idxmax()].reset_index(drop=True)
    first = int(np.flatnonzero((c["is_filler"] | c["is_repetition"] | c["is_fragment"]).to_numpy())[0])
    excerpt = c.iloc[max(0, first - 4): first + 14]["word"].str.replace(r"\[UH\]", "uh", regex=True) \
                                                         .str.replace(r"\[UM\]", "um", regex=True)
    return {"excerpt": " ".join(excerpt), "around": f"{mmss(c['chunk_start'].iloc[0])}-{mmss(c['chunk_end'].iloc[0])}"}


def questions() -> dict[str, str]:
    q = {}
    for s in ("train", "val", "test"):
        d = pd.read_csv(P.ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str}, usecols=["id", "question"])
        q.update(dict(zip(d["id"], d["question"])))
    m = pd.read_csv(P.S7_TABLES / "mit_answers.csv", dtype={"id": str})
    q.update(dict(zip(m["id"], m["question_text"].str.strip())))
    return q


def main() -> None:
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str})
    words = pd.read_csv(P.CRISPER_WORDS, dtype={"id": str})
    words_by_id = dict(tuple(words.groupby("id")))
    use = usable_dimensions()
    qtext = questions()
    P.S9.mkdir(parents=True, exist_ok=True)
    P.EVIDENCE_V2_EXAMPLES.mkdir(parents=True, exist_ok=True)
    n_obs = []
    with P.EVIDENCE_V2.open("w", encoding="utf-8") as fh:
        for r in lv.itertuples():
            obs, not_assessed, debug = [], [], {}
            for dim in KEY_TEST:
                level = getattr(r, f"{dim}_level")
                debug[dim] = {"level": level, "index": None if pd.isna(getattr(r, f"{dim}_index"))
                              else round(float(getattr(r, f"{dim}_index")), 1)}
                if not use[dim]:
                    not_assessed.append({"dimension": dim, "why": "not validated against human ratings"})
                    continue
                if getattr(r, f"{dim}_withheld"):
                    not_assessed.append({"dimension": dim, "why": "recording quality too low to judge"})
                    continue
                if level not in SAY[dim]:
                    continue
                kind, words_say = SAY[dim][level]
                o = {"dimension": dim, "kind": kind, "say": words_say}
                if kind == "issue":
                    o["tip"] = TIPS[dim]
                if dim == "hesitation" and level == "high":
                    o["reasons"] = [x for x in str(r.hesitation_reasons or "").split("; ") if x and x != "nan"]
                    ex = hesitation_example(words_by_id.get(r.id, pd.DataFrame()))
                    if ex:
                        o["example"] = ex
                obs.append(o)
            ev = {"answer_id": r.id, "dataset": r.dataset, "question": qtext.get(r.id, ""),
                  "observations": obs, "not_assessed": not_assessed,
                  "rules": "compared with other candidates' answers only; mention only the observations listed",
                  "_debug": debug}
            fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            n_obs.append(len(obs))
            if r.id in EXAMPLES or (r.dataset == "recruitview" and r.split == "test" and len(obs) >= 2
                                    and len(list(P.EVIDENCE_V2_EXAMPLES.glob("*.json"))) < 6):
                (P.EVIDENCE_V2_EXAMPLES / f"{r.id}.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False),
                                                                    encoding="utf-8")
    n_obs = np.asarray(n_obs)
    print(f"wrote {P.EVIDENCE_V2}: {len(n_obs)} answers; usable dimensions {use}")
    print(f"observations per answer: none {np.mean(n_obs == 0):.0%}, 1 {np.mean(n_obs == 1):.0%}, "
          f"2+ {np.mean(n_obs >= 2):.0%}")


if __name__ == "__main__":
    main()
