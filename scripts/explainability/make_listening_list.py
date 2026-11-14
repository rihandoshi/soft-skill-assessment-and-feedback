"""Phase 1, step 8: the answers to listen to before the levels are used (a bug check, not labelling).

Per skill (hesitation, expressiveness) and per end (high, low): the 10 most extreme test-split RecruitView answers
and the 5 most extreme MIT answers, one answer per person, recordings that pass the quality gate only. Each row has
the audio path, the time span the evidence JSON points at, and the reasons in words, plus empty columns to fill in
while listening:
  heard_ok    yes / no: does the level match what you hear?
  problem     if no: what is wrong (e.g. music, two speakers, cut-off audio, accent mistaken for hesitation)

Output  stage8_detectors/listening_check.csv
Run:    python scripts/explainability/make_listening_list.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import run_disfluency_classifier as D  # noqa: E402

N = {"recruitview": 10, "mit": 5}


def main() -> None:
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str})
    audio = D.all_clips().set_index("id")["audio_path"]
    ev = {}
    with P.EVIDENCE_V2.open(encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            ev[e["answer_id"]] = {o["dimension"]: o for o in e["observations"]}
    pool = lv[((lv["dataset"] == "recruitview") & (lv["split"] == "test") & ~lv["is_duplicate"].astype(bool)
               & lv["acoustic_ok"].astype(bool)) | (lv["dataset"] == "mit")]
    rows = []
    for dim in ["hesitation", "expressiveness"]:
        for level, ascending in (("high", False), ("low", True)):
            for ds, n in N.items():
                g = pool[(pool["dataset"] == ds) & (pool[f"{dim}_level"] == level)]
                g = g.sort_values(f"{dim}_index", ascending=ascending).drop_duplicates("participant").head(n)
                for r in g.itertuples():
                    o = ev.get(r.id, {}).get(dim, {})
                    rows.append({"skill": dim, "level": level, "dataset": ds, "id": r.id,
                                 "index": round(getattr(r, f"{dim}_index"), 1),
                                 "where": f"{o['where']['start']}-{o['where']['end']}" if "where" in o else "",
                                 "reasons": "; ".join(o.get("reasons", [])), "audio_path": audio.get(r.id, ""),
                                 "heard_ok": "", "problem": ""})
    out = P.S8 / "listening_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}: {len(rows)} answers")
    print(pd.DataFrame(rows).groupby(["skill", "level", "dataset"]).size().to_string())


if __name__ == "__main__":
    main()
