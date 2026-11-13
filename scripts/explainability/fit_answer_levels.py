"""Stage 8, step 4: one level per answer for each feedback dimension, compared with OTHER PEOPLE's answers only.

No personal baseline ("than usual") and no numbers reach the feedback: a level and the reasons behind it in words.
Reference = RecruitView train participants, unique recordings. Every component becomes its percentile among
those answers (0-100, oriented so that higher = more of the dimension); a dimension's index is the mean of its
components' percentiles, re-expressed as a train percentile so the cuts below mean the same for every dimension.

  hesitation      fillers per 100 words (CrisperWhisper), repetitions + cut-off words per 100 words
                  (CrisperWhisper), share of 3-s windows the classifier calls "interjection", share called word
                  repetition / block (classifier), long mid-sentence pauses per minute (Stage 4 word timing)
  confidence      predicted confidence_score (WavLM L7 + Ridge; out-of-fold on train)
  expressiveness  pitch variability (st) and the inverse of the monotone share (the parts MIT supported;
                  loudness variability left out: recording-sensitive)
  length          speaking span vs answers to the same question type

Levels: index <= 20 -> "low", >= 80 -> "high", else "normal" (low/high are named per dimension below).
A component is listed as a reason when its own percentile is >= 80 (for a "high" hesitation level).
Quality gate (Stage 6): hesitation and expressiveness are withheld on RecruitView answers that fail the acoustic
gate (recording too poor). MIT is far-field and fails the gate almost everywhere; the gate is reported there but not
applied, so the MIT validation sees every answer.

Output  stage8_detectors/tables/answer_levels.csv (RecruitView + MIT), stage8_detectors/level_cuts.json
        (train reference quantiles per component: everything needed to score a new interview)
Run:    python scripts/explainability/fit_answer_levels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

LOW, HIGH = 20.0, 80.0
QS = np.linspace(0, 100, 201)

# dimension -> list of (component, sign); sign -1 means "lower value = more of the dimension"
DIMENSIONS = {
    "hesitation": [("fillers_per_100w", +1), ("restarts_per_100w", +1), ("p_interjection", +1),
                   ("p_repetition_block", +1), ("long_mid_pause_rate_per_min", +1)],
    "confidence": [("pred_confidence", +1)],
    "expressiveness": [("pitch_std_st", +1), ("monotony_frac", -1)],
    "length": [("speech_span_s", +1)],
}
BY_QUESTION_TYPE = {"length"}
LEVEL_NAMES = {"hesitation": ("little hesitation", "normal", "hesitated a lot"),
               "confidence": ("did not sound confident", "normal", "sounded confident"),
               "expressiveness": ("flat delivery", "normal", "lively delivery"),
               "length": ("short answer", "normal", "long answer")}
REASON_WORDS = {"fillers_per_100w": "many filler sounds (um / uh)",
                "restarts_per_100w": "repeated or restarted words",
                "p_interjection": "many filled pauses",
                "p_repetition_block": "stumbles (repetitions / blocked words)",
                "long_mid_pause_rate_per_min": "long pauses in the middle of sentences"}
GATED = {"hesitation", "expressiveness"}


def load() -> pd.DataFrame:
    seg_cols = ["id", "split", "participant", "question_type", "is_duplicate", "duplicate_of", "speech_span_s",
                "long_mid_pause_rate_per_min", "pitch_std_st", "monotony_frac"]
    rv = pd.read_csv(P.S4_SEGMENTS, dtype={"id": str, "duplicate_of": str}, usecols=seg_cols).assign(dataset="recruitview")
    gate = pd.read_csv(P.S6_TABLES / "quality_gate_per_clip.csv", dtype={"id": str}).set_index("id")["acoustic_ok"]
    ok = rv["id"].map(gate)
    rv["acoustic_ok"] = ok.fillna(rv["duplicate_of"].map(gate)).fillna(True).astype(bool)
    mit = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str})
    mit = mit[[*seg_cols, "acoustic_ok"]].assign(dataset="mit", participant=mit["participant"].astype(str))
    df = pd.concat([rv.assign(participant=rv["participant"].astype(str)), mit], ignore_index=True)

    dis = pd.read_csv(P.DISFLUENCY_ANSWERS, dtype={"id": str})
    dis["p_repetition_block"] = dis["p_Wordrepetition"] + dis["p_block"]
    cri = pd.read_csv(P.CRISPER_ANSWERS, dtype={"id": str})
    conf = pd.read_csv(P.CONFIDENCE_PRED, dtype={"id": str})
    conf = conf[conf["target"] == "confidence_score"][["id", "prediction"]].rename(columns={"prediction": "pred_confidence"})
    df = (df.merge(dis[["id", "n_windows", "p_interjection", "p_repetition_block"]], on="id", how="left")
            .merge(cri[["id", "n_words", "n_fillers", "fillers_per_100w", "restarts_per_100w"]], on="id", how="left")
            .merge(conf, on="id", how="left"))
    return df


def pct(ref: np.ndarray, x: pd.Series) -> pd.Series:
    """Percentile (0-100) of x within the reference values (mid-rank for ties)."""
    ref = np.sort(ref[np.isfinite(ref)])
    lo = np.searchsorted(ref, x.to_numpy(float), side="left")
    hi = np.searchsorted(ref, x.to_numpy(float), side="right")
    out = 100.0 * (lo + hi) / 2.0 / len(ref)
    return pd.Series(np.where(np.isfinite(x.to_numpy(float)), out, np.nan), index=x.index)


def main() -> None:
    df = load()
    ref_mask = (df["dataset"] == "recruitview") & (df["split"] == "train") & (~df["is_duplicate"].astype(bool))
    cuts = {"reference": "RecruitView train participants, unique recordings", "low": LOW, "high": HIGH,
            "components": {}, "index": {}, "by_question_type": sorted(BY_QUESTION_TYPE)}
    for dim, comps in DIMENSIONS.items():
        cols = []
        for comp, sign in comps:
            col = f"pct_{comp}"
            if dim in BY_QUESTION_TYPE:
                df[col] = np.nan
                cuts["components"][comp] = {}
                for qt, g in df.groupby("question_type"):
                    ref = df.loc[ref_mask & (df["question_type"] == qt), comp].to_numpy(float)
                    df.loc[g.index, col] = pct(ref, g[comp])
                    cuts["components"][comp][qt] = np.nanpercentile(ref, QS).round(4).tolist()
            else:
                ref = df.loc[ref_mask, comp].to_numpy(float)
                df[col] = pct(ref, df[comp])
                cuts["components"][comp] = np.nanpercentile(ref, QS).round(4).tolist()
            if sign < 0:
                df[col] = 100.0 - df[col]
            cols.append(col)
        raw = df[cols].mean(axis=1, skipna=False)
        df[f"{dim}_index"] = pct(raw[ref_mask].to_numpy(float), raw)
        cuts["index"][dim] = {"components": [c for c, _ in comps], "signs": [s for _, s in comps],
                              "raw_mean_quantiles": np.nanpercentile(raw[ref_mask], QS).round(4).tolist()}
        lo_name, mid, hi_name = LEVEL_NAMES[dim]
        idx = df[f"{dim}_index"]
        df[f"{dim}_level"] = np.select([idx <= LOW, idx >= HIGH], ["low", "high"], "normal")
        df.loc[idx.isna(), f"{dim}_level"] = "missing"
        df[f"{dim}_wording"] = df[f"{dim}_level"].map({"low": lo_name, "normal": mid, "high": hi_name, "missing": ""})
        withheld = (df["dataset"] == "recruitview") & (~df["acoustic_ok"].astype(bool)) & (dim in GATED)
        df[f"{dim}_withheld"] = withheld
    # reasons behind a "high" hesitation level, in words
    hcols = [f"pct_{c}" for c, _ in DIMENSIONS["hesitation"]]
    df["hesitation_reasons"] = df[hcols].apply(
        lambda r: "; ".join(REASON_WORDS[c[4:]] for c in hcols if r[c] >= HIGH), axis=1)
    df.loc[df["hesitation_level"] != "high", "hesitation_reasons"] = ""

    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(P.ANSWER_LEVELS, index=False)
    P.LEVEL_CUTS.write_text(json.dumps(cuts, indent=1), encoding="utf-8")
    print(f"wrote {P.ANSWER_LEVELS}: {len(df)} answers; {P.LEVEL_CUTS}")
    for dim in DIMENSIONS:
        print(dim, df.groupby("dataset")[f"{dim}_level"].value_counts(normalize=True).unstack().round(2).to_string(), sep="\n")
    print("withheld (RecruitView, gate):", {d: int(df[f"{d}_withheld"].sum()) for d in GATED})


if __name__ == "__main__":
    main()
