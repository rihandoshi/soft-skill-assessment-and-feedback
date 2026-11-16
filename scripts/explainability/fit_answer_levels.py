"""Stage 8, step 4 (Phase 1 version): one level per answer for each audio feedback dimension, compared with OTHER
PEOPLE's answers only.

No personal baseline ("than usual") and no numbers reach the feedback: a level and the reasons behind it in words.
Reference = RecruitView train participants, unique recordings. Every component becomes its percentile among
those answers (0-100, oriented so that higher = more of the dimension); a dimension's index is the mean of its
components' percentiles, re-expressed as a train percentile so the cuts below mean the same for every dimension.

  hesitation      (the "fluency" skill, higher = more hesitant; audio only since Phase 1)
                  mean "interjection" probability of the disfluency classifier over 3-s windows,
                  mean "word repetition" + "block" probability (same windows),
                  mean "disfluent" probability of the Vox-Profile speech-flow model over the same windows,
                  long (>= 1 s) mid-sentence silent pauses per minute (Silero VAD + ASR sentences);
                  weighted 1/6, 1/6, 1/3, 1/3 (the two classifiers averaged, see WEIGHTS)
  confidence      predicted confidence_score (WavLM L7 + Ridge; out-of-fold on train). Kept for the record; it was
                  contradicted on MIT, so build_answer_evidence.py does not use it
  expressiveness  vocal activation = mean of arousal and dominance of the audeering emotion model over 4-s windows,
                  adjusted for the speaker's typical pitch: activation is regressed on the speaker's median F0 (Hz)
                  in the reference population and the residual is used. Pitch variability is kept only for the
                  reason in words.

Answer length is no longer a level (the text branch judges whether an answer ran long); `speech_span_s` stays in
the table for the text branch and fusion. The CrisperWhisper filler / restart columns also stay in the table, for
the text branch, but are not part of any audio level.

Levels: index <= 20 -> "low", >= 80 -> "high", else "normal" (low/high are named per dimension below).
A component is listed as a reason when its own percentile is >= 80 (for a "high" hesitation level).
Quality gate (Stage 6): hesitation and expressiveness are withheld on RecruitView answers that fail the acoustic
gate (recording too poor). MIT is far-field and fails the gate almost everywhere; the gate is reported there but not
applied, so the MIT validation sees every answer.

Speaker's typical pitch: the median over the speaker's answers of each answer's median F0 (RecruitView: all clips of
the participant; MIT: all answers of the interview; a new recording: all answers of that interview).

Output  stage8_detectors/tables/answer_levels.csv (RecruitView + MIT), stage8_detectors/level_cuts.json
        (train reference quantiles per component and the voice-pitch adjustment: everything needed to score a new
        interview), stage8_detectors/tables/flat_by_voice_pitch.csv (Phase 1 check: "flat" rate by voice pitch)
Run:    python scripts/explainability/fit_answer_levels.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

LOW, HIGH = 20.0, 80.0
QS = np.linspace(0, 100, 201)

# dimension -> list of (component, sign); sign -1 means "lower value = more of the dimension"
DIMENSIONS = {
    "hesitation": [("p_interjection", +1), ("p_repetition_block", +1), ("sf_p_disfluent", +1),
                   ("long_mid_pause_rate_per_min", +1)],
    "confidence": [("pred_confidence", +1)],
    "expressiveness": [("activation_adj", +1)],
}
# Component weights in the index (default 1). Hesitation: the two disfluency classifiers are averaged (vocametrix's
# two outputs share one half, the Vox-Profile speech-flow model the other) and together keep the 2/3 share the
# classifier had before; mid-sentence pauses keep 1/3. Fixed before looking at the MIT result of the combination.
WEIGHTS = {"p_interjection": 1 / 6, "p_repetition_block": 1 / 6, "sf_p_disfluent": 1 / 3,
           "long_mid_pause_rate_per_min": 1 / 3}
LEVEL_NAMES = {"hesitation": ("little hesitation", "normal", "hesitated a lot"),
               "confidence": ("did not sound confident", "normal", "sounded confident"),
               "expressiveness": ("flat delivery", "normal", "lively delivery")}
REASON_WORDS = {"p_interjection": "many filled pauses",
                "p_repetition_block": "stumbles (repeated or blocked words)",
                "long_mid_pause_rate_per_min": "long pauses in the middle of sentences"}
PITCH_REASON = {"low": "little rise and fall in the voice", "high": "plenty of rise and fall in the voice"}
GATED = {"hesitation", "expressiveness"}
SETTING_OF_DATASET = {"recruitview": "recorded_answer", "mit": "live_interview"}


def load() -> pd.DataFrame:
    seg_cols = ["id", "split", "participant", "question_type", "is_duplicate", "duplicate_of", "speech_span_s",
                "long_mid_pause_rate_per_min", "pitch_std_st", "monotony_frac", "f0_median_hz"]
    rv = pd.read_csv(P.S4_SEGMENTS, dtype={"id": str, "duplicate_of": str}, usecols=seg_cols).assign(dataset="recruitview")
    gate = pd.read_csv(P.S6_TABLES / "quality_gate_per_clip.csv", dtype={"id": str}).set_index("id")["acoustic_ok"]
    ok = rv["id"].map(gate)
    rv["acoustic_ok"] = ok.fillna(rv["duplicate_of"].map(gate)).fillna(True).astype(bool)
    ser = pd.read_csv(P.S5_SEGMENTS, dtype={"id": str}, usecols=["id", "ser_arousal_mean", "ser_dominance_mean"])
    rv = rv.merge(ser, on="id", how="left")
    mit = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str})
    mit = mit[[*seg_cols, "person", "ser_arousal_mean", "ser_dominance_mean", "acoustic_ok"]].assign(
        dataset="mit", participant=mit["participant"].astype(str))
    df = pd.concat([rv.assign(participant=rv["participant"].astype(str)), mit], ignore_index=True)

    dis = pd.read_csv(P.DISFLUENCY_ANSWERS, dtype={"id": str})
    dis["p_repetition_block"] = dis["p_Wordrepetition"] + dis["p_block"]
    cri = pd.read_csv(P.CRISPER_ANSWERS, dtype={"id": str})
    conf = pd.read_csv(P.CONFIDENCE_PRED, dtype={"id": str})
    conf = conf[conf["target"] == "confidence_score"][["id", "prediction"]].rename(columns={"prediction": "pred_confidence"})
    df = (df.merge(dis[["id", "n_windows", "p_interjection", "p_repetition_block"]], on="id", how="left")
            .merge(cri[["id", "n_words", "n_fillers", "fillers_per_100w", "restarts_per_100w"]], on="id", how="left")
            .merge(conf, on="id", how="left"))
    sf = pd.read_csv(P.S8_TABLES / "speech_flow_answers.csv", dtype={"id": str})
    df = df.merge(sf[["id", "sf_p_disfluent", "sf_share_disfluent"]], on="id", how="left")

    df["activation"] = (df["ser_arousal_mean"] + df["ser_dominance_mean"]) / 2.0
    df["speaker_f0_hz"] = df.groupby(["dataset", "participant"])["f0_median_hz"].transform("median")
    return df


def pct(ref: np.ndarray, x: pd.Series) -> pd.Series:
    """Percentile (0-100) of x within the reference values (mid-rank for ties)."""
    ref = np.sort(ref[np.isfinite(ref)])
    lo = np.searchsorted(ref, x.to_numpy(float), side="left")
    hi = np.searchsorted(ref, x.to_numpy(float), side="right")
    out = 100.0 * (lo + hi) / 2.0 / len(ref)
    return pd.Series(np.where(np.isfinite(x.to_numpy(float)), out, np.nan), index=x.index)


def fit_pitch_adjustment(df: pd.DataFrame, ref_mask: pd.Series) -> dict:
    """activation ~ a + b * speaker_f0_hz on the reference answers; returns the coefficients."""
    d = df.loc[ref_mask, ["activation", "speaker_f0_hz"]].dropna()
    b, a = np.polyfit(d["speaker_f0_hz"], d["activation"], 1)
    return {"intercept": float(a), "slope_per_hz": float(b), "fill_f0_hz": float(d["speaker_f0_hz"].median()),
            "n": int(len(d))}


def apply_pitch_adjustment(df: pd.DataFrame, adj: dict) -> pd.Series:
    f0 = df["speaker_f0_hz"].fillna(adj["fill_f0_hz"])  # no pitch track at all: treat as a typical voice
    return df["activation"] - (adj["intercept"] + adj["slope_per_hz"] * f0)




def score(df: pd.DataFrame, target: pd.Series, ref: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Levels for the `target` rows against the `ref` rows (boolean masks over df).

    Returns the per-answer columns for the target rows and the cuts (reference quantiles, pitch adjustment) that
    reproduce them for a new answer. Everything fitted (pitch slope, percentiles, index quantiles) uses ref only."""
    d = df.copy()
    adj = fit_pitch_adjustment(d, ref)
    d["activation_adj"] = apply_pitch_adjustment(d, adj)
    cuts = {"low": LOW, "high": HIGH, "n_reference_answers": int(ref.sum()), "components": {}, "index": {},
            "expressiveness_pitch_adjustment": {**adj, "formula": "activation_adj = activation - (intercept + "
                                                "slope_per_hz * speaker_f0_hz); activation = mean(arousal, dominance)"}}
    out = pd.DataFrame(index=d.index[target])
    out["activation_adj"] = d.loc[target, "activation_adj"]
    for dim, comps in DIMENSIONS.items():
        cols = []
        for comp, sign in comps:
            col = f"pct_{comp}"
            r = d.loc[ref, comp].to_numpy(float)
            d[col] = pct(r, d[comp])
            cuts["components"][comp] = np.nanpercentile(r, QS).round(5).tolist()
            if sign < 0:
                d[col] = 100.0 - d[col]
            cols.append(col)
            out[col] = d.loc[target, col]
        w = np.array([WEIGHTS.get(c, 1.0) for c, _ in comps])
        raw = pd.Series(d[cols].to_numpy(float) @ (w / w.sum()), index=d.index)  # NaN if any component is missing
        idx = pct(raw[ref].to_numpy(float), raw[target])
        cuts["index"][dim] = {"components": [c for c, _ in comps], "signs": [s for _, s in comps],
                              "weights": (w / w.sum()).round(6).tolist(),
                              "raw_mean_quantiles": np.nanpercentile(raw[ref], QS).round(5).tolist()}
        out[f"{dim}_index"] = idx
        lvl = pd.Series(np.select([idx <= LOW, idx >= HIGH], ["low", "high"], "normal"), index=idx.index)
        lvl[idx.isna()] = "missing"
        out[f"{dim}_level"] = lvl
    # reasons behind a "high" hesitation level, in words
    hcols = [f"pct_{c}" for c, _ in DIMENSIONS["hesitation"] if c in REASON_WORDS]  # speech-flow: no own wording
    out["hesitation_reasons"] = out[hcols].apply(
        lambda r: "; ".join(REASON_WORDS[c[4:]] for c in hcols if r[c] >= HIGH), axis=1)
    out.loc[out["hesitation_level"] != "high", "hesitation_reasons"] = ""
    # expressiveness: pitch variability is the reason in words, never the detector
    r = d.loc[ref, "pitch_std_st"].to_numpy(float)
    out["pct_pitch_std_st"] = pct(r, d.loc[target, "pitch_std_st"])
    cuts["components"]["pitch_std_st"] = np.nanpercentile(r, QS).round(5).tolist()
    lvl, pp = out["expressiveness_level"], out["pct_pitch_std_st"]
    out["expressiveness_reasons"] = np.select([(lvl == "low") & (pp <= LOW), (lvl == "high") & (pp >= HIGH)],
                                              [PITCH_REASON["low"], PITCH_REASON["high"]], "")
    # the unadjusted activation percentile, only for the voice-pitch check
    out["pct_activation_raw"] = pct(d.loc[ref, "activation"].to_numpy(float), d.loc[target, "activation"])
    return out, cuts


def score_leave_one_person_out(df: pd.DataFrame, rows: pd.Series, pool: pd.Series, person: pd.Series) -> pd.DataFrame:
    """Each person's answers scored against the pool without that person (both of their interviews left out)."""
    parts = []
    for p in person[rows].unique():
        mine = rows & (person == p)
        parts.append(score(df, mine, pool & (person != p))[0])
    return pd.concat(parts)


def flat_by_voice_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """Share of "flat" (expressiveness low) by speaker-pitch tercile, within each setting. Gated answers excluded."""
    rows = []
    ok = ~df["expressiveness_withheld"] & df["expressiveness_level"].ne("missing")
    for st, g in df[ok].groupby("setting"):
        g = g.dropna(subset=["speaker_f0_hz"])
        third = pd.qcut(g["speaker_f0_hz"], 3, labels=["low voices", "middle voices", "high voices"])
        for name, h in g.groupby(third, observed=True):
            rows.append({"setting": st, "voice_pitch": name, "n": len(h),
                         "f0_hz_range": f"{h['speaker_f0_hz'].min():.0f}-{h['speaker_f0_hz'].max():.0f}",
                         "share_flat_unadjusted": float((h["pct_activation_raw"] <= LOW).mean()),
                         "share_flat": float((h["expressiveness_level"] == "low").mean()),
                         "share_lively": float((h["expressiveness_level"] == "high").mean())})
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    df["setting"] = df["dataset"].map(SETTING_OF_DATASET)
    rv_ref = (df["dataset"] == "recruitview") & (df["split"] == "train") & (~df["is_duplicate"].astype(bool))
    is_rv, is_mit = df["dataset"] == "recruitview", df["dataset"] == "mit"
    person = df["participant"].where(is_rv, df["person"])

    # recorded_answer: RecruitView answers vs RecruitView train (as in Phase 1)
    rv_out, rv_cuts = score(df, is_rv, rv_ref)
    # live_interview: each MIT student vs the other 68 students (cuts for new recordings: all 69)
    mit_out = score_leave_one_person_out(df, is_mit, is_mit, person)
    _, live_cuts = score(df, is_mit, is_mit)
    # MIT scored against the recorded-answer reference: kept only to show the shift between settings
    mit_rvref, _ = score(df, is_mit, rv_ref)

    levels = pd.concat([rv_out, mit_out]).sort_index()
    df = df.join(levels)
    for dim in DIMENSIONS:
        lo_name, mid, hi_name = LEVEL_NAMES[dim]
        df[f"{dim}_wording"] = df[f"{dim}_level"].map({"low": lo_name, "normal": mid, "high": hi_name, "missing": ""})
        df[f"{dim}_withheld"] = is_rv & (~df["acoustic_ok"].astype(bool)) & (dim in GATED)
        df[f"{dim}_level_recorded_ref"] = mit_rvref[f"{dim}_level"].reindex(df.index)  # MIT rows only
        df[f"{dim}_index_recorded_ref"] = mit_rvref[f"{dim}_index"].reindex(df.index)

    cuts = {"settings": {
        "recorded_answer": {"reference": "RecruitView train participants, unique recordings", **rv_cuts},
        "live_interview": {"reference": "MIT interviews, all 69 students (for new recordings; MIT's own answers "
                                        "are scored leaving their student out)", **live_cuts}},
        "default_setting": "live_interview"}

    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(P.ANSWER_LEVELS, index=False)
    P.LEVEL_CUTS.write_text(json.dumps(cuts, indent=1), encoding="utf-8")
    fv = flat_by_voice_pitch(df)
    fv.to_csv(P.S8_TABLES / "flat_by_voice_pitch.csv", index=False)

    print(f"wrote {P.ANSWER_LEVELS}: {len(df)} answers; {P.LEVEL_CUTS}")
    for st, c in cuts["settings"].items():
        a = c["expressiveness_pitch_adjustment"]
        print(f"  {st}: pitch adjustment activation = {a['intercept']:.4f} + {a['slope_per_hz']:.6f} * F0_hz "
              f"(n = {a['n']})")
    for st, g in df.groupby("setting"):
        g = g.dropna(subset=["activation_adj", "speaker_f0_hz"])
        print(f"  {st}: rho(activation, speaker F0) = {spearmanr(g['activation'], g['speaker_f0_hz'])[0]:+.2f} "
              f"-> after adjustment {spearmanr(g['activation_adj'], g['speaker_f0_hz'])[0]:+.2f}")
    for dim in DIMENSIONS:
        print(dim, df.groupby("setting")[f"{dim}_level"].value_counts(normalize=True).unstack().round(2).to_string(),
              sep="\n")
    print("withheld (RecruitView, gate):", {d: int(df[f"{d}_withheld"].sum()) for d in GATED})
    print("flat / lively by voice pitch:", fv.round(2).to_string(index=False), sep="\n")


if __name__ == "__main__":
    main()
