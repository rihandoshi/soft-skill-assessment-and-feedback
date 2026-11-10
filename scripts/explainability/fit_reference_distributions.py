"""Stage 2: fit reference distributions on TRAIN participants (unique recordings) and apply them.

Fitting (train only, duplicates removed):
  - population quantile tables p0..p100 for every active and experimental descriptor
      answer level:   duration bucket x question type, fallback to bucket, then pooled, if a cell has < 50
      conciseness:    question type only (bucketing by duration would be circular)
      sentence level: question type (speech rate, articulation rate; pitch/trailing/rise pooled)
      pause level:    pooled distribution of individual mid-sentence pause durations
  - flag thresholds that are quantiles (e.g. hesitation = p90 of the longest mid-sentence pause) are read from
    these train tables at application time, so they are train-derived by construction.
Everything needed at inference is saved to results/explainability/stage2_reference/percentile_tables.json.

Applying (all splits; uses only the JSON + each clip's own speaker's other recordings):
  comparisons.csv          answer-level pop_pct and self_delta for every referenced descriptor
  sentence_comparisons.csv sentence-level pop_pct
  flags.csv                every flag that fired, with status (active / experimental / disabled)
  category_scores.csv      0-100 signal and category scores (active signals only)
  evidence_examples/*.json v0 evidence for 10 validation answers
  stage2_report.md         reference summary, fallbacks, flag firing rates vs the 5-15% target, decisions

Run:  python scripts/explainability/fit_reference_distributions.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_evidence as be  # noqa: E402
import delivery_reference as dr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import paths as P  # noqa: E402  (results layout)
import descriptor_tables as DT  # noqa: E402  (descriptor tables + Stage 5 columns)
DESC = P.S2
REGISTRY = DT.load_registry()
MIN_CELL = 50
VERSION = "stage2-v0"

CONCISENESS = {"audio_duration_s", "speech_span_s", "speech_time_s", "word_count", "n_sentences"}
RELATIVE_DELTA = {"speech_rate_wpm", "artic_rate_sps", "artic_rate_wpm", "sent_rate_mean_wpm", "sent_rate_max_wpm",
                  "audio_duration_s", "speech_span_s", "speech_time_s", "word_count"}
# Dashboard score shape per active answer-level signal. 'upper_band': 100 up to p75 of comparable answers,
# then linear to 0 at p100 (more is worse, but a typical answer is not penalised). Pace is one-sided for now:
# Stage 0 found no rater penalty for slower speech and Stage 1 verification found the slow tail of
# transcript-based rates is partly transcript omissions. Stage 3 replaces these provisional bands with
# EBM-derived ones.
SCORE_SHAPES = {
    "speech_rate_wpm": "upper_band", "artic_rate_sps": "upper_band", "sent_rate_cv": "upper_band",
    "rushed_opening_ratio": "upper_band",
    "speech_span_s": "upper_band", "word_count": "upper_band",
}
# Stage 3: mid-sentence pause signals are no longer scored. They have no within-participant relation with the
# ratings and the between-participant direction is the opposite of "fewer pauses = better".
# Where a signal may count as a STRENGTH (percentile range among comparable answers). Low pause/variability
# values are genuinely good (fluent, steady). Low speech/articulation rate is NOT a strength: the slow side is
# unvalidated and partly transcript omissions. Very short answers are not "concise", so conciseness needs >= p10.
STRENGTH_RANGES = {
    # Stage 3: a strength needs the same rater support as a flag. Only answer length has it (keep); rate
    # steadiness and an even opening were dropped (no within-participant relation).
    "speech_span_s": [10, 30], "word_count": [10, 30],
}
# Stage 3: the pausing category is removed from the dashboard (see SCORE_SHAPES); pause measurements are still
# normed and flagged as experimental for validation on MIT / annotation.
CATEGORIES = {
    "pace": ["speech_rate_wpm", "artic_rate_sps", "sent_rate_cv", "rushed_opening_ratio"],
    "conciseness": ["speech_span_s", "word_count"],
}
STAGE3_DIR = P.S3

# --------------------------------------------------------------------------------------------
# The seven audio soft-skill categories of EXPLAINABILITY_README §3, each classified per answer into a hard
# class (low / typical / high, with category-specific labels). Signals are (column, orientation toward the
# "high" pole, frame). Frame "population" = percentile among comparable train answers; "self_delta" =
# percentile of the change vs the speaker's usual among train answers (used where the raw value mostly
# identifies the voice). `status` / `use_for_feedback` follow Stages 0-3; experimental classes are produced
# for the dashboard and for validation (MIT, annotation) but are not used for advice yet.
# --------------------------------------------------------------------------------------------
PROFILE_CUTS = (15, 85)  # train percentiles of the category index that separate low / typical / high
CATEGORY_PROFILE = {
    "pace": {
        "title": "Pace & Rate Control", "status": "active", "use_for_feedback": True,
        "signals": [["speech_rate_wpm", 1, "population"], ["artic_rate_sps", 1, "population"]],
        "labels": {"low": "slow", "typical": "typical pace", "high": "fast"}, "attention_pole": "high",
        "note": ("The 'slow' class is provisional: the cleaned transcripts drop words, which makes some answers look "
                 "slower than they are (Stage 1 verification). Re-check after verbatim ASR (Stage 4)."
                 if P.DESCRIPTOR_SOURCE == "transcript" else
                 "Rates from verbatim ASR (Stage 4); slow answers are usually hesitant (pauses, filled pauses) rather "
                 "than slowly articulated. Feedback on the slow side waits for MIT / annotation (no rater support yet).")},
    "pausing_fluency": {
        "title": "Pausing & Fluency", "status": "experimental", "use_for_feedback": False,
        "signals": [["mid_pause_rate_per_min", 1, "population"], ["long_mid_pause_rate_per_min", 1, "population"],
                    ["mid_pause_max_s", 1, "population"]],
        "labels": {"low": "few mid-sentence pauses", "typical": "typical pausing", "high": "many mid-sentence pauses"},
        "attention_pole": None,
        "note": "Descriptive only: RecruitView raters reward more pausing between people and show no within-person "
                "penalty (Stage 3). Pauses classified by word boundaries (Stage 4); fillers are the text branch's. "
                "Validate against MIT 'Paused'."},
    "expressiveness": {
        "title": "Vocal Expressiveness", "status": "experimental", "use_for_feedback": False,
        "signals": [["pitch_std_st", 1, "population"], ["pitch_range_st", 1, "population"],
                    ["intensity_std_db", 1, "population"], ["monotony_frac", -1, "population"]],
        "labels": {"low": "monotone", "typical": "typical variation", "high": "highly varied"}, "attention_pole": "low",
        "note": "Pitch in semitones relative to the speaker; no RecruitView support yet (Stage 3). Validate against MIT "
                "'EngagingTone'/'Excited' (Stage 7)."},
    "projection": {
        "title": "Projection & Clarity", "status": "experimental", "use_for_feedback": False,
        "signals": [["end_drop_median_db", -1, "population"]],
        "labels": {"low": "steady sentence endings", "typical": "typical endings", "high": "trails off"},
        "attention_pole": "high",
        "note": "Trailing off at real sentence ends (Stage 4; weak within-person support on all three scores). ASR word "
                "confidence was tested as intelligibility and rejected (opposite direction). Quality gate: Stage 6."},
    "composure": ({
        "title": "Vocal Confidence & Composure", "status": "experimental", "use_for_feedback": False,
        "signals": [["ser_arousal_mean", 1, "self_delta"]],
        "labels": {"low": "calmer than usual", "typical": "usual composure",
                   "high": "more activated / tense than usual"},
        "attention_pole": "high", "min_signals": 1,
        "note": "Stage 5: vocal activation from a pretrained speech-emotion model (arousal; its dominance output is the "
                "same dimension here, rho 0.98), relative to the speaker's own other answers only (raw values are 87% "
                "speaker identity). Oriented by the RecruitView evidence: raters score CALMER speakers as more confident "
                "(between people rho -0.50; within a person weak, same direction). Describes how the voice sounded, never "
                "an inner state. Needs MIT 'Calm'/'NotStressed' (Stage 7) and a second SER model before feedback."}
        if P.S5_SEGMENTS.exists() else {
        "title": "Vocal Confidence & Composure", "status": "experimental", "use_for_feedback": False,
        "signals": [["f0_median_st", 1, "population"], ["jitter_local", 1, "self_delta"],
                    ["shimmer_local", 1, "self_delta"], ["artic_rate_sps", 1, "self_delta"]],
        "labels": {"low": "more relaxed than usual", "typical": "usual composure", "high": "more tense than usual"},
        "attention_pole": "high", "min_signals": 2,
        "note": "Interim stress proxy (README §3.5): raised pitch, jitter, shimmer and speed, all relative to the "
                "speaker's own other answers (raw values identify the voice). Replaced by speech-emotion dominance / "
                "arousal in Stage 5; validate against MIT 'Calm'/'NotStressed'."}),
    "assertiveness": {
        "title": "Assertive Delivery", "status": "experimental", "use_for_feedback": False,
        "signals": [["terminal_slope_median_st_s", 1, "population"]],
        "labels": {"low": "falling sentence endings", "typical": "typical endings", "high": "rising endings (uptalk)"},
        "attention_pole": "high",
        "note": "Only sentence-final pitch movement for now; hedging phrases come from the text branch."},
    "conciseness": {
        "title": "Conciseness", "status": "active", "use_for_feedback": True,
        "signals": [["speech_span_s", 1, "population"], ["word_count", 1, "population"]],
        "labels": {"low": "brief", "typical": "typical length", "high": "long"}, "attention_pole": "high",
        "note": "Relative to answers to the same question type. The only category whose relation with the ratings holds "
                "within the same person (Stages 0 and 3). 'Brief' is not flagged: very short answers are judged on "
                "content (text branch)."},
}
SELF_DELTA_SIGNALS = sorted({s[0] for spec in CATEGORY_PROFILE.values() for s in spec["signals"] if s[2] == "self_delta"})
# Not scored or flagged although measured and normed: pause_ratio / between-sentence pauses (Stage 0: more
# silence goes with HIGHER scores, pausing is not a fault), rate drift (direction of "bad" unknown).
FLAG_RULES = {
    "too_fast": {"category": "pace", "status": "active", "pop_pct_min": 95, "self_delta_min": 0.25,
                 "pop_pct_min_no_baseline": 97,
                 "rule": "a sentence (>= 4 s, >= 5 words) at or above p95 of comparable train sentences AND >= 25% faster "
                         "than the speaker's usual sentence rate (p97 alone if no speaker baseline). Calibrated on train: "
                         "the guide's p90 / +20% fired on 21% of answers",
                 "rater_support": "weak: faster speech goes with lower confidence scores within a participant "
                                  "(rho ~ -0.06) and lower scores between participants; the flag itself is not "
                                  "significant within participants (Stage 3). Describe, do not attribute the score to it."},
    "too_slow": {"category": "pace", "status": "disabled" if P.DESCRIPTOR_SOURCE == "transcript" else "experimental",
                 "pop_pct_max": 10, "self_delta_max": -0.20,
                 "rule": "mirror of too_fast: a sentence at or below p10 of comparable train sentences AND >= 20% slower than "
                         "the speaker's usual. Was disabled on transcript words (Stage 1: 5/25 slowest sentences were "
                         "transcript omissions); with verbatim ASR (Stage 4) the slowest sentences are real slow, hesitant "
                         "speech (second ASR model agrees, low articulation rate), so it is back as EXPERIMENTAL until MIT / "
                         "annotation show listeners hear it.",
                 "rater_support": "none (Stage 3: no within-participant relation; EBM shows no slow-side penalty)"},
    "rushed_opening": {"category": "pace", "status": "experimental", "ratio_min": 1.20, "pop_pct_min": 90,
                       "rule": "first two sentences >= 20% faster than the rest AND the ratio above p90 of comparable answers "
                               "(openings are faster than the rest in most answers: train median ratio ~1.14). "
                               "EXPERIMENTAL since Stage 3: no relation with any rater score within participants",
                       "rater_support": "none (Stage 3 decision: drop for RecruitView; re-test on MIT / annotation)"},
    "hesitation": {"category": "pausing", "status": "experimental", "min_s": 1.0, "pop_level": 90, "speaker_ratio_min": 1.0,
                   "rule": "longest mid-sentence pause >= max(1.0 s, p90 of comparable answers' longest mid-sentence pause) "
                           "AND not shorter than the speaker's usual longest mid-sentence pause. EXPERIMENTAL since Stage 3: "
                           "no within-participant relation, and between participants raters favour MORE pausing",
                   "rater_support": "none / contradicted between people (Stage 3); validate perceptually on MIT 'Paused' and "
                                    "the in-house annotation before any advice"},
    "long_answer": {"category": "conciseness", "status": "active", "ratio_to_qtype_median_min": 2.0,
                    "rule": "speech span >= 2 x the train median for the same question type. Calibrated on train: the "
                            "guide's 1.75 x fired on 19% of answers",
                    "rater_support": "weak but consistent: the same person's flagged answers get lower confidence and "
                                     "speaking-skills scores (within rho ~ -0.06); answer length is the only category whose "
                                     "within-participant relation survives in RecruitView (Stage 0 and 3)"},
    "slow_start": {"category": "pausing", "status": "experimental", "min_s": 1.0, "pop_pct_min": 90,
                   "rule": "silence before first speech above p90. EXPERIMENTAL: in RecruitView the clip start is not the "
                           "end of the question, so this is not a true response latency"},
    "monotone": {"category": "expressiveness", "status": "experimental", "pitch_pct_max": 15, "energy_pct_max": 30,
                 "rule": "pitch variability < p15 AND energy variability < p30 of comparable answers"},
    "trails_off": {"category": "projection", "status": "experimental", "drop_vs_speaker_db": 6.0, "sentence_frac_min": 0.4,
                   "min_sentences": 2, "rule": ">= 40% of sentences end > 6 dB lower than the speaker's usual end drop"},
    "uptalk": {"category": "assertiveness", "status": "experimental", "rise_st_s_min": 10.0, "sentence_frac_min": 0.5,
               "min_sentences": 2, "rule": ">= 50% of sentences end with a rise >= 3 st over 300 ms (>= 10 st/s)"},
    # Stage 5 (pretrained speech-emotion model; README §3.5). All experimental: describe audible behaviour only.
    "low_dominance": {"category": "composure", "status": "experimental", "pop_pct_max": 15, "self_pct_max": 15,
                      "rule": "answer dominance below p15 of comparable train answers AND its difference from the speaker's "
                              "usual below p15 of train speaker differences (sounds less assured than other people AND "
                              "than this speaker usually does); timestamp = lowest-dominance stretch",
                      "rater_support": "contradicted: RecruitView raters score LOWER-dominance (calmer) speakers as more "
                                       "confident (between people rho -0.50); the flag has no within-person relation "
                                       "(Stage 3). Never advise 'sound more dominant'. Kept only for MIT validation."},
    "tension_rise": {"category": "composure", "status": "experimental", "min_run_s": 6.0,
                     "rule": ">= 6 s of consecutive windows with arousal AND pitch both > 1 SD above the speaker's usual "
                             "(more activated and higher-pitched than this speaker in their other answers). Cannot tell "
                             "tension from enthusiasm: the SER model's arousal and dominance correlate at 0.98 (Stage 5)",
                     "rater_support": "weak but consistent: the same person's flagged answers get lower scores on all three "
                                      "targets (rho -0.06 to -0.07, CIs exclude 0; Stage 3), and more activated speakers "
                                      "score lower between people. Describe, do not attribute the score to it."},
    "composure_drop": {"category": "composure", "status": "experimental", "min_drop_sd": 1.0, "pop_pct_min": 90,
                       "rule": "the smoothed dominance curve falls by > 1 speaker SD within the answer AND the fall is above "
                               "p90 of comparable answers (same length bucket: longer answers have more windows and larger "
                               "maximum falls; > 1 SD alone fired on 70%); timestamp = from the peak to the low point",
                       "rater_support": "none (Stage 3: no within-person relation); validate on MIT / annotation"},
}
TARGET_RATE = (5.0, 15.0)


# --------------------------------------------------------------------------------------------
def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dtypes = {"id": str, "participant": str, "duplicate_of": str}
    seg = DT.load_segments()
    sent = pd.read_csv(P.SENTENCES, dtype=dtypes)
    pauses = pd.read_csv(P.PAUSES, dtype=dtypes)
    seg["bucket"] = seg["audio_duration_s"].map(dr.duration_bucket)
    bucket = seg.set_index("id")["bucket"]
    qtype = seg.set_index("id")["question_type"]
    for frame in (sent, pauses):
        frame["bucket"] = frame["id"].map(bucket)
        frame["question_type"] = frame["id"].map(qtype)
        frame["is_duplicate"] = frame["id"].map(seg.set_index("id")["is_duplicate"])
    return seg, sent, pauses


def reference_specs() -> list[dict]:
    specs = []
    for key, meta in REGISTRY.items():
        if meta["status"] == "diagnostic" or meta["unit"] == "bool":
            continue
        level, col = meta["level"], meta["name"]
        if level == "sentence" and col not in {"speech_rate_wpm", "artic_rate_sps", "pitch_std_st", "end_drop_db",
                                               "terminal_slope_st_s"}:
            continue
        if level == "segment":
            cond = "qtype" if col in CONCISENESS else "bucket_x_qtype"
        else:
            cond = "qtype" if col in {"speech_rate_wpm", "artic_rate_sps"} else "pooled"
        specs.append({"key": key, "level": level, "column": col, "conditioning": cond, "status": meta["status"],
                      "category": meta["category"], "unit": meta["unit"],
                      "delta_kind": "relative" if col in RELATIVE_DELTA else "absolute",
                      "score_shape": SCORE_SHAPES.get(col, "none") if level == "segment" else "none",
                      "band": [25, 75],
                      "strength_range": STRENGTH_RANGES.get(col) if level == "segment" else None})
    specs.append({"key": "pause.mid_sentence_duration_s", "level": "pause", "column": "duration_s",
                  "conditioning": "pooled", "status": "active", "category": "pausing", "unit": "s",
                  "delta_kind": "absolute", "score_shape": "none", "band": [25, 75]})
    return specs


def fit(seg, sent, pauses) -> dict:
    train_seg = seg[(seg["split"] == "train") & ~seg["is_duplicate"]]
    train_sent = sent[(sent["split"] == "train") & ~sent["is_duplicate"]]
    train_mid = pauses[(pauses["split"] == "train") & ~pauses["is_duplicate"].astype(bool)
                       & (pauses["pause_class"] == "mid_sentence") & pauses["counted"]]
    refs = {}
    for spec in reference_specs():
        frame = {"segment": train_seg, "sentence": train_sent, "pause": train_mid}[spec["level"]]
        col = spec["column"]
        tables = {"*|*": dr.fit_quantiles(frame[col])}
        if spec["conditioning"] in ("qtype", "bucket_x_qtype"):
            for q, g in frame.groupby("question_type"):
                tables[f"*|{q}"] = dr.fit_quantiles(g[col])
        if spec["conditioning"] == "bucket_x_qtype":
            for b, g in frame.groupby("bucket"):
                tables[f"{b}|*"] = dr.fit_quantiles(g[col])
            for (b, q), g in frame.groupby(["bucket", "question_type"]):
                tables[f"{b}|{q}"] = dr.fit_quantiles(g[col])
        refs[spec["key"]] = {**{k: v for k, v in spec.items() if k != "key"}, "tables": tables}
    # Stage 3 results travel with the references (metadata only: bands are used for scoring only if identified).
    dec_path, band_path = P.S3_DECISIONS, P.S3_BANDS
    if dec_path.exists() and band_path.exists():
        decisions = json.loads(dec_path.read_text(encoding="utf-8"))
        bands = json.loads(band_path.read_text(encoding="utf-8"))["bands"]
        for key, spec in refs.items():
            if spec["level"] != "segment":
                continue
            col = spec["column"]
            info = {}
            if col in decisions:
                info["decision"] = decisions[col]["decision"]
                info["reason"] = decisions[col]["reason"]
            if col in bands:
                b = bands[col]
                info.update(ebm_shape=b["shape"], ebm_band_value=b["band_value"],
                            band_identified=b.get("band_identified", False))
                if b.get("band_identified"):
                    spec["optimal_band_value"] = b["band_value"]
            if info:
                spec["stage3"] = info
    # Speaker-relative references for the category profile: distribution of (value - speaker's usual) over
    # train answers. The speaker's usual is the leave-one-out median of their other recordings.
    for col in SELF_DELTA_SIGNALS:
        kind = "relative" if col in RELATIVE_DELTA else "absolute"
        base = dr.speaker_baseline(seg, col)
        deltas = pd.Series([dr.self_delta(v, b, kind) for v, b in zip(seg[col], base)], index=seg.index)
        train_mask = (seg["split"] == "train") & ~seg["is_duplicate"]
        refs[f"self_delta.{col}"] = {"level": "self_delta", "column": col, "conditioning": "pooled", "status": "experimental",
                                     "category": "composure", "unit": "relative change" if kind == "relative" else "change",
                                     "delta_kind": kind, "score_shape": "none", "band": [25, 75], "strength_range": None,
                                     "tables": {"*|*": dr.fit_quantiles(deltas[train_mask])}}
    return {
        "version": VERSION,
        "fitted_on": {"split": "train", "unique_recordings_only": True, "n_answers": int(len(train_seg)),
                      "n_participants": int(train_seg["participant"].nunique()), "n_sentences": int(len(train_sent)),
                      "n_mid_sentence_pauses": int(len(train_mid))},
        "levels": dr.LEVELS.astype(int).tolist(),
        "min_cell_size": MIN_CELL,
        "duration_buckets": {name: [lo, hi if np.isfinite(hi) else None] for name, lo, hi in dr.DURATION_BUCKETS},
        "references": refs,
        "flag_rules": FLAG_RULES,
        "categories": CATEGORIES,
        "notes": {"speaker_baseline": "leave-one-out median of the same speaker's other recordings (copies excluded), "
                                      ">= 2 others; for a new user, the other segments of the same interview",
                  "status": "only 'active' flags/signals reach the evidence JSON; experimental ones are for Stage 3 "
                            "validation; disabled ones are computed for monitoring only"},
    }


# --------------------------------------------------------------------------------------------
def apply(seg, sent, pauses, ref: dr.ReferenceSet) -> dict[str, pd.DataFrame]:
    tables = be.DeliveryTables(seg, sent, pauses, ref)
    comp_rows, flag_rows, score_rows = [], [], []
    seg_specs = {k: s for k, s in ref.refs.items() if s["level"] == "segment"}
    for cid in tables.seg.index:
        ctx = tables.context(cid)
        row = {"id": cid}
        for key, spec in seg_specs.items():
            v = ctx.seg.get(spec["column"])
            pct, cell, _ = ref.pct(key, v, ctx.bucket, ctx.qtype) if v is not None else (np.nan, "", 0)
            base = ctx.baseline.get(spec["column"], np.nan)
            row[f"{spec['column']}__pct"] = pct
            row[f"{spec['column']}__self_delta"] = dr.self_delta(v, base, spec["delta_kind"]) if v is not None else np.nan
            row[f"{spec['column']}__cell"] = cell
        comp_rows.append(row)
        for f in dr.all_flags(ctx, ref):
            flag_rows.append({"id": cid, "type": f["type"], "category": f["category"], "status": f["status"],
                              "start_s": f.get("start_s"), "end_s": f.get("end_s"),
                              "value": f["measurement"]["value"], "descriptor": f["measurement"]["descriptor"],
                              "pop_pct": f["comparison"].get("pop_pct"), "self_delta": f["comparison"].get("self_delta"),
                              "magnitude": f["magnitude"], "sentence": f.get("sentence")})
        sig, cat = dr.scores(ctx, ref)
        score_rows.append({"id": cid, **{f"score_{k}": v for k, v in sig.items()}, **{f"category_{k}": v for k, v in cat.items()}})
    keys = seg[["id", "split", "participant", "question_type", "bucket", "is_duplicate"]]
    comp = keys.merge(pd.DataFrame(comp_rows), on="id")
    flags = keys.merge(pd.DataFrame(flag_rows), on="id") if flag_rows else pd.DataFrame()
    scores = keys.merge(pd.DataFrame(score_rows), on="id")
    sent_comp = sent[["id", "sent_idx", "split", "question_type", "bucket", "speech_rate_wpm", "artic_rate_sps"]].copy()
    for col in ("speech_rate_wpm", "artic_rate_sps"):
        key = f"sentence.{col}"
        sent_comp[f"{col}__pct"] = [ref.pct(key, v, b, q)[0] for v, b, q in
                                    zip(sent_comp[col], sent_comp["bucket"], sent_comp["question_type"])]
    profile = keys.merge(tables.profile().reset_index(), on="id")
    return {"comparisons": comp, "flags": flags, "scores": scores, "sentence_comparisons": sent_comp, "tables": tables,
            "profile": profile}


def fit_category_profile(seg, sent, pauses, data: dict) -> dict:
    """Train-only cut points for the hard classes: the category index's p15 / p85 among unique train answers,
    and the same for its change vs the speaker's usual."""
    spec_only = {**data, "category_profile": {c: dict(s) for c, s in CATEGORY_PROFILE.items()}}
    tables = be.DeliveryTables(seg, sent, pauses, dr.ReferenceSet(spec_only))
    prof = tables.profile()
    train_ids = seg.loc[(seg["split"] == "train") & ~seg["is_duplicate"], "id"]
    tr = prof.loc[train_ids]
    lo, hi = PROFILE_CUTS
    out = {}
    for cat, spec in CATEGORY_PROFILE.items():
        idx, delta = tr[f"{cat}_index"].dropna(), tr[f"{cat}_delta"].dropna()
        out[cat] = {**spec,
                    "thresholds": {"low_max": float(np.percentile(idx, lo)), "high_min": float(np.percentile(idx, hi))},
                    "vs_usual_thresholds": {"lower_max": float(np.percentile(delta, lo)),
                                            "higher_min": float(np.percentile(delta, hi))},
                    "fitted_on": {"split": "train", "unique_recordings_only": True, "n_answers": int(len(idx)),
                                  "cuts_percentiles": list(PROFILE_CUTS)}}
    return out


def firing_rates(seg: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        clips = seg[(seg["split"] == split) & ~seg["is_duplicate"]]
        for name, rule in FLAG_RULES.items():
            fired = set(flags.loc[(flags["type"] == name), "id"]) & set(clips["id"]) if len(flags) else set()
            rows.append({"flag": name, "status": rule["status"], "category": rule["category"], "split": split,
                         "n_clips": len(clips), "rate_pct": 100 * len(fired) / len(clips)})
    return pd.DataFrame(rows).pivot_table(index=["flag", "status", "category"], columns="split", values="rate_pct").reset_index()


def write_report(ref: dr.ReferenceSet, rates: pd.DataFrame, out: dict, examples: list[str]) -> Path:
    L = []
    add = L.append
    fo = ref.data["fitted_on"]
    add("# Stage 2: Reference distributions, flags and scores\n")
    add(f"Descriptor tables: **{P.DESCRIPTOR_SOURCE}** (`{P.SEGMENTS.relative_to(P.ROOT).as_posix()}`).\n")
    add(f"Generated by `scripts/explainability/fit_reference_distributions.py` (`{VERSION}`). References fitted on "
        f"**train participants only, unique recordings only**: {fo['n_answers']} answers from {fo['n_participants']} "
        f"participants, {fo['n_sentences']} sentences, {fo['n_mid_sentence_pauses']} mid-sentence pauses. Val/test are only "
        "compared against them.\n")
    add("At inference, `delivery_reference.py` + `percentile_tables.json` are all that is needed: a new answer's "
        "measurements are compared with the stored quantiles, and with the median of the same speaker's other answers "
        "in that interview. No training data is loaded.\n")

    add("## Three layers\n")
    add("| Layer | Where | Example |")
    add("|---|---|---|")
    add("| Measurement | Stage 1 tables | sentence speech rate = 196 WPM |")
    add("| Comparison | Stage 2 (this) | p94 of comparable train sentences; +32% vs the speaker's usual 148 WPM |")
    add("| Feedback | Stage 9 (LLM, not yet) | \"Your pace was relatively fast here; slowing slightly may help clarity.\" |")
    add("")

    add("## Reference tables\n")
    add(f"Quantiles p0..p100 per cell. A cell needs >= {MIN_CELL} train recordings, otherwise the lookup falls back "
        "(bucket x question type -> bucket -> pooled; question type -> pooled).\n")
    add("| Reference | status | conditioning | cells with n >= 50 | smallest cell used | pooled median |")
    add("|---|---|---|---|---|---|")
    for key, spec in ref.refs.items():
        t = spec["tables"]
        used = {c: v["n"] for c, v in t.items() if v["n"] >= MIN_CELL}
        add(f"| `{key}` | {spec['status']} | {spec['conditioning']} | {len(used)}/{len(t)} | "
            f"{min(used.values()) if used else 0} | {t['*|*']['q'][50]:.3g} {spec['unit']} |")
    add("")
    small = [(c, v["n"]) for c, v in ref.refs["segment.speech_rate_wpm"]["tables"].items() if v["n"] < MIN_CELL]
    add(f"Cells below the minimum (answer level, fall back to their duration bucket): {small}.\n")

    add("## Do the train references transfer? (validation answers)\n")
    add("If the train-fitted distributions describe unseen participants, ~10% of validation answers should fall below "
        "train p10 and ~10% above p90 of their comparable cell.\n")
    add("| Descriptor | status | val below p10 (%) | val above p90 (%) |")
    add("|---|---|---|---|")
    comp = out["comparisons"]
    val = comp[(comp["split"] == "val") & ~comp["is_duplicate"]]
    for key, spec in ref.refs.items():
        if spec["level"] != "segment":
            continue
        p = val[f"{spec['column']}__pct"].dropna()
        if len(p):
            add(f"| `{spec['column']}` | {spec['status']} | {100 * (p < 10).mean():.1f} | {100 * (p > 90).mean():.1f} |")
    add("\nDescriptors with many tied values (e.g. pause rates that are often 0) cannot hit 10% exactly; the mid-rank "
        "percentile puts the whole tie at its centre.\n")

    add("## Flag firing rates (% of unique answers)\n")
    add(f"Target for active flags: {TARGET_RATE[0]:.0f}-{TARGET_RATE[1]:.0f}% of answers (0% = threshold wrong, ~40% = not "
        "informative). Thresholds start from the stage guide; `too_fast` and `long_answer` were recalibrated on the "
        "**train** firing rate only (grid: p90/p95/p97 x +20/25/30%, and 1.75/2/2.25/2.5 x median). Val/test are reported "
        "as a check and were not used for tuning.\n")
    add("| Flag | status | category | train | val | test | train in target? | rule |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in rates.iterrows():
        ok = "" if r.status != "active" else ("yes" if TARGET_RATE[0] <= r.train <= TARGET_RATE[1] else "**no**")
        add(f"| `{r.flag}` | {r.status} | {r.category} | {r.train:.1f} | {r.val:.1f} | {r.test:.1f} | {ok} | "
            f"{FLAG_RULES[r.flag]['rule']} |")
    add("")
    flags = out["flags"]
    uniq = out["scores"][~out["scores"]["is_duplicate"]]
    act = flags[flags["status"] == "active"] if len(flags) else flags
    per_clip = act.groupby("id").size().reindex(uniq["id"], fill_value=0)
    add(f"Active flags per answer (unique, all splits): {', '.join(f'{k}: {v}' for k, v in per_clip.value_counts().sort_index().items())} "
        f"(answers with 0 flags: {100 * (per_clip == 0).mean():.0f}%). The evidence JSON keeps at most 3, ranked by magnitude.\n")

    add("## Category profile: one hard class per audio soft-skill category\n")
    add("Every answer gets one class in each of the seven categories of EXPLAINABILITY_README §3. Category index = mean "
        "of its signals' percentiles (oriented toward the 'high' pole) among comparable train answers; classes cut at the "
        f"train p{PROFILE_CUTS[0]} / p{PROFILE_CUTS[1]} of the index (unique train recordings). `vs your usual` compares the "
        "index with the speaker's other answers (same cut points on the train distribution of that difference). No model "
        "is trained; only distributions and cut points are estimated. Only categories with `use_for_feedback` drive "
        "advice; the rest are reported (dashboard, temporal model, validation) until MIT / annotation support them.\n")
    prof = out["profile"]
    pu = prof[~prof["is_duplicate"]]
    add("| Category | status | for feedback | low / typical / high labels | train % low / typ / high | val % low / typ / high | val % lower / similar / higher than usual |")
    add("|---|---|---|---|---|---|---|")
    for cat, spec in ref.data["category_profile"].items():
        def dist(split, col, keys):
            v = pu.loc[pu["split"] == split, col].dropna()
            return " / ".join(f"{100 * (v == k).mean():.0f}" for k in keys) if len(v) else "n/a"
        labels = spec["labels"]
        add(f"| {spec['title']} | {spec['status']} | {'yes' if spec['use_for_feedback'] else 'no'} | "
            f"{labels['low']} / {labels['typical']} / {labels['high']} | {dist('train', f'{cat}_level', dr.PROFILE_LEVELS)} | "
            f"{dist('val', f'{cat}_level', dr.PROFILE_LEVELS)} | {dist('val', f'{cat}_vs_usual', ('lower', 'similar', 'higher'))} |")
    add("")
    cats = list(ref.data["category_profile"])
    corr = pu[pu["split"] == "train"][[f"{c}_index" for c in cats]].corr(method="spearman")
    add("**Are the categories distinct?** Spearman correlation between category indices (train). Values near 0 mean "
        "the categories describe different things; values near ±1 mean two categories are the same measurement.\n")
    add("| | " + " | ".join(cats) + " |")
    add("|---|" + "---|" * len(cats))
    for a in cats:
        add(f"| {a} | " + " | ".join(f"{corr.loc[f'{a}_index', f'{b}_index']:+.2f}" for b in cats) + " |")
    add("")
    flags = out["flags"]
    if len(flags):
        add("**Consistency with the flags** (share of flagged answers that are in the matching 'high' class):\n")
        for ftype, cat, level in (("too_fast", "pace", "high"), ("long_answer", "conciseness", "high"),
                                  ("hesitation", "pausing_fluency", "high"), ("monotone", "expressiveness", "low"),
                                  ("trails_off", "projection", "high"), ("uptalk", "assertiveness", "high")):
            ids = set(flags.loc[flags["type"] == ftype, "id"]) & set(pu["id"])
            if ids:
                share = (pu.set_index("id").loc[sorted(ids), f"{cat}_level"] == level).mean()
                add(f"- `{ftype}` ({len(ids)} answers): {100 * share:.0f}% in {cat} = {level}")
        add("\nFlags and classes are complementary: a class describes the whole answer against other people; a flag marks "
            "a specific moment that is unusual against other people **and** the speaker's own usual (with a timestamp "
            "and the quoted sentence for the feedback).\n")

    add("## Category scores (dashboard only)\n")
    add("Signal scores use provisional bands (100 up to p75 of comparable answers, linear to 0 at p100); Stage 3 "
        "replaces them with EBM-derived optimal bands. Category = equal-weight mean of its signals.\n")
    add("| Category | signals | train median | train p10 | train p90 |")
    add("|---|---|---|---|---|")
    tr = uniq[uniq["split"] == "train"]
    for name, cols in CATEGORIES.items():
        v = tr[f"category_{name}"].dropna()
        add(f"| {name} | {', '.join(cols)} | {v.median():.0f} | {v.quantile(.1):.0f} | {v.quantile(.9):.0f} |")
    add("")

    add("## Decisions carried over from Stages 0-1\n")
    add("- **Only active descriptors produce flags, scores and evidence.** Experimental ones (pitch, trailing off, "
        "uptalk, latency) get reference tables and flags in `flags.csv` for Stage 3 validation, but never reach the LLM.")
    add("- **`too_slow` is disabled** (no rater support in Stage 0; transcript omissions inflate the slow tail, Stage 1 "
        "verification). Pace scoring is one-sided for the same reason." if P.DESCRIPTOR_SOURCE == "transcript" else
        "- **`too_slow` is experimental** (Stage 4: with verbatim ASR the slow tail is real speech; still no rater "
        "support, so it is computed for validation but never reaches the LLM). Pace scoring stays one-sided.")
    add("- **Pause ratio and between-sentence pauses are normed but neither scored nor flagged**: in Stage 0, more silence "
        "went with *higher* scores, even within a participant.")
    add("- **Loudness and absolute pitch have no reference tables at all** (diagnostic: they identify the speaker/microphone).")
    add("- **Duplicates** are excluded from fitting, and from each other's speaker baseline.")
    add("- **Stage 3 (RecruitView validation)**: `hesitation` and `rushed_opening` moved to experimental (no within-"
        "participant relation; raters favour *more* pausing between people); the pausing category left the dashboard "
        "scores and strengths; `too_fast` and `long_answer` stay active with a `rater_support` note that travels into the "
        "evidence JSON so the feedback can be worded descriptively. EBM optimal bands are stored as metadata "
        "(`stage3` in each reference) and only replace the provisional percentile band when Stage 3 found them "
        "identified. See `results/explainability/stage3_recruitview_validation/report.md`.")
    add("")

    add("## Evidence JSON v0\n")
    add(f"`scripts/explainability/build_evidence.py` produces it for any clip. Examples for 10 validation answers: "
        f"`evidence_examples/` ({', '.join(examples)}). Read them next to the audio (see the listening check).\n")
    path = P.S2_REPORT
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    seg, sent, pauses = load_tables()
    print("Fitting reference distributions on train (unique recordings)...")
    data = fit(seg, sent, pauses)
    print("Fitting category-profile thresholds on train (unique recordings)...")
    data["category_profile"] = fit_category_profile(seg, sent, pauses, data)
    P.S2_TABLES.mkdir(parents=True, exist_ok=True)
    P.PERCENTILE_TABLES.write_text(json.dumps(data, indent=1), encoding="utf-8")
    ref = dr.ReferenceSet(data)
    print("Applying to all splits...")
    out = apply(seg, sent, pauses, ref)
    out["comparisons"].to_csv(P.S2_TABLES / "comparisons.csv", index=False)
    out["sentence_comparisons"].to_csv(P.S2_TABLES / "sentence_comparisons.csv", index=False)
    out["flags"].to_csv(P.S2_TABLES / "flags.csv", index=False)
    out["scores"].to_csv(P.S2_TABLES / "category_scores.csv", index=False)
    out["profile"].to_csv(P.S2_TABLES / "category_profile.csv", index=False)
    rates = firing_rates(seg, out["flags"])
    print(rates.round(1).to_string(index=False))
    ex_dir = P.EVIDENCE_EXAMPLES
    ex_dir.mkdir(exist_ok=True)
    for old in ex_dir.glob("*.json"):
        old.unlink()
    questions = be.question_lookup()
    pool = out["tables"].seg
    examples = pool[(pool["split"] == "val") & ~pool["is_duplicate"]].sample(10, random_state=0)["id"].tolist()
    for cid in examples:
        ev = be.build_evidence(cid, out["tables"], ref, questions.get(cid))
        (ex_dir / f"{cid}.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(ref, rates, out, examples)
    print(f"Done. Report: {report}")


if __name__ == "__main__":
    main()
