"""Stage 10 (Phase 2): what changes when a live interview is compared with live interviews instead of with
self-recorded answers.

Reads stage8_detectors/tables/answer_levels.csv as written by fit_answer_levels.py (levels per setting: RecruitView
answers vs RecruitView train = `recorded_answer`; each MIT student vs the other 68 students = `live_interview`; plus,
for MIT only, the levels and indices it would get against the recorded-answer reference: `*_recorded_ref`).

1. Component shift. Each MIT answer's tie-aware percentile among the recorded-answer reference (as the levels
   compute it). Median percentile (50 = no shift) and the shares at <= 20 / >= 80, next to the same shares within
   the reference itself (ties make them differ from 20%). Criterion, fixed before looking: a component "keeps one
   reference" when the median percentile is within 40-60 AND both live tail shares are within 5 points of the
   reference's own; otherwise it "needs its own reference".
2. Level rates per setting (the Phase 2 target: low and high each 15-25% of answers, every dimension, every setting)
   and how MIT levels move from the recorded-answer reference to the live-interview one.
3. Does the change of reference change what the indices track? Interview-level Spearman with the pre-registered
   MIT labels for both versions of each index, partial = controlling answer length and recording quality (+ the
   speaker's typical pitch for expressiveness); bootstrap 95% CIs over students.
4. The voice-pitch slope of expressiveness per setting and the "flat" share by voice-pitch tercile.

Output  stage10_reference_settings/report.md, tables/component_shift.csv, tables/level_rates.csv,
        tables/level_moves.csv, tables/validation_by_reference.csv
Run:    python scripts/explainability/compare_settings.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import fit_answer_levels as F  # noqa: E402
import mit_label_reliability as LR  # noqa: E402
from validate_categories_mit import boot_ci, corr_pair  # noqa: E402

COMPONENTS = [  # (component, what it is, used in a level?)
    ("p_interjection", "filled-pause score (vocametrix)", True),
    ("p_repetition_block", "stumble score (vocametrix)", True),
    ("sf_p_disfluent", "speech-flow disfluent probability", True),
    ("long_mid_pause_rate_per_min", "long mid-sentence pauses per minute", True),
    ("activation", "vocal activation (before pitch adjustment)", True),
    ("pitch_std_st", "pitch variability (reason in words only)", False),
    ("fillers_per_100w", "CrisperWhisper fillers per 100 words (text branch)", False),
    ("speech_span_s", "speaking time (text branch / fusion)", False),
]
DIMS = ["hesitation", "expressiveness"]
TESTS = [("hesitation", "NoFillers", -1), ("hesitation", "Overall", -1),
         ("expressiveness", "EngagingTone", +1), ("expressiveness", "Excited", +1)]


def component_shift(df: pd.DataFrame) -> pd.DataFrame:
    ref = df[(df["dataset"] == "recruitview") & (df["split"] == "train") & ~df["is_duplicate"].astype(bool)]
    mit = df[df["dataset"] == "mit"]
    rows = []
    for comp, what, used in COMPONENTS:
        r, m = ref[comp].dropna().to_numpy(float), mit[comp].dropna()
        # tie-aware (mid-rank) percentiles, exactly as the levels use them; with many tied values (e.g. 80% of
        # recorded answers have no long mid-sentence pause) the reference's own tails are not 20%, so the live
        # tails are compared with the reference's own tails
        pm, pr = F.pct(r, m), F.pct(r, pd.Series(r))
        med_pct = float(pm.median())
        below, above = float((pm <= 20).mean()), float((pm >= 80).mean())
        rb, ra = float((pr <= 20).mean()), float((pr >= 80).mean())
        keeps = 40 <= med_pct <= 60 and abs(below - rb) <= 0.05 and abs(above - ra) <= 0.05
        rows.append({"component": comp, "what": what, "used_in_level": used,
                     "recorded_median": float(np.median(r)), "live_median": float(m.median()),
                     "live_median_pct_in_recorded": med_pct,
                     "recorded_share_low": rb, "live_share_low": below,
                     "recorded_share_high": ra, "live_share_high": above,
                     "verdict": "keeps one reference" if keeps else "needs its own reference"})
    return pd.DataFrame(rows)


def level_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates, moves = [], []
    mit = df[df["dataset"] == "mit"]
    for dim in DIMS + ["confidence"]:
        for st, g in df.groupby("setting"):
            vc = g[f"{dim}_level"].value_counts(normalize=True)
            rates.append({"dimension": dim, "reference": st, "applied_to": st, "low": vc.get("low", 0.0),
                          "normal": vc.get("normal", 0.0), "high": vc.get("high", 0.0)})
        vc = mit[f"{dim}_level_recorded_ref"].value_counts(normalize=True)
        rates.append({"dimension": dim, "reference": "recorded_answer", "applied_to": "live_interview",
                      "low": vc.get("low", 0.0), "normal": vc.get("normal", 0.0), "high": vc.get("high", 0.0)})
        if dim in DIMS:
            ct = pd.crosstab(mit[f"{dim}_level_recorded_ref"], mit[f"{dim}_level"])
            for a in ct.index:
                for b in ct.columns:
                    moves.append({"dimension": dim, "level_vs_recorded_reference": a,
                                  "level_vs_live_reference": b, "n_answers": int(ct.loc[a, b])})
    return pd.DataFrame(rates), pd.DataFrame(moves)


def interview_table(df: pd.DataFrame) -> pd.DataFrame:
    seg = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str})[
        ["id", "audio_duration_s", "quality_composite"]]
    lv = df[df["dataset"] == "mit"].merge(seg, on="id", how="left")
    cols = [f"{d}_index" for d in DIMS] + [f"{d}_index_recorded_ref" for d in DIMS] + \
           ["audio_duration_s", "quality_composite", "speaker_f0_hz"]
    iv = lv.groupby("participant")[cols].mean()
    iv["person"] = lv.groupby("participant")["person"].first()
    s = LR.load_scores()
    iv = iv.join(s[s["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean(), how="inner")
    return iv.reset_index()


def validation(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    iv = interview_table(df)
    rng = np.random.default_rng(seed)
    rows = []
    for dim, label, sign in TESTS:
        ctl = ["audio_duration_s", "quality_composite"] + (["speaker_f0_hz"] if dim == "expressiveness" else [])
        for ref, col in (("recorded_answer", f"{dim}_index_recorded_ref"), ("live_interview", f"{dim}_index")):
            r, rp = corr_pair(iv, col, label, ctl)
            lo, hi, plo, phi = boot_ci(iv, col, label, ctl, n_boot, rng)
            rows.append({"dimension": dim, "label": label, "expected": "+" if sign > 0 else "-", "reference": ref,
                         "rho": r, "ci_lo": lo, "ci_hi": hi, "partial": rp, "p_lo": plo, "p_hi": phi})
        print(f"  {dim:15s} vs {label:13s} " + "  ".join(
            f"{x['reference']}: {x['rho']:+.2f} (partial {x['partial']:+.2f})" for x in rows[-2:]), flush=True)
    return pd.DataFrame(rows)


def pc(v: float) -> str:
    return f"{v:.0%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str}, low_memory=False)
    cuts = json.loads(P.LEVEL_CUTS.read_text())
    shift = component_shift(df)
    rates, moves = level_tables(df)
    val = validation(df, args.n_boot, args.seed)
    fv = pd.read_csv(P.S8_TABLES / "flat_by_voice_pitch.csv")

    P.S10_TABLES.mkdir(parents=True, exist_ok=True)
    shift.to_csv(P.S10_TABLES / "component_shift.csv", index=False)
    rates.to_csv(P.S10_TABLES / "level_rates.csv", index=False)
    moves.to_csv(P.S10_TABLES / "level_moves.csv", index=False)
    val.to_csv(P.S10_TABLES / "validation_by_reference.csv", index=False)

    md = ["# Stage 10 (Phase 2): one reference per recording setting", "",
          "Generated by `compare_settings.py`. `recorded_answer` = RecruitView answers compared with RecruitView train "
          "answers (unique recordings). `live_interview` = each MIT student's answers compared with the other 68 "
          "students' answers (both of a student's interviews are left out together). New live recordings use all 69 "
          "students (`level_cuts.json`).", "",
          "## 1. Which components shift between settings", "",
          "Each live-interview answer's percentile among recorded answers (tie-aware, as in the levels). Median "
          "percentile: 50 = no shift. Tail shares: answers at or below the 20th / at or above the 80th percentile, "
          "next to the same shares within the recorded answers themselves (not always 20%: many recorded answers "
          "tie, e.g. 80% have no long mid-sentence pause). A component keeps one reference only if the median "
          "percentile is within 40-60 and both live tail shares are within 5 points of the recorded ones (fixed in "
          "advance).", "",
          "| component | used in a level | recorded median | live median | live median as recorded percentile | "
          "low tail: recorded / live | high tail: recorded / live | verdict |", "|---|---|---|---|---|---|---|---|"]
    for r in shift.itertuples():
        md.append(f"| {r.what} (`{r.component}`) | {'yes' if r.used_in_level else 'no'} | {r.recorded_median:.3g} | "
                  f"{r.live_median:.3g} | {r.live_median_pct_in_recorded:.0f} | {pc(r.recorded_share_low)} / "
                  f"{pc(r.live_share_low)} | {pc(r.recorded_share_high)} / {pc(r.live_share_high)} | {r.verdict} |")
    md += ["", "## 2. Level rates", "", "Target: `low` and `high` each 15-25% of answers in every setting.", "",
           "| dimension | reference | applied to | low | normal | high |", "|---|---|---|---|---|---|"]
    for r in rates.itertuples():
        md.append(f"| {r.dimension} | {r.reference} | {r.applied_to} | {pc(r.low)} | {pc(r.normal)} | {pc(r.high)} |")
    md += ["", "How MIT answers move (rows: level against the recorded-answer reference; columns: against the "
           "live-interview reference):", ""]
    for dim in DIMS:
        ct = moves[moves["dimension"] == dim].pivot(index="level_vs_recorded_reference",
                                                    columns="level_vs_live_reference", values="n_answers").fillna(0)
        md += [f"**{dim}**", "", "| vs recorded \\ vs live | " + " | ".join(ct.columns) + " |",
               "|---|" + "---|" * len(ct.columns)]
        md += [f"| {i} | " + " | ".join(str(int(v)) for v in row) + " |" for i, row in ct.iterrows()] + [""]
    md += ["## 3. Do the indices still track the MIT ratings?", "",
           "Interview level (138 interviews, 69 students), 9-rater mean. Partial = controlling answer length and "
           f"recording quality (+ speaker's typical pitch for expressiveness). {args.n_boot} bootstrap resamples over "
           "students.", "", "| dimension | label | expected | reference | rho [95% CI] | partial [95% CI] |",
           "|---|---|---|---|---|---|"]
    for r in val.itertuples():
        md.append(f"| {r.dimension} | {r.label} | {r.expected} | {r.reference} | {r.rho:+.2f} [{r.ci_lo:+.2f}, "
                  f"{r.ci_hi:+.2f}] | {r.partial:+.2f} [{r.p_lo:+.2f}, {r.p_hi:+.2f}] |")
    md += ["", "## 4. Voice-pitch adjustment per setting", "", "| setting | intercept | slope per Hz | n reference answers |",
           "|---|---|---|---|"]
    for st, c in cuts["settings"].items():
        a = c["expressiveness_pitch_adjustment"]
        md.append(f"| {st} | {a['intercept']:.4f} | {a['slope_per_hz']:.6f} | {a['n']} |")
    md += ["", "\"Flat\" share by voice-pitch tercile (quality-gated answers excluded):", "",
           "| setting | voice pitch | F0 range (Hz) | n | flat before adjustment | flat | lively |",
           "|---|---|---|---|---|---|---|"]
    for r in fv.itertuples():
        md.append(f"| {r.setting} | {r.voice_pitch} | {r.f0_hz_range} | {r.n} | {pc(r.share_flat_unadjusted)} | "
                  f"{pc(r.share_flat)} | {pc(r.share_lively)} |")
    P.S10_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {P.S10_REPORT}")
    print(shift[["component", "live_median_pct_in_recorded", "recorded_share_low", "live_share_low",
                 "recorded_share_high", "live_share_high", "verdict"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
