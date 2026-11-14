"""Stage 8, step 5: do the per-answer detectors match what MIT's human raters and transcripts say? (no new labels)

Unit: interview (MIT labels are per interview): detector values averaged over the interview's answers; label = mean
of the 9 Turkers. 138 interviews of 69 students; bootstrap CIs resample students. Partial = controlling answer
length and recording quality, as in Stage 7, plus the speaker's typical pitch (median F0) for the voice-based
expressiveness measures (Phase 1). Pre-registered expectations below;
a test counts as "supported" when both the raw and the partial CI exclude 0 in the expected direction.

Besides the labels, the hesitation detectors are checked against the Turker transcripts' filler rate
(`tr_fill`, fillers per 100 words, Stage 7 `filler_check.csv`), next to Stage 4 Whisper (`asr_fill`).

Output  stage8_detectors/report.md, tables/validation_mit.csv, tables/discriminant_mit.csv, tables/interview_detectors.csv
Run:    python scripts/explainability/validate_detectors_mit.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import mit_label_reliability as LR  # noqa: E402
from validate_categories_mit import boot_ci, corr_pair, verdict  # noqa: E402

MIN_RELIABILITY = 0.5
# measures that track the speaker's voice type: their tests also control the speaker's typical pitch (Phase 1)
VOICE_BASED = {"expressiveness_index", "share_flat", "activation", "activation_adj", "pitch_std_st"}

# (dimension, measure, target, expected sign). The hesitation index is audio only since Phase 1; the CrisperWhisper
# rows stay as information for the text branch (they are not part of any audio level).
TESTS = [
    ("hesitation", "hesitation_index", "NoFillers", -1),
    ("hesitation", "hesitation_index", "tr_fill", +1),
    ("hesitation", "hesitation_index", "Overall", -1),
    ("hesitation", "share_hesitant", "NoFillers", -1),
    ("hesitation: classifier interjection", "p_interjection", "tr_fill", +1),
    ("hesitation: classifier interjection", "p_interjection", "NoFillers", -1),
    ("hesitation: classifier repetition/block", "p_repetition_block", "NoFillers", -1),
    ("hesitation: long mid-sentence pauses", "long_mid_pause_rate_per_min", "Overall", -1),
    ("text branch: CrisperWhisper fillers", "fillers_per_100w", "tr_fill", +1),
    ("text branch: CrisperWhisper fillers", "fillers_per_100w", "NoFillers", -1),
    ("text branch: Stage 4 Whisper fillers", "asr_fill", "tr_fill", +1),
    ("text branch: CrisperWhisper restarts", "restarts_per_100w", "NoFillers", -1),
    ("confidence", "pred_confidence", "Overall", +1),
    ("confidence", "pred_confidence", "RecommendHiring", +1),
    ("confidence", "pred_confidence", "NotAwkward", +1),
    ("confidence", "pred_confidence", "Calm", +1),
    ("confidence", "share_unconfident", "Overall", -1),
    ("expressiveness", "expressiveness_index", "EngagingTone", +1),
    ("expressiveness", "expressiveness_index", "Excited", +1),
    ("expressiveness", "share_flat", "EngagingTone", -1),
    ("expressiveness: activation before pitch adjustment", "activation", "EngagingTone", +1),
    ("expressiveness: pitch variability (reason only)", "pitch_std_st", "EngagingTone", +1),
]
# Phase 1 GPU runs: tested only when their outputs exist (run_speech_flow.py; run_ser_windows.py --model odyssey
# --dataset mit). The speech-flow model joins the hesitation index only if its NoFillers test is supported; the
# Odyssey model must agree with audeering on MIT (answer-level rho reported in the report).
SPEECH_FLOW = P.S8_TABLES / "speech_flow_answers.csv"
ODYSSEY_MIT = P.S7_TABLES / "mit_windows_odyssey.csv"
OPTIONAL_TESTS = [
    ("hesitation: speech-flow second opinion", "sf_p_disfluent", "NoFillers", -1, SPEECH_FLOW),
    ("hesitation: speech-flow second opinion", "sf_p_disfluent", "tr_fill", +1, SPEECH_FLOW),
    ("hesitation: speech-flow second opinion", "sf_p_disfluent", "Overall", -1, SPEECH_FLOW),
    ("expressiveness: Odyssey activation", "activation_odyssey", "EngagingTone", +1, ODYSSEY_MIT),
    ("expressiveness: Odyssey activation", "activation_odyssey", "Excited", +1, ODYSSEY_MIT),
]
VOICE_BASED |= {"activation_odyssey"}
AGREEMENT: dict[str, tuple[int, float]] = {}
DISCRIMINANT = ["hesitation_index", "fillers_per_100w", "p_interjection", "pred_confidence", "expressiveness_index",
                "activation", "pitch_std_st", "speech_span_s"]


def interview_table() -> pd.DataFrame:
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str})
    lv = lv[lv["dataset"] == "mit"].copy()
    seg = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str})[
        ["id", "person", "session", "audio_duration_s", "quality_composite"]]
    lv = lv.merge(seg, on="id", how="left")
    extra = []
    if SPEECH_FLOW.exists() and "sf_p_disfluent" not in lv:
        lv = lv.merge(pd.read_csv(SPEECH_FLOW, dtype={"id": str})[["id", "sf_p_disfluent"]], on="id", how="left")
    if "sf_p_disfluent" in lv:
        extra.append("sf_p_disfluent")
    if ODYSSEY_MIT.exists():
        lv = lv.merge(answer_activation(ODYSSEY_MIT).rename("activation_odyssey"), left_on="id", right_index=True,
                      how="left")
        lv = lv.merge(answer_activation(P.S7_TABLES / "mit_windows_audeering.csv").rename("activation_audeering_w"),
                      left_on="id", right_index=True, how="left")
        extra += ["activation_odyssey", "activation_audeering_w"]
    lv["share_hesitant"] = (lv["hesitation_level"] == "high").astype(float)
    lv["share_unconfident"] = (lv["confidence_level"] == "low").astype(float)
    lv["share_flat"] = (lv["expressiveness_level"] == "low").astype(float)
    num = ["hesitation_index", "confidence_index", "expressiveness_index", "fillers_per_100w",
           "restarts_per_100w", "p_interjection", "p_repetition_block", "long_mid_pause_rate_per_min",
           "pred_confidence", "activation", "activation_adj", "pitch_std_st", "speaker_f0_hz",
           "speech_span_s", "audio_duration_s", "quality_composite", "share_hesitant",
           "share_unconfident", "share_flat", *extra]
    iv = lv.groupby("participant")[num].mean()
    if "activation_odyssey" in lv:
        a = lv[["activation_odyssey", "activation_audeering_w"]].dropna()
        AGREEMENT["odyssey"] = (len(a), float(a.corr("spearman").iloc[0, 1]))
    iv["n_answers"] = lv.groupby("participant").size()
    iv["person"] = lv.groupby("participant")["person"].first()
    iv["session"] = lv.groupby("participant")["session"].first()
    s = LR.load_scores()
    iv = iv.join(s[s["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean(), how="inner")
    iv = iv.join(pd.read_csv(P.S7_TABLES / "filler_check.csv").set_index("interview")[["tr_fill", "asr_fill"]])
    iv.index.name = "interview"
    return iv.reset_index()


def answer_activation(path: Path) -> pd.Series:
    """Per answer: mean over 4-s windows with >= 50% speech of (arousal + dominance) / 2."""
    w = pd.read_csv(path, dtype={"id": str})
    w = w[w["speech_frac"] >= 0.5]
    return ((w["arousal"] + w["dominance"]) / 2.0).groupby(w["id"]).mean()


def fmt(v: float) -> str:
    return "" if pd.isna(v) else f"{v:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    iv = interview_table()
    rel = pd.read_csv(P.S7_RELIABILITY).set_index("label")["icc2_k"]
    controls = ["audio_duration_s", "quality_composite"]

    def ctl_for(x: str) -> list[str]:
        if x == "speech_span_s":
            return ["quality_composite"]
        return controls + ["speaker_f0_hz"] if x in VOICE_BASED else controls

    tests = TESTS + [t[:4] for t in OPTIONAL_TESTS if t[1] in iv]
    rows = []
    for dim, x, y, sign in tests:
        ctl = ctl_for(x)
        r, rp = corr_pair(iv, x, y, ctl)
        lo, hi, plo, phi = boot_ci(iv, x, y, ctl, args.n_boot, rng)
        reliable = y in ("tr_fill",) or rel.get(y, 0) >= MIN_RELIABILITY
        rows.append({"dimension": dim, "measure": x, "target": y, "expected": "+" if sign > 0 else "-",
                     "rho": r, "ci_lo": lo, "ci_hi": hi, "partial": rp, "p_lo": plo, "p_hi": phi,
                     "target_icc2k": rel.get(y, np.nan), "verdict": verdict(sign, lo, hi, plo, phi, reliable)})
        print(f"{x:30s} vs {y:16s} {r:+.2f} [{lo:+.2f},{hi:+.2f}]  partial {rp:+.2f} [{plo:+.2f},{phi:+.2f}]  "
              f"{rows[-1]['verdict']}", flush=True)
    val = pd.DataFrame(rows)

    labels = [l for l in LR.LABELS if rel.get(l, 0) >= MIN_RELIABILITY] + ["tr_fill"]
    disc = pd.DataFrame({m: {l: corr_pair(iv, m, l, ctl_for(m))[1] for l in labels} for m in DISCRIMINANT}).T

    pp = []
    pre = iv[iv["session"] == "pre"].set_index("person")
    post = iv[iv["session"] == "post"].set_index("person")
    both = pre.index.intersection(post.index)
    for m in ["hesitation_index", "p_interjection", "p_repetition_block", "long_mid_pause_rate_per_min",
              "fillers_per_100w", "pred_confidence", "expressiveness_index", "tr_fill"]:
        d = (post.loc[both, m] - pre.loc[both, m]).dropna()
        pp.append({"measure": m, "n": len(d), "pre_median": pre.loc[both, m].median(),
                   "post_median": post.loc[both, m].median(), "median_change": d.median(),
                   "share_improved_down": (d < 0).mean(), "p_wilcoxon": wilcoxon(d).pvalue if len(d) > 5 else np.nan})
    pp = pd.DataFrame(pp)

    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    iv.to_csv(P.S8_TABLES / "interview_detectors.csv", index=False)
    val.to_csv(P.S8_TABLES / "validation_mit.csv", index=False)
    disc.to_csv(P.S8_TABLES / "discriminant_mit.csv")
    pp.to_csv(P.S8_TABLES / "pre_post_mit.csv", index=False)

    md = ["# Stage 8: per-answer detectors validated on MIT", "",
          "Generated by `validate_detectors_mit.py`. Interview level (n = %d interviews, %d students); 9-rater mean; "
          "bootstrap 95%% CIs over students (%d resamples); partial = controlling answer length and recording "
          "quality, plus the speaker's typical pitch for expressiveness measures. `tr_fill` = fillers per 100 words "
          "in the Turker transcripts; `asr_fill` = the same from Stage 4 Whisper. Phase 1 definitions: hesitation is "
          "audio only, expressiveness is pitch-adjusted vocal activation, length is no longer a level."
          % (len(iv), iv["person"].nunique(), args.n_boot), "",
          "## Pre-registered tests", "",
          "| dimension | measure | vs | expected | rho [95% CI] | partial [95% CI] | label ICC(2,k) | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for r in val.itertuples():
        md.append(f"| {r.dimension} | `{r.measure}` | {r.target} | {r.expected} | {r.rho:+.2f} [{r.ci_lo:+.2f}, "
                  f"{r.ci_hi:+.2f}] | {r.partial:+.2f} [{r.p_lo:+.2f}, {r.p_hi:+.2f}] | "
                  f"{'' if pd.isna(r.target_icc2k) else f'{r.target_icc2k:.2f}'} | **{r.verdict}** |")
    if "odyssey" in AGREEMENT:
        n, rho = AGREEMENT["odyssey"]
        md += ["", f"Odyssey vs audeering activation on MIT answers: Spearman {rho:+.2f} (n = {n}; on RecruitView "
                   "the two agreed at 0.90-0.92)."]
        print(f"Odyssey vs audeering activation, MIT answers: {rho:+.2f} (n = {n})")
    md += ["", "## Discriminant matrix (partial rho; reliable labels only)", "",
           "| measure | " + " | ".join(disc.columns) + " |", "|---|" + "---|" * len(disc.columns)]
    for m, r in disc.iterrows():
        md.append(f"| `{m}` | " + " | ".join(fmt(v) for v in r) + " |")
    md += ["", "## Pre vs post coaching (same students, Wilcoxon)", "",
           "| measure | n | pre median | post median | median change | share down | p |", "|---|---|---|---|---|---|---|"]
    for r in pp.itertuples():
        md.append(f"| `{r.measure}` | {r.n} | {r.pre_median:.3f} | {r.post_median:.3f} | {r.median_change:+.3f} | "
                  f"{r.share_improved_down:.0%} | {r.p_wilcoxon:.3f} |")
    P.S8_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {P.S8_REPORT}")


if __name__ == "__main__":
    main()
