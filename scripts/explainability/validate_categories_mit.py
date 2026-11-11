"""Stage 7, step 3: validate the audio categories against MIT's human labels.

Unit = interview (labels are per interview). Answer-level measures (mit_run_pipeline.py) are averaged over the
interview's answers; flags become the share of answers flagged. Label = mean of the 9 Turkers (the AGGR row agrees
at rho 0.89-0.99, mit_label_reliability.py). Nothing here is fitted: thresholds and references are RecruitView's.

Pre-registered tests (markdown/EXPLAINABILITY_STAGES.md, Stage 7): each measure has its own label(s) and an
expected sign ("0" = direction not assumed). Every correlation is reported
  rho          Spearman over interviews
  rho_partial  controlling the interview's mean answer length (longer answers = more of everything) and the MIT
               quality composite (robustness: MIT audio is far-field)
  CI           95% bootstrap, resampling STUDENTS (both of a student's interviews together), 2000 draws
  ceiling      sqrt(ICC(2,k)) of the label: the largest correlation a perfect measure could reach
Verdicts: supported = CI excludes 0 in the expected direction (raw and partial); contradicted = CI excludes 0 in the
opposite direction; no support = otherwise; unreliable label = ICC(2,k) < 0.5 (cannot validate either way).

Also: discriminant matrix (each category's key measure vs every label), pre / post coaching paired tests, and the
domain shift of MIT vs RecruitView train percentiles.

Outputs  stage7_mit_validation/report.md, tables/{interview_measures,validation,discriminant,pre_post}.csv
Run:     python scripts/explainability/validate_categories_mit.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import mit_label_reliability as LR  # noqa: E402

T = P.S7_TABLES
MIN_RELIABILITY = 0.5
DURATION_BASED = {"speech_span_s", "flag_long_answer", "word_count"}  # never control length for length measures (Stage 3)
MIN_ALIGNED = 0.8   # interviews where < 80% of ASR words align to the transcript may have truncated answers

# (category, measure, label, expected sign)
TESTS = [
    ("pace", "speech_rate_wpm", "SpeakingRate", 0), ("pace", "artic_rate_sps", "SpeakingRate", 0),
    ("pace", "pace_dev_abs", "SpeakingRate", -1), ("pace", "pace_dev_abs", "Overall", -1),
    ("pace", "flag_too_fast", "Overall", -1),
    ("pausing", "mid_pause_rate_per_min", "Paused", 0), ("pausing", "long_mid_pause_rate_per_min", "Paused", 0),
    ("pausing", "mid_pause_rate_per_min", "Overall", -1), ("pausing", "flag_hesitation", "Overall", -1),
    ("pausing", "filled_pause_rate_per_min", "NoFillers", -1),
    ("expressiveness", "pitch_std_st", "EngagingTone", 1), ("expressiveness", "pitch_std_st", "Excited", 1),
    ("expressiveness", "intensity_std_db", "EngagingTone", 1), ("expressiveness", "intensity_std_db", "Excited", 1),
    ("expressiveness", "monotony_frac", "EngagingTone", -1), ("expressiveness", "expressiveness_index", "EngagingTone", 1),
    ("expressiveness", "flag_monotone", "EngagingTone", -1),
    ("composure", "ser_arousal_mean", "Calm", -1), ("composure", "ser_arousal_mean", "NotStressed", -1),
    ("composure", "ser_dominance_mean", "Calm", 0), ("composure", "flag_tension_rise", "NotStressed", -1),
    ("composure", "flag_tension_rise", "Calm", -1),
    ("assertiveness", "terminal_slope_median_st_s", "Focused", -1),
    ("assertiveness", "terminal_slope_median_st_s", "NotAwkward", -1), ("assertiveness", "flag_uptalk", "NotAwkward", -1),
    ("projection", "end_drop_median_db", "Focused", -1), ("projection", "flag_trails_off", "NotAwkward", -1),
    ("conciseness", "speech_span_s", "StructuredAnswers", -1), ("conciseness", "speech_span_s", "Overall", -1),
    ("conciseness", "flag_long_answer", "StructuredAnswers", -1),
]
# one key measure per category for the discriminant matrix (sign-aligned so "more = better")
KEY = {"pace": ("pace_dev_abs", -1), "pausing": ("mid_pause_rate_per_min", -1), "fillers": ("filled_pause_rate_per_min", -1),
       "expressiveness": ("pitch_std_st", 1), "composure": ("ser_arousal_mean", -1),
       "assertiveness": ("terminal_slope_median_st_s", -1), "projection": ("end_drop_median_db", -1),
       "conciseness": ("speech_span_s", -1)}
OWN = {"pace": ["SpeakingRate"], "pausing": ["Paused"], "fillers": ["NoFillers"], "expressiveness": ["EngagingTone", "Excited"],
       "composure": ["Calm", "NotStressed"], "assertiveness": ["Focused", "NotAwkward"], "projection": ["Focused", "NotAwkward"],
       "conciseness": ["StructuredAnswers"]}
PRE_POST = ["filled_pause_rate_per_min", "speech_rate_wpm", "mid_pause_rate_per_min", "pitch_std_st", "intensity_std_db",
            "ser_arousal_mean", "speech_span_s", "terminal_slope_median_st_s"]


# ============================================================================================
def interview_table() -> pd.DataFrame:
    seg = pd.read_csv(T / "mit_segments.csv")
    prof = pd.read_csv(T / "mit_profile.csv")
    flags = pd.read_csv(T / "mit_flags.csv")
    seg = seg.merge(prof[["id", *[c for c in prof if c.endswith("_index")]]], on="id", how="left")
    for ftype in flags["type"].unique():
        seg[f"flag_{ftype}"] = seg["id"].isin(set(flags.loc[flags["type"] == ftype, "id"])).astype(float)
    # pace deviation from the RecruitView train median (U-shape test): distance in population percentile points
    seg["pace_pct"] = np.interp(seg["speech_rate_wpm"], *_pace_q())
    seg["pace_dev_abs"] = (seg["pace_pct"] - 50).abs()
    num = seg.select_dtypes("number").columns.difference(["is_duplicate", "duplicate_group_size"])
    iv = seg.groupby("participant")[list(num)].mean()
    iv["n_answers"] = seg.groupby("participant").size()
    iv["person"] = seg.groupby("participant")["person"].first()
    iv["session"] = seg.groupby("participant")["session"].first()
    s = LR.load_scores()
    labels = s[s["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean()
    iv = iv.join(labels, how="inner")
    al = pd.read_csv(T / "mit_alignment.csv").set_index("interview")
    iv["align_ok"] = al["asr_words_labelled"].reindex(iv.index) >= MIN_ALIGNED
    iv.index.name = "interview"
    return iv.reset_index()


def _rank_resid(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    Xz = np.column_stack([np.ones(len(y)), Z])
    beta, *_ = np.linalg.lstsq(Xz, y, rcond=None)
    return y - Xz @ beta


def corr_pair(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> tuple[float, float]:
    d = df[[x, y, *controls]].dropna()
    if len(d) < 10 or d[x].nunique() < 3:
        return np.nan, np.nan
    rx, ry = rankdata(d[x]), rankdata(d[y])
    Z = np.column_stack([rankdata(d[c]) for c in controls])
    rp = np.corrcoef(_rank_resid(rx, Z), _rank_resid(ry, Z))[0, 1]
    return float(np.corrcoef(rx, ry)[0, 1]), float(rp)


def boot_ci(df: pd.DataFrame, x: str, y: str, controls: list[str], n_boot: int, rng) -> tuple[float, ...]:
    persons = df["person"].unique()
    rows = {p: np.flatnonzero(df["person"].to_numpy() == p) for p in persons}
    raw, part = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows[p] for p in rng.choice(persons, len(persons))])
        r, rp = corr_pair(df.iloc[idx], x, y, controls)
        raw.append(r)
        part.append(rp)
    lo, hi = np.nanpercentile(raw, [2.5, 97.5])
    plo, phi = np.nanpercentile(part, [2.5, 97.5])
    return lo, hi, plo, phi


def verdict(sign, lo, hi, plo, phi, reliable) -> str:
    if not reliable:
        return "unreliable label"
    if sign == 0:
        return "relation (direction not assumed)" if (lo > 0 or hi < 0) else "no relation"
    good = (lo > 0 and plo > 0) if sign > 0 else (hi < 0 and phi < 0)
    bad = (hi < 0 or phi < 0) if sign > 0 else (lo > 0 or plo > 0)
    return "supported" if good else "contradicted" if bad else ("raw only" if ((lo > 0) if sign > 0 else (hi < 0)) else "no support")


# ============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    iv = interview_table()
    iv.to_csv(T / "interview_measures.csv", index=False)
    rel = pd.read_csv(P.S7_RELIABILITY).set_index("label")
    controls = ["audio_duration_s", "quality_composite"]
    L = []
    add = L.append

    # ---------------- pre-registered tests ----------------
    rows = []
    for cat, x, y, sign in TESTS:
        if x not in iv:
            rows.append({"category": cat, "measure": x, "label": y, "verdict": "measure missing"})
            continue
        ctl = ["quality_composite"] if x in DURATION_BASED else controls
        r, rp = corr_pair(iv, x, y, ctl)
        lo, hi, plo, phi = boot_ci(iv, x, y, ctl, args.n_boot, rng)
        reliable = rel.loc[y, "icc2_k"] >= MIN_RELIABILITY
        r_al, rp_al = corr_pair(iv[iv["align_ok"]], x, y, ctl)
        rows.append({"category": cat, "measure": x, "label": y, "expected": sign, "rho": r, "ci_low": lo, "ci_high": hi,
                     "rho_partial": rp, "partial_ci_low": plo, "partial_ci_high": phi, "rho_partial_aligned": rp_al,
                     "label_icc2k": rel.loc[y, "icc2_k"], "ceiling": np.sqrt(max(rel.loc[y, "icc2_k"], 0)),
                     "verdict": verdict(sign, lo, hi, plo, phi, reliable)})
    val = pd.DataFrame(rows)
    val.to_csv(T / "validation.csv", index=False)

    # ---------------- discriminant ----------------
    disc = pd.DataFrame({cat: {lab: s * corr_pair(iv, m, lab, ["quality_composite"] if m in DURATION_BASED else controls)[1]
                               for lab in LR.LABELS}
                         for cat, (m, s) in KEY.items() if m in iv}).T
    disc.to_csv(T / "discriminant.csv")

    # ---------------- pre / post ----------------
    pp = []
    piv = iv.pivot_table(index="person", columns="session", values=PRE_POST + ["Overall", "NoFillers", "EngagingTone", "Calm"])
    for c in PRE_POST + ["Overall", "NoFillers", "EngagingTone", "Calm"]:
        d = piv[c].dropna()
        if len(d) < 10:
            continue
        diff = d["post"] - d["pre"]
        pp.append({"measure": c, "n_pairs": len(d), "pre_median": d["pre"].median(), "post_median": d["post"].median(),
                   "median_change": diff.median(), "share_increased": (diff > 0).mean(), "wilcoxon_p": wilcoxon(diff)[1]})
    pp = pd.DataFrame(pp)
    pp.to_csv(T / "pre_post.csv", index=False)

    # ---------------- report ----------------
    seg = pd.read_csv(T / "mit_segments.csv")
    ans = pd.read_csv(T / "mit_answers.csv")
    al = pd.read_csv(T / "mit_alignment.csv")
    add("# Stage 7: MIT validation of the audio categories\n")
    add("Generated by `scripts/explainability/validate_categories_mit.py`. MIT Interview Dataset: 138 mock interviews, "
        "69 MIT undergraduates, each interviewed before (`P#`) and after (`PP#`) coaching, rated per interview by the "
        "same 9 Turkers. Nothing is fitted on MIT: pipeline, references, thresholds and gate are RecruitView's.\n")
    add("## 1. Data\n")
    ok = al[al["status"] == "ok"]
    add(f"- Interviews transcribed and aligned: {len(ok)} of {len(al)}; median share of ASR words given a speaker "
        f"{ok['asr_words_labelled'].median():.0%}, of transcript words matched {ok['transcript_tokens_matched'].median():.0%}.")
    add(f"- All five questions found in {(ok['questions_found'].astype(str).str.replace('.0', '', regex=False) == '12345').mean():.0%} "
        f"of interviews; answers cut: {len(ans)} (median {ans['audio_duration_s'].median():.0f} s); interviews with labels "
        f"and answers: {len(iv)}.")
    add(f"- **Recording quality:** MIT is a far-field room microphone. Median speech level {seg['speech_level_dbfs'].median():.0f} dBFS, "
        f"SNR {seg['snr_vad_db'].median():.0f} dB, SQUIM PESQ {seg['squim_pesq'].median():.2f}; the RecruitView acoustic gate "
        f"would fail **{1 - seg['acoustic_ok'].mean():.0%}** of MIT answers. The gate is a feedback rule, so it is not applied "
        "here: recording conditions are the same for every interview, and every test is repeated with the quality "
        "composite partialled out.\n")
    add("## 2. Label reliability (step 0)\n")
    add("Same 9 Turkers on every interview. ICC(2,k) = reliability of their mean (the label used here); "
        f"labels below {MIN_RELIABILITY} cannot validate anything.\n")
    add("| Label | ICC(2,1) | ICC(2,k) | split-half | ceiling for rho |")
    add("|---|---|---|---|---|")
    for lab, r in rel.iterrows():
        add(f"| {lab} | {r.icc2_1:.2f} | {r.icc2_k:.2f} | {r.split_half_sb:.2f} | {np.sqrt(max(r.icc2_k, 0)):.2f} |")
    add("\n**`SpeakingRate` (0.03) and `Paused` (0.26) are not reliable**: the raters do not agree on them, so Pace and "
        "Pausing cannot be validated against their own labels. `EngagingTone`, `Excited`, `NoFillers` are good.\n")
    add("## 3. Pre-registered tests\n")
    add(f"Interview level, n = {len(iv)} interviews of {iv['person'].nunique()} students. rho = Spearman; partial = "
        "controlling mean answer length and the quality composite (quality only for length measures, since controlling "
        "length would remove what they measure). CIs resample students.\n")
    add(f"`partial, aligned only` repeats the partial rho on the {int(iv['align_ok'].sum())} interviews where >= "
        f"{MIN_ALIGNED:.0%} of ASR words aligned to the transcript (check that alignment problems do not drive a result).\n")
    add("| Category | Measure | Label | expected | rho [CI] | partial [CI] | partial, aligned only | ceiling | verdict |")
    add("|---|---|---|---|---|---|---|---|---|")
    for _, r in val.iterrows():
        if r["verdict"] == "measure missing":
            add(f"| {r.category} | `{r.measure}` | {r.label} | | | | | | measure missing |")
            continue
        sgn = {1: "+", -1: "-", 0: "?"}[int(r.expected)]
        add(f"| {r.category} | `{r.measure}` | {r.label} | {sgn} | {r.rho:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}] | "
            f"{r.rho_partial:+.2f} [{r.partial_ci_low:+.2f}, {r.partial_ci_high:+.2f}] | {r.rho_partial_aligned:+.2f} | "
            f"{r.ceiling:.2f} | **{r.verdict}** |")
    add("\n## 4. Discriminant check\n")
    add("Partial rho (length + quality controlled) of each category's key measure, sign-aligned so positive = "
        "the expected 'better delivery' direction, with every label. A category is specific if its own label (bold) is "
        "among its strongest; a row that is high everywhere is halo (raters liking the person).\n")
    labs = list(rel.index)
    add("| Category (measure) | " + " | ".join(labs) + " |")
    add("|---|" + "---|" * len(labs))
    for cat in disc.index:
        cells = []
        for lab in labs:
            v = disc.loc[cat, lab]
            s = f"{v:+.2f}" if np.isfinite(v) else ""
            cells.append(f"**{s}**" if lab in OWN.get(cat, []) else s)
        add(f"| {cat} (`{KEY[cat][0]}`) | " + " | ".join(cells) + " |")
    add("\n## 5. Pre / post coaching\n")
    add("Paired over students (Wilcoxon signed-rank). Coaching should reduce fillers if the measure captures them.\n")
    add("| Measure | pairs | pre median | post median | median change | share increased | p |")
    add("|---|---|---|---|---|---|---|")
    for _, r in pp.iterrows():
        add(f"| `{r.measure}` | {r.n_pairs} | {r.pre_median:.3g} | {r.post_median:.3g} | {r.median_change:+.3g} | "
            f"{r.share_increased:.0%} | {r.wilcoxon_p:.3f} |")
    add("\n## 6. Domain shift (MIT vs RecruitView train references)\n")
    pace_pct = seg["pace_pct"] if "pace_pct" in seg else pd.Series(np.interp(seg["speech_rate_wpm"], *_pace_q()))
    add(f"- MIT median pace {seg['speech_rate_wpm'].median():.0f} wpm = RecruitView train percentile "
        f"{np.nanmedian(pace_pct):.0f} (50 = RecruitView median).")
    add("- Share of MIT answers flagged per type (RecruitView train rates in `stage2_reference/report.md`): "
        + ", ".join(f"`{c[5:]}` {iv[c].mean():.0%}" for c in sorted(c for c in iv if c.startswith("flag_"))) + ".\n")
    P.S7_REPORT.write_text("\n".join(L), encoding="utf-8")
    print(val[["category", "measure", "label", "rho", "rho_partial", "verdict"]].round(2).to_string(index=False))
    print(f"wrote {P.S7_REPORT}")


def _pace_q() -> tuple[np.ndarray, np.ndarray]:
    """RecruitView train quantiles of answer-level speech rate (pooled cell) -> (values, percentiles)."""
    q = np.asarray(json.loads(P.PERCENTILE_TABLES.read_text())["references"]["segment.speech_rate_wpm"]["tables"]["*|*"]["q"], float)
    return q, np.linspace(0, 100, len(q))


if __name__ == "__main__":
    main()
