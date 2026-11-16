"""Stage 11 (Phase 3, Part A), steps 3-6: confidence and clarity from the speaking-style tagger, tested exactly as
pre-registered in stage11_style_tagger/preregistration.md (written before the tagger was run).

Values per answer (tag means over 10-s windows, run_style_tagger.py):
  style_confidence  mean(authoritative, flowing, enunciated) - mean(hesitant, soft, hushed, stammering)
  style_clarity     mean(crisp, enunciated) - slurred
  tag_hesitant, tag_monotone

Step 4 (descriptive): Spearman with speaking time, recording quality, vocal activation, speaker pitch and the fluency
index; share of variance between speakers; median tag rates RecruitView vs MIT.
Tests (MIT, interview level, 2,000 bootstrap resamples over students; pass = raw and partial CI exclude 0 in the
expected direction):
  C1-C3  style_confidence vs Overall / RecommendHiring / NotAwkward, +, controls length + quality + activation
  C4     tag_hesitant vs NoFillers, -, controls length + quality
  E1     tag_monotone vs EngagingTone, -, controls length + quality + speaker F0
Halo: partial correlation of style_confidence with every reliable MIT label.
Clarity: accent check (median RecruitView answer as a percentile of MIT; < 25 = accent effect; repeated on the
best-quality third of each dataset), between-person Spearman with RecruitView speaking_skills (must be >= 0).
Exploratory (added after the pre-registration, reported as such): the same between-person check of
style_confidence against RecruitView confidence_score.

Output  stage11_style_tagger/report.md, tables/style_values.csv, tables/tracks.csv, tables/tag_rates.csv,
        tables/tests_mit.csv, tables/halo_mit.csv, tables/listening_style.csv
Run:    python scripts/explainability/analyze_style_tagger.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import fit_answer_levels as F  # noqa: E402
import mit_label_reliability as LR  # noqa: E402
import run_disfluency_classifier as D  # noqa: E402
from validate_categories_mit import boot_ci, corr_pair, verdict  # noqa: E402

ASSURED = ["authoritative", "flowing", "enunciated"]
UNSURE = ["hesitant", "soft", "hushed", "stammering"]
CLEAR, UNCLEAR = ["crisp", "enunciated"], ["slurred"]
VALUES = ["style_confidence", "style_clarity", "tag_hesitant", "tag_monotone"]
LEN_Q = ["audio_duration_s", "quality_composite"]
TESTS = [("C1", "style_confidence", "Overall", +1, LEN_Q + ["activation"]),
         ("C2", "style_confidence", "RecommendHiring", +1, LEN_Q + ["activation"]),
         ("C3", "style_confidence", "NotAwkward", +1, LEN_Q + ["activation"]),
         ("C4", "tag_hesitant", "NoFillers", -1, LEN_Q),
         ("E1", "tag_monotone", "EngagingTone", -1, LEN_Q + ["speaker_f0_hz"])]
TRACK = ["speech_span_s", "quality_composite", "activation", "speaker_f0_hz", "hesitation_index"]
MIN_RELIABILITY = 0.5


def tag(t: str) -> str:
    return f"tag_{t.replace('-', '_')}"


def load() -> pd.DataFrame:
    st = pd.read_csv(P.STYLE_ANSWERS, dtype={"id": str})
    st["style_confidence"] = st[[tag(t) for t in ASSURED]].mean(1) - st[[tag(t) for t in UNSURE]].mean(1)
    st["style_clarity"] = st[[tag(t) for t in CLEAR]].mean(1) - st[[tag(t) for t in UNCLEAR]].mean(1)
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str, "duplicate_of": str}, low_memory=False)[
        ["id", "dataset", "split", "participant", "person", "is_duplicate", "duplicate_of", "speech_span_s",
         "activation", "speaker_f0_hz", "hesitation_index"]]
    df = lv.merge(st.drop(columns=["dataset"]), on="id", how="left")
    df["person"] = df["participant"].where(df["dataset"] == "recruitview", df["person"])
    # recording quality and duration: RecruitView from Stages 6 / 4, MIT from Stage 7
    gate = pd.read_csv(P.S6_TABLES / "quality_gate_per_clip.csv", dtype={"id": str}).set_index("id")["quality_composite"]
    dur = pd.read_csv(P.S4_SEGMENTS, dtype={"id": str}, usecols=["id", "audio_duration_s"]).set_index("id")["audio_duration_s"]
    mit = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str}).set_index("id")
    rv = df["dataset"] == "recruitview"
    df["quality_composite"] = np.where(rv, df["id"].map(gate).fillna(df["duplicate_of"].map(gate)),
                                       df["id"].map(mit["quality_composite"]))
    df["audio_duration_s"] = np.where(rv, df["id"].map(dur), df["id"].map(mit["audio_duration_s"]))
    rvl = pd.concat([pd.read_csv(P.ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str},
                                 usecols=["id", "speaking_skills", "confidence_score"]) for s in ("train", "val", "test")])
    return df.merge(rvl, on="id", how="left")


def between_speaker_share(v: pd.Series, g: pd.Series) -> float:
    d = pd.DataFrame({"v": v, "g": g}).dropna()
    m = d.groupby("g")["v"].transform("mean")
    return float(((m - d["v"].mean()) ** 2).sum() / ((d["v"] - d["v"].mean()) ** 2).sum())


def tracks(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    uniq = df[~df["is_duplicate"].fillna(False).astype(bool)]
    for ds, g in uniq.groupby("dataset"):
        for v in VALUES:
            r = {"dataset": ds, "value": v, "n": int(g[v].notna().sum()),
                 "between_speaker_share": between_speaker_share(g[v], g["person"])}
            for t in TRACK:
                d = g[[v, t]].dropna()
                r[f"rho_{t}"] = spearmanr(d[v], d[t])[0] if len(d) > 10 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def tag_rates(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df[~df["is_duplicate"].fillna(False).astype(bool)]
    cols = [c for c in df.columns if c.startswith("tag_")] + ["style_confidence", "style_clarity"]
    t = uniq.groupby("dataset")[cols].median().T
    t["mit_pct_of_recruitview_median"] = [float(F.pct(uniq.loc[uniq["dataset"] == "mit", c].dropna().to_numpy(),
                                                      pd.Series([t.loc[c, "recruitview"]])).iloc[0]) for c in cols]
    return t.reset_index(names="measure")


def interview_table(df: pd.DataFrame) -> pd.DataFrame:
    m = df[df["dataset"] == "mit"]
    iv = m.groupby("participant")[VALUES + LEN_Q + ["activation", "speaker_f0_hz"]].mean()
    iv["person"] = m.groupby("participant")["person"].first()
    s = LR.load_scores()
    return iv.join(s[s["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean(), how="inner").reset_index()


def accent_check(df: pd.DataFrame) -> pd.DataFrame:
    uniq = df[~df["is_duplicate"].fillna(False).astype(bool)].dropna(subset=["style_clarity", "quality_composite"])
    rows = []
    for name, sub in (("all answers", uniq),
                      ("best-quality third of each dataset", uniq[uniq.groupby("dataset")["quality_composite"]
                                                                  .transform(lambda q: q >= q.quantile(2 / 3))])):
        rv, mit = sub[sub["dataset"] == "recruitview"], sub[sub["dataset"] == "mit"]
        for v in ("style_clarity", "tag_slurred", "style_confidence"):
            p = float(F.pct(mit[v].to_numpy(), pd.Series([rv[v].median()])).iloc[0])
            rows.append({"subset": name, "value": v, "n_recruitview": len(rv), "n_mit": len(mit),
                         "recruitview_median": rv[v].median(), "mit_median": mit[v].median(),
                         "recruitview_median_as_mit_percentile": p})
    return pd.DataFrame(rows)


def person_relation(df: pd.DataFrame, x: str, y: str, n_boot: int, rng) -> dict:
    rv = df[(df["dataset"] == "recruitview") & ~df["is_duplicate"].fillna(False).astype(bool)]
    pm = rv.groupby("person")[[x, y]].mean().dropna()
    r = spearmanr(pm[x], pm[y])[0]
    bs = [spearmanr(*pm.iloc[rng.integers(0, len(pm), len(pm))][[x, y]].to_numpy().T)[0] for _ in range(n_boot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"x": x, "y": y, "n_people": len(pm), "rho": r, "ci_lo": lo, "ci_hi": hi}


def listening(df: pd.DataFrame) -> pd.DataFrame:
    audio = D.all_clips().set_index("id")["audio_path"]
    uniq = df[~df["is_duplicate"].fillna(False).astype(bool)]
    rows = []
    for v in ("style_confidence", "style_clarity"):
        for end, asc in (("highest", False), ("lowest", True)):
            g = uniq.dropna(subset=[v]).sort_values(v, ascending=asc).drop_duplicates("person").head(10)
            for r in g.itertuples():
                rows.append({"value": v, "end": end, "dataset": r.dataset, "id": r.id,
                             "score": round(getattr(r, v), 3), "audio_path": audio.get(r.id, ""),
                             "heard_ok": "", "problem": ""})
    return pd.DataFrame(rows)


def f2(v) -> str:
    return "" if pd.isna(v) else f"{v:+.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    df = load()
    rel = pd.read_csv(P.S7_RELIABILITY).set_index("label")["icc2_k"]
    tr, rates, acc = tracks(df), tag_rates(df), accent_check(df)
    iv = interview_table(df)

    tests = []
    for tid, x, y, sign, ctl in TESTS:
        r, rp = corr_pair(iv, x, y, ctl)
        lo, hi, plo, phi = boot_ci(iv, x, y, ctl, args.n_boot, rng)
        tests.append({"test": tid, "measure": x, "label": y, "expected": "+" if sign > 0 else "-",
                      "controls": " + ".join(ctl), "rho": r, "ci_lo": lo, "ci_hi": hi, "partial": rp, "p_lo": plo,
                      "p_hi": phi, "label_icc2k": rel.get(y, np.nan),
                      "verdict": verdict(sign, lo, hi, plo, phi, rel.get(y, 0) >= MIN_RELIABILITY)})
        print(f"{tid} {x:17s} vs {y:16s} {r:+.2f} [{lo:+.2f},{hi:+.2f}] partial {rp:+.2f} [{plo:+.2f},{phi:+.2f}] "
              f"{tests[-1]['verdict']}", flush=True)
    tests = pd.DataFrame(tests)
    labels = [l for l in LR.LABELS if rel.get(l, 0) >= MIN_RELIABILITY]
    halo = pd.DataFrame([{"label": l, "icc2k": rel[l],
                          "partial_style_confidence": corr_pair(iv, "style_confidence", l, LEN_Q + ["activation"])[1],
                          "partial_style_clarity": corr_pair(iv, "style_clarity", l, LEN_Q)[1]} for l in labels])
    pr_clar = person_relation(df, "style_clarity", "speaking_skills", args.n_boot, rng)
    pr_conf = person_relation(df, "style_confidence", "confidence_score", args.n_boot, rng)
    lst = listening(df)

    conf_pass = bool((tests.loc[tests["test"].isin(["C1", "C2", "C3"]), "verdict"] == "supported").all())
    acc_main = acc[acc["value"] == "style_clarity"]
    accent = bool((acc_main["recruitview_median_as_mit_percentile"] < 25).any())
    clar_rv_ok = pr_clar["rho"] >= 0
    clar_state = "dropped (accent effect)" if accent else (
        "dropped (negative RecruitView relation)" if not clar_rv_ok else "candidate (listening check open)")

    P.S11_TABLES.mkdir(parents=True, exist_ok=True)
    df[["id", "dataset", "person", "n_windows", *VALUES]].to_csv(P.S11_TABLES / "style_values.csv", index=False)
    tr.to_csv(P.S11_TABLES / "tracks.csv", index=False)
    rates.to_csv(P.S11_TABLES / "tag_rates.csv", index=False)
    acc.to_csv(P.S11_TABLES / "accent_check.csv", index=False)
    tests.to_csv(P.S11_TABLES / "tests_mit.csv", index=False)
    halo.to_csv(P.S11_TABLES / "halo_mit.csv", index=False)
    pd.DataFrame([pr_clar, pr_conf]).to_csv(P.S11_TABLES / "recruitview_person_relations.csv", index=False)
    lst.to_csv(P.S11_TABLES / "listening_style.csv", index=False)

    md = ["# Stage 11: speaking-style tagger (Phase 3, Part A)", "",
          "Generated by `analyze_style_tagger.py`, following `preregistration.md`. Model "
          "`tiantiaf/wavlm-large-voice-quality`, 10-s windows (5-s hop, >= 50% speech), tag means per answer.", "",
          f"**Confidence: {'PASSED' if conf_pass else 'FAILED'}** (C1-C3 must all be supported).  ",
          f"**Clarity: {clar_state}.**", "",
          "## 1. What the values track (unique recordings, Spearman across answers)", "",
          "| dataset | value | n | between-speaker share | " + " | ".join(TRACK) + " |",
          "|---|---|---|---|" + "---|" * len(TRACK)]
    for r in tr.itertuples():
        md.append(f"| {r.dataset} | `{r.value}` | {r.n} | {r.between_speaker_share:.0%} | " +
                  " | ".join(f2(getattr(r, f'rho_{t}')) for t in TRACK) + " |")
    md += ["", "## 2. Pre-registered tests on MIT (interview level)", "",
           "| test | measure | vs | expected | controls | rho [95% CI] | partial [95% CI] | label ICC | verdict |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in tests.itertuples():
        md.append(f"| {r.test} | `{r.measure}` | {r.label} | {r.expected} | {r.controls} | {r.rho:+.2f} "
                  f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {r.partial:+.2f} [{r.p_lo:+.2f}, {r.p_hi:+.2f}] | "
                  f"{r.label_icc2k:.2f} | **{r.verdict}** |")
    md += ["", "## 3. Halo check (partial rho with every reliable MIT label)", "",
           "| label | ICC | `style_confidence` (length, quality, activation) | `style_clarity` (length, quality) |",
           "|---|---|---|---|"]
    md += [f"| {r.label} | {r.icc2k:.2f} | {f2(r.partial_style_confidence)} | {f2(r.partial_style_clarity)} |"
           for r in halo.itertuples()]
    md += ["", "## 4. Clarity", "", "Accent check: where the median RecruitView answer sits in the MIT distribution "
           "(percentile; < 25 for `style_clarity` = accent effect).", "",
           "| subset | value | RecruitView median | MIT median | RecruitView median as MIT percentile |",
           "|---|---|---|---|---|"]
    md += [f"| {r.subset} | `{r.value}` | {r.recruitview_median:.3f} | {r.mit_median:.3f} | "
           f"{r.recruitview_median_as_mit_percentile:.0f} |" for r in acc.itertuples()]
    md += ["", "RecruitView, between people (person means, unique recordings):", "",
           "| x | y | people | rho [95% CI] | note |", "|---|---|---|---|---|",
           f"| `style_clarity` | speaking_skills | {pr_clar['n_people']} | {pr_clar['rho']:+.2f} "
           f"[{pr_clar['ci_lo']:+.2f}, {pr_clar['ci_hi']:+.2f}] | pre-registered (must be >= 0) |",
           f"| `style_confidence` | confidence_score | {pr_conf['n_people']} | {pr_conf['rho']:+.2f} "
           f"[{pr_conf['ci_lo']:+.2f}, {pr_conf['ci_hi']:+.2f}] | exploratory |", "",
           "## 5. Tag rates (median per answer, unique recordings)", "",
           "`mit_pct` = the RecruitView median as a percentile of the MIT distribution (50 = same).", "",
           "| measure | RecruitView | MIT | mit_pct |", "|---|---|---|---|"]
    md += [f"| `{r.measure}` | {r.recruitview:.3f} | {r.mit:.3f} | {r.mit_pct_of_recruitview_median:.0f} |"
           for r in rates.itertuples()]
    md += ["", "## 6. Listening", "", "`tables/listening_style.csv`: the 10 highest and 10 lowest answers for each "
           "value (one per person), with audio paths and empty `heard_ok` / `problem` columns."]
    P.S11_REPORT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {P.S11_REPORT}; confidence {'PASSED' if conf_pass else 'FAILED'}; clarity {clar_state}")
    print(acc.round(3).to_string(index=False))
    print(pd.DataFrame([pr_clar, pr_conf]).round(3).to_string(index=False))
    print(tr.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
