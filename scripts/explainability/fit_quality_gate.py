"""Stage 6, step 2: is loudness a recording artefact, and where should the quality gate sit?

Inputs   stage6_quality/quality_scores.csv (run_quality_scores.py), descriptor tables + Stage 2 flags + targets
         (via stage3_validate_recruitview.load_all), VAD RMS from the vad_pause features
Outputs  stage6_quality/quality_gate.json   gate rule read by build_evidence.py (fitted on TRAIN unique recordings)
         stage6_quality/report.md, tables/quality_rater_corr.csv, tables/flag_rate_by_quality.csv

Gate (two levels):
  basic      the answer gets no delivery feedback at all: no speech, transcript timing suspect, > 1% dropouts
             (the v0 checks)
  acoustic   energy / pitch / SER-based categories and flags are suppressed: quality composite below the train
             p10, or more than 0.1% of samples clipped. Timing-based categories (pace, pausing, conciseness) stay.
Quality composite = mean of the train percentiles of squim_pesq, squim_stoi and snr_vad_db (higher = better).

Run:  python scripts/explainability/fit_quality_gate.py [--n-boot 500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import stage3_validate_recruitview as S3  # noqa: E402
import stage0_confound_audit as s0  # noqa: E402
import extract_ser_descriptors as ES  # noqa: E402

COMPOSITE = ["squim_pesq", "squim_stoi", "snr_vad_db"]
QUALITY = ["squim_pesq", "squim_stoi", "squim_si_sdr", "snr_vad_db", "noise_floor_dbfs", "clip_frac"]
LOUDNESS = ["intensity_mean_db", "vad_mean_rms_db", "speech_level_dbfs"]
GATE_PCT = 10
CLIP_FRAC_MAX = 0.001   # fixed: train p99 is 0 (only 0.3% of recordings have any clipped sample)
ACOUSTIC_CATEGORIES = ["expressiveness", "projection", "composure", "assertiveness"]
ACOUSTIC_FLAGS = ["monotone", "trails_off", "uptalk", "low_dominance", "tension_rise", "composure_drop"]
# Timing flags that fire more often on poor audio (Stage 6 flag-rate table). Provisional: withheld below the gate
# but still computed (flags.csv) so Stage 7 / 8 can decide whether the extra firings are real or a VAD artefact.
PROVISIONAL_FLAGS = {"hesitation": "fires 1.7x more often below the gate; mid-pause detection moves with quality "
                                   "(within rho +0.20). Withheld on poor audio until Stage 7 / 8 decide."}
SHORT = s0.TARGET_SHORT
QS = np.linspace(0, 100, 101)


def pct_of(value: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Percentile (0-100) of each value against stored train quantiles (linear interpolation)."""
    return np.interp(value, quantiles, QS)


def composite(df: pd.DataFrame, ref: dict) -> pd.Series:
    parts = [pct_of(df[c].to_numpy(float), np.asarray(ref[c])) for c in COMPOSITE]
    arr = np.vstack(parts)
    return pd.Series(np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0), index=df.index)


def load() -> pd.DataFrame:
    df = S3.unique_recordings(S3.load_all())
    q = pd.read_csv(P.QUALITY_SCORES, dtype={"id": str})
    vad = pd.read_csv(P.VAD_FEATURES / "pause_features_all_splits.csv", dtype={"id": str})[["id", "mean_rms"]]
    vad["vad_mean_rms_db"] = 20 * np.log10(vad["mean_rms"].clip(lower=1e-10))
    return df.merge(q, on="id", how="left").merge(vad[["id", "vad_mean_rms_db"]], on="id", how="left")


def residualise_on_quality(df: pd.DataFrame, col: str) -> pd.Series:
    """Rank of `col` with the ranks of the quality measures regressed out (linear, pooled)."""
    cols = [col, *QUALITY]
    d = df[cols].rank()
    ok = d.notna().all(axis=1)
    Xq = np.column_stack([np.ones(ok.sum()), d.loc[ok, QUALITY].to_numpy()])
    y = d.loc[ok, col].to_numpy()
    beta, *_ = np.linalg.lstsq(Xq, y, rcond=None)
    out = pd.Series(np.nan, index=df.index)
    out[ok] = y - Xq @ beta
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    P.S6_TABLES.mkdir(parents=True, exist_ok=True)
    df = load()
    train = df[df["split"] == "train"].reset_index(drop=True)
    L = []
    add = L.append

    # ---------------- gate fit (train, no labels) ----------------
    ref = {c: np.nanpercentile(train[c].to_numpy(float), QS).tolist() for c in COMPOSITE}
    df["quality_composite"] = composite(df, ref)
    train = df[df["split"] == "train"].reset_index(drop=True)
    cut = float(np.nanpercentile(train["quality_composite"], GATE_PCT))
    clip_max = CLIP_FRAC_MAX
    df["acoustic_ok"] = (df["quality_composite"] >= cut) & ~(df["clip_frac"] > clip_max)
    df.loc[df["quality_composite"].isna(), "acoustic_ok"] = False
    gate = {"version": "v1 (Stage 6)", "fitted_on": "train unique recordings",
            "composite_measures": COMPOSITE, "composite_train_quantiles": ref,
            "composite_min": round(cut, 2), "composite_min_rule": f"train p{GATE_PCT} of the composite",
            "clip_frac_max": clip_max, "clip_frac_rule": "fixed 0.1% of samples at |x| >= 0.99",
            "acoustic_categories": ACOUSTIC_CATEGORIES, "acoustic_flags": ACOUSTIC_FLAGS,
            "provisional_flags": PROVISIONAL_FLAGS,
            "basic_checks": ["has_speech", "transcript_timing_ok", "dropouts_below_1pct"]}
    P.QUALITY_GATE.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    df[["id", "split", "quality_composite", "acoustic_ok"]].to_csv(P.S6_TABLES / "quality_gate_per_clip.csv", index=False)

    add("# Stage 6: recording-quality gate\n")
    add("Generated by `scripts/explainability/fit_quality_gate.py` from `run_quality_scores.py`. Quality models: "
        "torchaudio `SQUIM_OBJECTIVE` (estimates STOI / PESQ / SI-SDR without a clean reference; trained on DNS 2020 "
        "simulated noise) and signal checks (VAD SNR, clipping). Unique recordings; gate and correlations fitted on train.\n")

    # ---------------- 1. what the measures look like ----------------
    add("## 1. Quality measures\n")
    add("| Measure | median | p10 | p90 | ICC (speaker share) | Spearman with `video_quality_low` |")
    add("|---|---|---|---|---|---|")
    for c in QUALITY:
        v = df[c]
        add(f"| `{c}` | {v.median():.3g} | {v.quantile(.1):.3g} | {v.quantile(.9):.3g} | {ES.icc1(df, c):.2f} | "
            f"{spearmanr(df[c], df['video_quality_low'], nan_policy='omit')[0]:+.2f} |")
    cm = df[QUALITY].corr("spearman").round(2)
    add("\nSpearman between the measures:\n")
    add("| | " + " | ".join(f"`{c}`" for c in QUALITY) + " |")
    add("|---|" + "---|" * len(QUALITY))
    for c in QUALITY:
        add(f"| `{c}` | " + " | ".join(f"{cm.loc[c, d]:+.2f}" for d in QUALITY) + " |")
    add(f"\nClipping: {(df['clip_frac'] > 0).mean():.1%} of recordings have any sample at |x| >= 0.99; "
        f"{(df['clip_frac'] > 0.001).mean():.1%} more than 0.1% of samples.\n")

    # ---------------- 2. loudness question ----------------
    add("## 2. Is the loudness-score relation a recording artefact?\n")
    add("Each loudness measure is tested as-is and after regressing out (on ranks) all quality measures "
        f"({', '.join(QUALITY)}). `between` = speaker means; `within` = same participant, controlling duration and "
        "question type. * = 95% bootstrap CI excludes 0. Quality measures themselves are listed too: if raters "
        "penalise bad audio, that is a rating confound to keep in mind, not a delivery cue.\n")
    feats = []
    for c in LOUDNESS:
        train[f"{c}__q_adj"] = residualise_on_quality(train, c)
        feats += [c, f"{c}__q_adj"]
    feats += QUALITY
    print(f"bootstrap ({args.n_boot}) on {len(train)} train recordings ...", flush=True)
    sub = train.dropna(subset=QUALITY + LOUDNESS).reset_index(drop=True)
    corr = s0.correlation_audit(sub, feats, args.n_boot, args.seed)
    corr.to_csv(P.S6_TABLES / "quality_rater_corr.csv", index=False)

    def cell(f, t, m):
        r = corr[(corr.feature == f) & (corr.target == t) & (corr.measure == m)].iloc[0]
        return f"{r.rho:+.2f}{'*' if (r.ci_low > 0 or r.ci_high < 0) else ''}"
    add(f"n = {len(sub)} train recordings with all measures.\n")
    add("| Measure | " + " | ".join(f"between {SHORT[t]}" for t in s0.TARGETS) + " | "
        + " | ".join(f"within {SHORT[t]}" for t in s0.TARGETS) + " |")
    add("|---|" + "---|" * (2 * len(s0.TARGETS)))
    for f in feats:
        label = f"`{f.replace('__q_adj', '')}` quality-adjusted" if f.endswith("__q_adj") else f"`{f}`"
        add(f"| {label} | " + " | ".join(cell(f, t, "between") for t in s0.TARGETS) + " | "
            + " | ".join(cell(f, t, "within_partial") for t in s0.TARGETS) + " |")
    add("")

    # ---------------- 3. does quality distort the descriptors / flags? ----------------
    add("## 3. Do the acoustic descriptors and flags depend on recording quality?\n")
    desc = [c for c in ["end_drop_median_db", "pitch_std_st", "intensity_std_db", "monotony_frac",
                        "terminal_slope_median_st_s", "ser_arousal_mean", "jitter_local", "shimmer_local",
                        "speech_rate_wpm", "mid_pause_rate_per_min"] if c in df]
    add("Within-participant Spearman with the quality composite (does the same person's worse recording change the "
        "descriptor?). Timing descriptors are listed as a comparison.\n")
    add("| Descriptor | within rho with quality composite | between rho |")
    add("|---|---|---|")
    multi = df.groupby("participant")["id"].transform("size") >= 2
    for c in desc:
        d = df[multi][["participant", c, "quality_composite"]].dropna()
        w = spearmanr(d[c] - d.groupby("participant")[c].transform("mean"),
                      d["quality_composite"] - d.groupby("participant")["quality_composite"].transform("mean"))[0]
        pm = df.groupby("participant")[[c, "quality_composite"]].mean().dropna()
        add(f"| `{c}` | {w:+.2f} | {spearmanr(pm[c], pm['quality_composite'])[0]:+.2f} |")
    flag_cols = [c for c in df if c.startswith("flag_")]
    low = df["quality_composite"] < cut
    rates = pd.DataFrame({"flag": [c[5:] for c in flag_cols],
                          "rate_low_quality": [df.loc[low, c].mean() for c in flag_cols],
                          "rate_rest": [df.loc[~low, c].mean() for c in flag_cols]})
    rates["ratio"] = rates["rate_low_quality"] / rates["rate_rest"].replace(0, np.nan)
    rates.to_csv(P.S6_TABLES / "flag_rate_by_quality.csv", index=False)
    add(f"\nFlag rates below vs above the gate (composite < train p{GATE_PCT}; all splits, no labels used):\n")
    add("| Flag | below gate | above gate | ratio | gated? |")
    add("|---|---|---|---|---|")
    for _, r in rates.iterrows():
        add(f"| `{r.flag}` | {r.rate_low_quality:.1%} | {r.rate_rest:.1%} | {r.ratio:.2f} | "
            f"{'yes' if r.flag in ACOUSTIC_FLAGS else 'yes (provisional)' if r.flag in PROVISIONAL_FLAGS else 'no'} |")
    add("")

    # ---------------- 4. the gate ----------------
    add("## 4. The gate\n")
    add(f"- **acoustic**: quality composite >= **{cut:.1f}** (train p{GATE_PCT}) and clipping <= {clip_max:.2e} "
        f"(fixed). Fails -> categories {', '.join(ACOUSTIC_CATEGORIES)} and flags "
        f"{', '.join(ACOUSTIC_FLAGS)} are withheld; pace / pausing / conciseness stay. Provisionally also withheld "
        f"(still computed, not removed): {', '.join(PROVISIONAL_FLAGS)}.")
    add("- **basic** (v0, unchanged): speech present, transcript timing OK, dropouts <= 1%. Fails -> no delivery feedback.")
    fail = df.groupby("split")["acoustic_ok"].apply(lambda s: 1 - s.mean())
    add("- Acoustic gate failure rate: " + ", ".join(f"{k} {v:.1%}" for k, v in fail.items()) + ".")
    vq = df.groupby("video_quality")["acoustic_ok"].apply(lambda s: 1 - s.mean())
    add("- By RecruitView `video_quality`: " + ", ".join(f"{k} {v:.1%}" for k, v in vq.items()) + ".")
    by_p = df.groupby("participant")["acoustic_ok"].mean()
    add(f"- Speakers with every answer gated: {(by_p == 0).sum()} of {len(by_p)}; with some but not all: "
        f"{((by_p > 0) & (by_p < 1)).sum()}.\n")
    add("Written to `quality_gate.json`; `build_evidence.py` applies it (`quality_gate` field, withheld flags listed).\n")
    P.S6_REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {P.S6_REPORT}, {P.QUALITY_GATE}")


if __name__ == "__main__":
    main()
