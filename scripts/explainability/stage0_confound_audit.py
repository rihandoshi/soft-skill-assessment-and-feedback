"""Stage 0: confound audit (see markdown/EXPLAINABILITY_STAGES.md).

Question answered: which existing acoustic measures relate to the RecruitView scores because of
*delivery* (the relation holds within a participant, across that participant's own answers), and
which only because of *who is speaking / which microphone* (the relation exists only between
participants)? The second kind must never become absolute advice ("speak more quietly").

Uses TRAIN participants only. Reads existing cached feature tables; computes nothing from audio.

For every feature x target it reports five Spearman-type correlations:
  raw             between clips (plain Spearman)
  partial         between clips, controlling for duration, question type and video_quality
                  (rank-transform, residualise feature and target on the controls with OLS, then
                  correlate the residuals)
  between         participant means of feature vs participant means of target
  within          subtract each participant's mean from feature and target, then Spearman
  within_partial  as `within`, but also residualised on (participant-demeaned) duration and
                  question type, so a feature that merely tracks answer length is not counted
                  as delivery
95% CIs come from a participant-cluster bootstrap (participants resampled with replacement).

Each feature is then classified as
  delivery           within-participant relation (within_partial, or within for duration
                     proxies) has a CI excluding 0 and |rho| >= --min-effect for some target
  weak delivery      within-participant CI excludes 0 but |rho| < --min-effect
  speaker/mic effect no within relation, but a raw / partial / between relation with
                     |rho| >= --min-effect and CI excluding 0
  no relation        none of the above
It also reports ICC(1): how much of the feature's variance is "who the speaker is".

Finally it fits an Explainable Boosting Machine (interpret) on
[duration, question_type, transcript WPM, pause ratio, max pause, mean RMS] -> each target and saves
the shape-function plots.

Outputs (results/explainability/):
  (results/explainability/stage0_confound_audit/)
  report.md                       human-readable report (notes.md = hand-written interpretation)
  tables/correlations.csv         long table: feature x target x measure (+ CI)
  tables/classification.csv       one row per feature
  tables/ebm_importances.csv
  plots/*.png                     EBM shape functions

Run:  python scripts/explainability/stage0_confound_audit.py [--bootstrap 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SPLIT_DIR = ROOT / "Datasets" / "Split Dataset"
AUDIO_META_DIR = ROOT / "Datasets" / "RecruitView Audio"
QUESTION_TYPES = ROOT / "Datasets" / "question_types.csv"
LEGACY_DIR = ROOT / "outputs" / "audio" / "features" / "legacy"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402  (results layout)
NEW_DIR = P.HANDCRAFTED_FEATURES
VAD_DIR = P.VAD_FEATURES
FEATURE_DIMS = P.HANDCRAFTED_DIMS
OUT_DIR = P.S0
TABLE_DIR = P.S0_TABLES
PLOT_DIR = P.S0_PLOTS

TARGETS = ["confidence_score", "speaking_skills", "overall_performance"]
TARGET_SHORT = {"confidence_score": "conf", "speaking_skills": "speak", "overall_performance": "overall"}
MEASURES = ["raw", "partial", "between", "within", "within_partial"]
DURATION = "duration_s"
QTYPE = "question_type"
QTYPE_ORDER = ["intro", "behavioural", "self_assessment", "opinion"]
EBM_FEATURES = {
    DURATION: "Answer duration (s)",
    QTYPE: "Question type",
    "new_transcript_speech_rate_wpm": "Transcript WPM",
    "vad_pause_ratio": "Pause ratio (VAD)",
    "vad_maximum_pause_duration_seconds": "Max pause (s, VAD)",
    "vad_mean_rms": "Mean RMS loudness",
}
DURATION_PROXY_RHO = 0.90  # |rho(feature, duration)| above this => the feature *is* duration
TIMESTAMP_RE = re.compile(r"\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
def _feature_list(name: str) -> list[str]:
    dims = pd.read_csv(FEATURE_DIMS)
    return json.loads(dims.loc[dims["feature_set"] == name, "feature_names"].iloc[0])


def load_split(split: str) -> tuple[pd.DataFrame, list[str]]:
    """One row per clip: ids, controls, targets and every cached acoustic feature."""
    meta = pd.read_csv(SPLIT_DIR / f"{split}.csv", dtype={"id": str})
    meta = meta[["id", "user_no", "question_id", "video_quality", "transcript", *TARGETS]]
    meta = meta.rename(columns={"user_no": "participant"})

    audio = pd.read_csv(AUDIO_META_DIR / f"{split}_audio_metadata.csv", dtype={"id": str},
                        usecols=["id", "duration_seconds"]).rename(columns={"duration_seconds": DURATION})

    legacy_cols = _feature_list("A_original")
    legacy = pd.read_csv(LEGACY_DIR / f"{split}_features.csv", dtype={"id": str})[["id", *legacy_cols]]
    new = pd.read_csv(NEW_DIR / f"{split}_new_features.csv", dtype={"id": str})
    vad = pd.read_csv(VAD_DIR / f"{split}_pause_features.csv", dtype={"id": str})
    vad = vad.drop(columns=["user_id", "question_id", "split", "duration_seconds"])
    vad = vad.rename(columns={c: f"vad_{c}" for c in vad.columns if c != "id"})

    qtypes = pd.read_csv(QUESTION_TYPES)[["question_id", QTYPE]]

    df = (meta.merge(audio, on="id", how="inner")
              .merge(legacy, on="id", how="inner")
              .merge(new, on="id", how="inner")
              .merge(vad, on="id", how="inner")
              .merge(qtypes, on="question_id", how="left"))
    if len(df) != len(meta):
        raise RuntimeError(f"{split}: {len(meta)} clips in split but {len(df)} after joining feature tables")
    if df[QTYPE].isna().any():
        raise RuntimeError("Some question_ids have no question type; re-run make_question_types.py")

    df["transcript_word_count"] = df["transcript"].fillna("").map(
        lambda t: len(WORD_RE.findall(TIMESTAMP_RE.sub(" ", t))))
    df["video_quality_low"] = (df["video_quality"].str.lower() == "low").astype(float)
    df = df.drop(columns=["transcript"])

    feature_cols = [DURATION, "transcript_word_count", *legacy_cols,
                    *[c for c in new.columns if c != "id"], *[c for c in vad.columns if c != "id"]]
    return df.reset_index(drop=True), feature_cols


# --------------------------------------------------------------------------------------------
# Correlation machinery (all measures reduce to a Pearson correlation of transformed vectors)
# --------------------------------------------------------------------------------------------
def rank(a: np.ndarray) -> np.ndarray:
    """Column-wise average ranks; NaN stays NaN."""
    return pd.DataFrame(a).rank(method="average").to_numpy(dtype=float, copy=True)


def demean(a: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(a)
    return (frame - frame.groupby(groups).transform("mean")).to_numpy(dtype=float, copy=True)


def residualise(a: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """OLS residuals of every column of `a` on `controls` (+ intercept), fitted on non-NaN rows."""
    design = np.column_stack([np.ones(len(controls)), controls])
    out = np.full_like(a, np.nan, dtype=float)
    for j in range(a.shape[1]):
        ok = ~np.isnan(a[:, j])
        beta, *_ = np.linalg.lstsq(design[ok], a[ok, j], rcond=None)
        out[ok, j] = a[ok, j] - design[ok] @ beta
    return out


def masked_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson of each column of x with y, ignoring NaN in x (y must be finite)."""
    mask = ~np.isnan(x)
    n = mask.sum(0)
    yy = np.where(mask, y[:, None], np.nan)
    dx = x - np.nanmean(x, axis=0)
    dy = yy - np.nanmean(yy, axis=0)
    dx, dy = np.where(mask, dx, 0.0), np.where(mask, dy, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (dx * dy).sum(0) / np.sqrt((dx ** 2).sum(0) * (dy ** 2).sum(0))
    r[n < 10] = np.nan
    return r


def icc1(values: np.ndarray, groups: np.ndarray) -> float:
    """ICC(1) for unbalanced groups: share of variance attributable to the participant."""
    frame = pd.DataFrame({"v": values, "g": groups}).dropna()
    sizes = frame.groupby("g")["v"].size()
    means = frame.groupby("g")["v"].mean()
    k, n_total = len(sizes), len(frame)
    grand = frame["v"].mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(((frame["v"] - frame["g"].map(means)) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (n_total - k)
    n0 = (n_total - float((sizes ** 2).sum()) / n_total) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


def control_matrix(df: pd.DataFrame) -> np.ndarray:
    """[rank(duration), question-type dummies, video_quality_low]."""
    dur = rank(df[[DURATION]].to_numpy())
    dummies = pd.get_dummies(pd.Categorical(df[QTYPE], categories=QTYPE_ORDER), drop_first=True).to_numpy(float)
    return np.column_stack([dur, dummies, df["video_quality_low"].to_numpy()])


def correlation_audit(df: pd.DataFrame, features: list[str], n_boot: int, seed: int) -> pd.DataFrame:
    groups = df["participant"].to_numpy()
    X = df[features].to_numpy(dtype=float)
    Y = df[TARGETS].to_numpy(dtype=float)
    controls = control_matrix(df)
    controls_within = demean(controls[:, :-1], groups)  # video quality is a participant-level constant

    # Clip-level transformed matrices, one per measure.
    transformed = {
        "raw": (rank(X), rank(Y)),
        "partial": (residualise(rank(X), controls), residualise(rank(Y), controls)),
        "within": (rank(demean(X, groups)), rank(demean(Y, groups))),
        "within_partial": (residualise(demean(rank(X), groups), controls_within),
                           residualise(demean(rank(Y), groups), controls_within)),
    }
    # Participants with a single clip carry no within-participant information.
    multi = df.groupby("participant")["id"].transform("size").to_numpy() >= 2
    for key in ("within", "within_partial"):
        tx, ty = transformed[key]
        tx[~multi], ty[~multi] = np.nan, np.nan

    # Participant-level matrices for the between measure.
    pmeans = df.groupby("participant")[features + TARGETS].mean()
    between = (rank(pmeans[features].to_numpy(float)), rank(pmeans[TARGETS].to_numpy(float)))
    participants = pmeans.index.to_numpy()

    def all_measures(clip_idx: np.ndarray | None, part_idx: np.ndarray | None) -> dict[str, np.ndarray]:
        """Correlations (features x targets) for every measure on a (re)sample."""
        out = {}
        for key, (tx, ty) in transformed.items():
            sx, sy = (tx, ty) if clip_idx is None else (tx[clip_idx], ty[clip_idx])
            rows = []
            for t in range(len(TARGETS)):
                ok = ~np.isnan(sy[:, t])
                rows.append(masked_pearson(sx[ok], sy[ok, t]))
            out[key] = np.column_stack(rows)
        bx, by = between if part_idx is None else (between[0][part_idx], between[1][part_idx])
        out["between"] = np.column_stack([masked_pearson(bx, by[:, t]) for t in range(len(TARGETS))])
        return out

    point = all_measures(None, None)

    rows_of = {p: np.flatnonzero(groups == p) for p in participants}
    rng = np.random.default_rng(seed)
    boot = {m: np.empty((n_boot, len(features), len(TARGETS))) for m in MEASURES}
    start = time.time()
    for b in range(n_boot):
        draw = rng.integers(0, len(participants), len(participants))
        clip_idx = np.concatenate([rows_of[participants[i]] for i in draw])
        res = all_measures(clip_idx, draw)
        for m in MEASURES:
            boot[m][b] = res[m]
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot} ({time.time() - start:.0f}s)", flush=True)

    duration_rho = masked_pearson(rank(X), rank(df[[DURATION]].to_numpy())[:, 0])
    records = []
    for j, feat in enumerate(features):
        for t, target in enumerate(TARGETS):
            for m in MEASURES:
                lo, hi = np.nanpercentile(boot[m][:, j, t], [2.5, 97.5])
                records.append({"feature": feat, "target": target, "measure": m,
                                "rho": point[m][j, t], "ci_low": lo, "ci_high": hi,
                                "rho_with_duration": duration_rho[j]})
    return pd.DataFrame(records)


def classify(corr: pd.DataFrame, df: pd.DataFrame, features: list[str], min_effect: float) -> pd.DataFrame:
    def significant(row) -> bool:
        return row.ci_low > 0 or row.ci_high < 0

    def holds(row) -> bool:
        return significant(row) and abs(row.rho) >= min_effect

    groups = df["participant"].to_numpy()
    out = []
    for feat in features:
        sub = corr[corr.feature == feat].set_index(["target", "measure"])
        dur_rho = float(sub["rho_with_duration"].iloc[0])
        proxy = abs(dur_rho) >= DURATION_PROXY_RHO
        within_key = "within" if proxy else "within_partial"
        per_target = {}
        for target in TARGETS:
            within_row = sub.loc[(target, within_key)]
            if holds(within_row):
                per_target[target] = "delivery"
            elif significant(within_row):
                # Within-participant effects are inherently attenuated (less variance, more noise),
                # so a reliable but small within effect is still delivery, just weak.
                per_target[target] = "weak delivery"
            elif any(holds(sub.loc[(target, m)]) for m in ("raw", "partial", "between")):
                per_target[target] = "speaker/mic"
            else:
                per_target[target] = "none"
        labels = set(per_target.values())
        overall = next((name for key, name in [("delivery", "delivery"), ("weak delivery", "weak delivery"),
                                               ("speaker/mic", "speaker/mic effect")] if key in labels),
                       "no relation")
        o = sub.xs("overall_performance")
        attenuation = o.loc["within", "rho"] / o.loc["raw", "rho"] if abs(o.loc["raw", "rho"]) >= 0.05 else np.nan
        out.append({
            "feature": feat,
            "class": overall,
            **{f"class_{TARGET_SHORT[t]}": per_target[t] for t in TARGETS},
            "duration_proxy": proxy,
            "rho_with_duration": dur_rho,
            "icc_speaker_share": icc1(rank(df[[feat]].to_numpy())[:, 0], groups),
            "within_over_raw_overall": attenuation,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# Side analyses: question type, video quality
# --------------------------------------------------------------------------------------------
def question_type_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    demeaned = df[TARGETS] - df.groupby("participant")[TARGETS].transform("mean")
    for qt in QTYPE_ORDER:
        mask = df[QTYPE] == qt
        row = {"question_type": qt, "n_clips": int(mask.sum()),
               "median_duration_s": float(df.loc[mask, DURATION].median())}
        for t in TARGETS:
            row[f"mean_{TARGET_SHORT[t]}"] = float(df.loc[mask, t].mean())
            row[f"within_mean_{TARGET_SHORT[t]}"] = float(demeaned.loc[mask, t].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    kw = {t: stats.kruskal(*[df.loc[df[QTYPE] == q, t] for q in QTYPE_ORDER]).pvalue for t in TARGETS}
    kw_within = {t: stats.kruskal(*[demeaned.loc[df[QTYPE] == q, t] for q in QTYPE_ORDER]).pvalue for t in TARGETS}
    table.attrs["kruskal_p"] = kw
    table.attrs["kruskal_p_within"] = kw_within
    return table


def video_quality_effects(df: pd.DataFrame) -> dict:
    res = {"n_low": int(df["video_quality_low"].sum()), "n_high": int((1 - df["video_quality_low"]).sum())}
    per_part = df.groupby("participant")["video_quality_low"].nunique()
    res["participants_with_mixed_quality"] = int((per_part > 1).sum())
    for col in [*TARGETS, "vad_mean_rms", DURATION]:
        low, high = df.loc[df.video_quality_low == 1, col], df.loc[df.video_quality_low == 0, col]
        res[col] = {"median_low": float(low.median()), "median_high": float(high.median()),
                    "mannwhitney_p": float(stats.mannwhitneyu(low, high).pvalue)}
    return res


# --------------------------------------------------------------------------------------------
# EBM
# --------------------------------------------------------------------------------------------
def fit_ebms(train: pd.DataFrame, val: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from interpret.glassbox import ExplainableBoostingRegressor

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(EBM_FEATURES)
    importances, val_scores, plot_files = [], {}, {}
    warnings.filterwarnings("ignore", message="Missing values detected")  # a few clips lack WPM; EBM bins NaN separately
    for target in TARGETS:
        ebm = ExplainableBoostingRegressor(interactions=0, random_state=seed)
        ebm.fit(train[cols], train[target])
        val_scores[target] = float(stats.spearmanr(ebm.predict(val[cols]), val[target]).statistic)
        for name, imp in zip(ebm.term_names_, ebm.term_importances()):
            importances.append({"target": target, "feature": name, "importance": float(imp)})

        explanation = ebm.explain_global()
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for i, (ax, name) in enumerate(zip(axes.ravel(), ebm.term_names_)):
            data = explanation.data(i)
            scores = np.asarray(data["scores"], dtype=float)
            lower = np.asarray(data.get("lower_bounds", scores), dtype=float)
            upper = np.asarray(data.get("upper_bounds", scores), dtype=float)
            if ebm.feature_types_in_[i] == "nominal":
                labels = [str(n) for n in data["names"]]
                ax.bar(labels, scores, yerr=[scores - lower, upper - scores], color="#4C72B0", capsize=4)
                ax.tick_params(axis="x", rotation=20)
            else:
                edges = np.asarray(data["names"], dtype=float)
                # Clip the open-ended outer bins to the observed 1st-99th percentile for readability.
                lo_x, hi_x = np.nanpercentile(train[name], [1, 99])
                edges = np.clip(edges, lo_x, hi_x)
                ax.stairs(scores, edges, baseline=None, color="#4C72B0", linewidth=2)
                ax.fill_between(edges[:-1], lower, upper, step="post", alpha=0.25, color="#4C72B0")
                ax.set_xlim(lo_x, hi_x)
                rug = train[name].dropna().to_numpy()
                ax.plot(rug, np.full_like(rug, ax.get_ylim()[0]), "|", color="grey", alpha=0.15, markersize=8)
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set_title(EBM_FEATURES[name])
            ax.set_ylabel(f"contribution to {target}")
        fig.suptitle(f"EBM shape functions -> {target} (train n={len(train)}, val Spearman={val_scores[target]:.3f})")
        fig.tight_layout()
        path = PLOT_DIR / f"ebm_shapes_{target}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        plot_files[target] = path
    return pd.DataFrame(importances), {"val_spearman": val_scores, "plots": plot_files}


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------
def _fmt(row) -> str:
    return f"{row.rho:+.2f} [{row.ci_low:+.2f}, {row.ci_high:+.2f}]"


def _cell(corr: pd.DataFrame, feat: str, target: str, measure: str) -> str:
    row = corr[(corr.feature == feat) & (corr.target == target) & (corr.measure == measure)].iloc[0]
    return _fmt(row)


def write_report(df, corr, cls, qt_table, vq, ebm_imp, ebm_meta, args) -> Path:
    n_part = df["participant"].nunique()
    lines: list[str] = []
    add = lines.append
    add("# Stage 0: Confound audit\n")
    add(f"Generated by `scripts/explainability/stage0_confound_audit.py` on the **train split only** "
        f"({len(df)} clips, {n_part} participants). {args.bootstrap} participant-cluster bootstrap resamples, "
        f"seed {args.seed}. Classification threshold: 95% CI excluding 0 **and** |ρ| ≥ {args.min_effect}.\n")
    add("Measures (all Spearman-type): **raw** = between clips; **partial** = between clips controlling for "
        "duration, question type and video_quality; **between** = participant means; **within** = after "
        "subtracting each participant's mean; **within_partial** = within, additionally controlling for "
        "duration and question type. Delivery is judged on *within_partial* (on *within* for features that are "
        f"themselves duration proxies, |ρ(feature, duration)| ≥ {DURATION_PROXY_RHO}). CIs are not corrected "
        "for multiple comparisons, so treat borderline single-target results with care.\n")

    # Headline numbers
    add("## 1. Headline confounds (target: overall_performance)\n")
    add("| Measure | raw | partial | between | within | within_partial | ICC (speaker share) |")
    add("|---|---|---|---|---|---|---|")
    for feat in [DURATION, "transcript_word_count", "vad_mean_rms", "rms_energy_mean", "vad_pause_ratio",
                 "new_transcript_speech_rate_wpm", "f0_hz_mean", "new_relative_pitch_std_semitones",
                 "vad_maximum_pause_duration_seconds"]:
        icc = cls.loc[cls.feature == feat, "icc_speaker_share"].iloc[0]
        add(f"| `{feat}` | " + " | ".join(_cell(corr, feat, "overall_performance", m) for m in MEASURES)
            + f" | {icc:.2f} |")
    add("")

    # Duration / conciseness
    add("## 2. Does duration need its own category (Conciseness)?\n")
    add("| Target | raw | between | within |")
    add("|---|---|---|---|")
    within_sig = []
    for t in TARGETS:
        row = corr[(corr.feature == DURATION) & (corr.target == t) & (corr.measure == "within")].iloc[0]
        within_sig.append(row.ci_high < 0 and abs(row.rho) >= args.min_effect)
        add(f"| {t} | {_cell(corr, DURATION, t, 'raw')} | {_cell(corr, DURATION, t, 'between')} | {_fmt(row)} |")
    add("")
    if any(within_sig):
        add("**Decision: yes.** Longer answers score lower *within* the same participant too, so answer length is "
            "a delivery-level behaviour that raters penalise, not only a person-level confound. Make it an explicit "
            "**Conciseness** category (duration relative to the question-type median), and keep controlling for "
            "duration in every other category so they do not become 'you talked too long' in disguise. Note "
            "the between-participant effect is much larger than the within effect: part of it is still the person.\n")
    else:
        add("**Decision: not as a delivery category.** The duration effect does not survive within participants; "
            "treat it as a confound to control for, not as advice.\n")

    # Never-absolute list
    add("## 3. Measures that must never become absolute advice\n")
    add("A measure lands here if (a) its relation with the scores disappears within participants once duration and "
        "question type are controlled (*speaker/mic effect*: it ranks people, or just tracks answer length), "
        "or (b) at least half of its variance is the speaker (ICC ≥ 0.5) **and** it is an absolute level (loudness, "
        "Hz pitch, energy), so a percentile against other people mostly ranks voices and microphones. "
        "These may only be used **speaker-relative** (vs the person's own other answers), as a duration-normalised "
        "rate, or not at all.\n")
    level_pattern = re.compile(r"rms|f0_hz|_hz$|energy|vad_rms_threshold")
    never = cls[(cls["class"] == "speaker/mic effect")
                | ((cls["icc_speaker_share"] >= 0.5) & cls["feature"].str.contains(level_pattern))]
    add("| Feature | Class | ICC | Reason |")
    add("|---|---|---|---|")
    for _, r in never.sort_values("icc_speaker_share", ascending=False).iterrows():
        reason = []
        if r["class"] == "speaker/mic effect":
            reason.append("no within-participant relation after duration/question-type control")
        if r["icc_speaker_share"] >= 0.5 and level_pattern.search(r.feature):
            reason.append("absolute level, mostly a speaker property")
        add(f"| `{r.feature}` | {r['class']} | {r.icc_speaker_share:.2f} | {'; '.join(reason)} |")
    add("")

    # Within-participant signals
    add("## 3b. Signals that hold within participants (candidates for per-answer feedback)\n")
    add("*delivery* / *weak delivery* features: the same person scores differently on answers where this measure "
        "differs, after controlling for duration and question type. Values are the within-participant correlation "
        "used for classification (within_partial; within for duration proxies). Many rows are near-duplicates "
        "(the same quantity from the legacy, `new_` and `vad_` extractors).\n")
    add("| Feature | Class | ICC | conf | speak | overall |")
    add("|---|---|---|---|---|---|")
    keep = cls[cls["class"].isin(["delivery", "weak delivery"])].copy()
    keep["_o"] = keep["class"].map({"delivery": 0, "weak delivery": 1})
    for _, r in keep.sort_values(["_o", "feature"]).iterrows():
        key = "within" if r.duration_proxy else "within_partial"
        add(f"| `{r.feature}`{' (dur. proxy)' if r.duration_proxy else ''} | {r['class']} | {r.icc_speaker_share:.2f} | "
            + " | ".join(_cell(corr, r.feature, t, key) for t in TARGETS) + " |")
    add("")

    # Video quality
    add("## 4. Recording quality (`video_quality`)\n")
    add(f"{vq['n_low']} Low vs {vq['n_high']} High clips. {vq['participants_with_mixed_quality']} participants have "
        "clips of both qualities, so video_quality is essentially a participant/recording-setup property.\n")
    add("| Variable | median (Low) | median (High) | Mann-Whitney p |")
    add("|---|---|---|---|")
    for col in [*TARGETS, "vad_mean_rms", DURATION]:
        v = vq[col]
        add(f"| {col} | {v['median_low']:.3f} | {v['median_high']:.3f} | {v['mannwhitney_p']:.3g} |")
    add("")

    # Question types
    add("## 5. Question types\n")
    add("Labels: `Datasets/question_types.csv` (made by `scripts/explainability/make_question_types.py`). "
        "`within_mean_*` = mean of the participant-demeaned score, i.e. whether the *same* people score "
        "higher or lower on this type of question.\n")
    add("| Type | clips | median dur (s) | " + " | ".join(f"mean {TARGET_SHORT[t]}" for t in TARGETS) + " | "
        + " | ".join(f"within {TARGET_SHORT[t]}" for t in TARGETS) + " |")
    add("|---|---|---|" + "---|" * (2 * len(TARGETS)))
    for _, r in qt_table.iterrows():
        add(f"| {r.question_type} | {r.n_clips} | {r.median_duration_s:.1f} | "
            + " | ".join(f"{r[f'mean_{TARGET_SHORT[t]}']:+.3f}" for t in TARGETS) + " | "
            + " | ".join(f"{r[f'within_mean_{TARGET_SHORT[t]}']:+.3f}" for t in TARGETS) + " |")
    add("")
    add("Kruskal-Wallis p (raw scores): " + ", ".join(f"{TARGET_SHORT[t]} {p:.3g}" for t, p in qt_table.attrs["kruskal_p"].items())
        + ". Within-participant: " + ", ".join(f"{TARGET_SHORT[t]} {p:.3g}" for t, p in qt_table.attrs["kruskal_p_within"].items()) + ".\n")

    # EBM
    add("## 6. EBM shape functions\n")
    add("`ExplainableBoostingRegressor(interactions=0)` fitted on train with "
        + ", ".join(f"`{c}`" for c in EBM_FEATURES) + ". Each panel shows how the predicted score moves with "
        "that variable, holding the others fixed (shaded band = bagging uncertainty, grey ticks = data density). "
        "Validation Spearman is only a sanity check; this model is for reading shapes, not for prediction.\n")
    add("| Target | val Spearman | " + " | ".join(EBM_FEATURES.values()) + " |")
    add("|---|---|" + "---|" * len(EBM_FEATURES))
    for t in TARGETS:
        imp = ebm_imp[ebm_imp.target == t].set_index("feature")["importance"]
        add(f"| {t} | {ebm_meta['val_spearman'][t]:.3f} | " + " | ".join(f"{imp.get(c, np.nan):.3f}" for c in EBM_FEATURES) + " |")
    add("\n(Importances = mean absolute contribution, in target units.)\n")
    for t in TARGETS:
        rel = ebm_meta["plots"][t].relative_to(OUT_DIR).as_posix()
        add(f"### {t}\n\n![EBM shapes {t}]({rel})\n")

    # Full classification
    add("## 7. Feature classification\n")
    counts = cls["class"].value_counts().to_dict()
    add("Counts: " + ", ".join(f"**{k}**: {v}" for k, v in counts.items()) + ".\n")
    add("Per-target classes are listed as conf / speak / overall (D = delivery, S = speaker/mic, – = none). "
        "Correlations shown are for overall_performance. `within/raw` < 0.5 means most of the raw relation is "
        "between people even where a within effect exists.\n")
    add("| Feature | Class | per target | ρ(dur) | ICC | raw | partial | within | within_partial | within/raw |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    abbrev = {"delivery": "D", "weak delivery": "d", "speaker/mic": "S", "none": "–"}
    order = {"delivery": 0, "weak delivery": 1, "speaker/mic effect": 2, "no relation": 3}
    cls_sorted = cls.assign(_o=cls["class"].map(order)).sort_values(["_o", "feature"])
    for _, r in cls_sorted.iterrows():
        per = "/".join(abbrev[r[f"class_{TARGET_SHORT[t]}"]] for t in TARGETS)
        ratio = "" if pd.isna(r.within_over_raw_overall) else f"{r.within_over_raw_overall:.2f}"
        proxy = " (dur. proxy)" if r.duration_proxy else ""
        add(f"| `{r.feature}`{proxy} | {r['class']} | {per} | {r.rho_with_duration:+.2f} | {r.icc_speaker_share:.2f} | "
            + " | ".join(_cell(corr, r.feature, "overall_performance", m) for m in ("raw", "partial", "within", "within_partial"))
            + f" | {ratio} |")
    add("")
    add("Full numbers for all targets and measures: `tables/correlations.csv`; one row per feature: "
        "`tables/classification.csv`.\n")

    path = OUT_DIR / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=1000, help="participant-cluster bootstrap resamples")
    parser.add_argument("--min-effect", type=float, default=0.10, help="minimum |rho| to count a relation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading train/val tables...")
    train, features = load_split("train")
    val, _ = load_split("val")
    train_users = set((SPLIT_DIR / "train_users.txt").read_text().split())
    if not set(train["participant"].astype(str)) <= train_users:
        raise RuntimeError("Train table contains participants outside train_users.txt")
    print(f"  train: {len(train)} clips, {train['participant'].nunique()} participants, {len(features)} features")

    print("Correlation audit...")
    corr = correlation_audit(train, features, args.bootstrap, args.seed)
    cls = classify(corr, train, features, args.min_effect)
    corr.to_csv(TABLE_DIR / "correlations.csv", index=False)
    cls.to_csv(TABLE_DIR / "classification.csv", index=False)

    print("Question type / video quality effects...")
    qt_table = question_type_effects(train)
    vq = video_quality_effects(train)

    print("Fitting EBMs...")
    ebm_imp, ebm_meta = fit_ebms(train, val, args.seed)
    ebm_imp.to_csv(TABLE_DIR / "ebm_importances.csv", index=False)

    report = write_report(train, corr, cls, qt_table, vq, ebm_imp, ebm_meta, args)
    print(f"Done. Report: {report}")
    print(cls["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
