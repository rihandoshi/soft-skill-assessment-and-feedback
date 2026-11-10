"""Stage 3: validate and calibrate the delivery signals on RecruitView (Path B).

Questions (markdown/EXPLAINABILITY_STAGES.md, Stage 3):
  1. Per-signal check     Does each Stage 1 signal / Stage 2 category score / flag relate to the rater scores
                          WITHIN participants (after duration + question type)? A signal that only works
                          between participants measures the person, not the delivery. (Train split, unique
                          recordings, participant-bootstrap CIs; same machinery as Stage 0.)
  2. Optimal bands        EBM on train: [signals + duration + question type] -> overall_performance. Shape of
                          each pace / variability signal; band = region within 10% of the shape's peak.
                          Participant-bootstrap CIs on the band edges. Saved for Stage 2 to use.
  3. Concept bottleneck   Ridge and EBM on the delivery signals only, in the SAME repeated participant-grouped
                          CV folds as the WavLM baseline (results/repeated_cv_v1/cv_splits.json), with and
                          without duration. Compared with WavLM L7 + Ridge pooled OOF Spearman.
  4. Faithfulness         How much of WavLM's out-of-fold prediction can the signals explain (cross-validated R²)?
                          Low R² -> the feedback must not say "your score was low BECAUSE ...".
  5. Grouped attribution  Per-clip contribution of each category to the bottleneck models' predictions
                          (exact for Ridge = linear SHAP; EBM additive terms), summed by category.
  6. Decisions            keep / weak / revise / drop for every signal, category and flag.

The WavLM OOF predictions come from the original repeated-CV run (git commit 45bae4d; the file was later
removed from the repo for size). They are extracted to outputs/explainability/ (gitignored); the
legacy script is not modified or re-run.

Outputs (results/explainability/stage3_recruitview_validation/):
  report.md                        generated report (notes.md = hand-written interpretation)
  decisions.csv / decisions.json   keep / weak / revise / drop per signal, category score and flag
  bands.json                       EBM shapes and optimal bands (read by fit_reference_distributions.py)
  tables/signal_checks.csv         per signal x target x measure (raw / partial / between / within / within_partial)
  tables/bottleneck.csv            bottleneck CV results vs WavLM
  tables/faithfulness.csv          R² of WavLM OOF predictions explained by the signals
  tables/category_contributions.csv per-clip category contributions (OOF, repeat 0)
  tables/rater_noise.csv           rating noise estimated from duplicate recordings
  plots/ebm_bands_overall.png      EBM shapes with bands

Run:  python scripts/explainability/stage3_validate_recruitview.py [--bootstrap 2000] [--band-bootstrap 20]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage0_confound_audit as s0  # noqa: E402

warnings.filterwarnings("ignore", message="Missing values detected")
ROOT = Path(__file__).resolve().parents[2]
import paths as P  # noqa: E402  (results layout)
import descriptor_tables as DT  # noqa: E402  (descriptor tables + Stage 5 columns)
DESC = P.S1
OUT = P.S3
TABLES = P.S3_TABLES
PLOTS = P.S3_PLOTS
CV_SPLITS = P.CV_SPLITS
OOF_CACHE = P.WAVLM_OOF
OOF_COMMIT = "45bae4d"
TARGETS = s0.TARGETS
SHORT = s0.TARGET_SHORT

# Answer-level signals entering the analyses, with their category and the direction we ASSUME when we use
# them for feedback (sign of the expected correlation with the rater scores, within participants).
SIGNALS = {
    # pace (active)
    "speech_rate_wpm": ("pace", -1), "artic_rate_sps": ("pace", -1), "sent_rate_mean_wpm": ("pace", -1),
    "sent_rate_max_wpm": ("pace", -1), "sent_rate_cv": ("pace", -1), "rushed_opening_ratio": ("pace", -1),
    "rate_drift_ratio": ("pace", 0),
    # pausing (active)
    "response_latency_s": ("pausing", -1), "pause_ratio": ("pausing", +1), "mid_pause_rate_per_min": ("pausing", -1),
    "long_mid_pause_rate_per_min": ("pausing", -1), "mid_pause_max_s": ("pausing", -1),
    "mid_pause_time_ratio": ("pausing", -1), "between_pause_time_ratio": ("pausing", +1),
    # expressiveness / projection / assertiveness / composure (experimental)
    "pitch_std_st": ("expressiveness", +1), "pitch_range_st": ("expressiveness", +1),
    "intensity_std_db": ("expressiveness", +1), "monotony_frac": ("expressiveness", -1),
    "end_drop_median_db": ("projection", +1), "terminal_slope_median_st_s": ("assertiveness", -1),
    "f0_median_st": ("composure", -1), "jitter_local": ("composure", -1), "shimmer_local": ("composure", -1),
}
DURATION_SIGNALS = {"audio_duration_s": ("conciseness", -1), "speech_span_s": ("conciseness", -1),
                    "speech_time_s": ("conciseness", -1), "word_count": ("conciseness", -1),
                    "n_sentences": ("conciseness", -1)}
DIAGNOSTIC_SIGNALS = {"intensity_mean_db": ("diagnostic", 0), "f0_median_hz": ("diagnostic", 0)}
# Stage 4 columns: correlation tests only (kept out of the bottleneck models so they stay comparable). ASR word
# confidence = intelligibility proxy (but also accent / microphone); filled-pause rate is reported for the text
# branch, which owns filler feedback.
ASR_SIGNALS = ({"asr_prob_mean": ("projection", +1), "filled_pause_rate_per_min": ("pausing", -1)}
               if P.DESCRIPTOR_SOURCE == "asr" else {})
# Stage 5 SER columns: correlation tests only (dominance = "sounds assured": expected + with the scores; arousal:
# no direction assumed; the speaker-relative moment measures: expected -).
SER_SIGNALS = ({"ser_dominance_mean": ("composure", +1), "ser_arousal_mean": ("composure", -1),
                "ser_valence_mean": ("diagnostic", 0), "ser_dominance_drop_sd": ("composure", -1),
                "ser_tension_run_s": ("composure", -1)} if P.S5_SEGMENTS.exists() else {})
CATEGORY_SCORES = {"category_pace": ("pace", +1), "category_pausing": ("pausing", +1),
                   "category_conciseness": ("conciseness", +1)}
# Stage 2 category-profile indices (0-100, toward each category's "high" pole). Expected sign: the pole that the
# feedback treats as needing attention should go with LOWER scores (pace: fast; expressiveness: monotone = low
# pole, so +; pausing: no assumption, raters reward pausing).
PROFILE_INDICES = {"pace_index": ("pace", -1), "pausing_fluency_index": ("pausing", 0),
                   "expressiveness_index": ("expressiveness", +1), "projection_index": ("projection", -1),
                   "composure_index": ("composure", -1), "assertiveness_index": ("assertiveness", -1),
                   "conciseness_index": ("conciseness", -1)}
BAND_SIGNALS = ["speech_rate_wpm", "artic_rate_sps", "sent_rate_cv", "rushed_opening_ratio", "mid_pause_rate_per_min",
                "mid_pause_max_s", "speech_span_s", "word_count", "pitch_std_st", "pause_ratio"]
BAND_TOL = 0.10  # band = where the shape is within 10% of its range below the peak
BAND_IDENTIFIED_MAX_CI = 0.25  # a band edge counts as identified if its 95% CI spans < 25% of the observed p5-p95 range
DURATION_BASED = set(DURATION_SIGNALS) | {"category_conciseness", "flag_long_answer", "conciseness_index"}  # judged on plain within (controlling duration would remove the signal itself)


# ============================================================================================
# Data
# ============================================================================================
def load_targets() -> pd.DataFrame:
    frames = [pd.read_csv(ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str})
              [["id", "user_no", "video_quality", *TARGETS]] for s in ("train", "val", "test")]
    return pd.concat(frames, ignore_index=True)


def load_all() -> pd.DataFrame:
    dt = {"id": str, "participant": str, "duplicate_of": str}
    seg = DT.load_segments()
    scores = pd.read_csv(P.S2_TABLES / "category_scores.csv", dtype={"id": str})
    flags = pd.read_csv(P.S2_TABLES / "flags.csv", dtype={"id": str})
    df = seg.merge(load_targets().drop(columns=["video_quality"]), on="id")
    df = df.merge(scores[["id", *[c for c in CATEGORY_SCORES if c in scores]]], on="id", how="left")
    profile_path = P.S2_TABLES / "category_profile.csv"
    if profile_path.exists():
        prof = pd.read_csv(profile_path, dtype={"id": str})
        df = df.merge(prof[["id", *[c for c in PROFILE_INDICES if c in prof]]], on="id", how="left")
    for ftype, g in flags.groupby("type"):
        df[f"flag_{ftype}"] = df["id"].isin(set(g["id"])).astype(float)
    df["duration_s"] = df["audio_duration_s"]
    df["video_quality_low"] = (df["video_quality"].str.lower() == "low").astype(float)
    return df


def unique_recordings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per recording; targets = mean over the independently rated copies (less label noise)."""
    means = df.groupby("duplicate_of")[TARGETS].mean()
    out = df[~df["is_duplicate"]].copy()
    out[TARGETS] = means.loc[out["duplicate_of"]].to_numpy()
    return out.reset_index(drop=True)


def flag_signals(df: pd.DataFrame) -> dict:
    return {c: ("flag", -1) for c in df.columns if c.startswith("flag_")}


# ============================================================================================
# 1. Per-signal checks + 6. decisions
# ============================================================================================
def signal_checks(train_u: pd.DataFrame, features: dict, n_boot: int, seed: int, min_effect: float):
    cols = [c for c in features if c in train_u and train_u[c].notna().sum() > 50 and train_u[c].nunique() > 1]
    corr = s0.correlation_audit(train_u, cols, n_boot, seed)
    cls = s0.classify(corr, train_u, cols, min_effect)
    return corr, cls


def decide(corr: pd.DataFrame, cls: pd.DataFrame, features: dict, bands: dict) -> pd.DataFrame:
    """keep: expected direction holds within participants (|rho| >= min effect) for >= 1 target, no target
    significantly the other way. weak: same but only small effects. revise: relation exists but only between
    people, or in the opposite direction, or no direction assumed. drop: no relation at all."""
    rows = []
    for _, c in cls.iterrows():
        feat = c["feature"]
        group, expected = features[feat]
        key = "within" if (c["duration_proxy"] or feat in DURATION_BASED) else "within_partial"
        w = corr[(corr.feature == feat) & (corr.measure == key)].set_index("target")
        sig_pos = [t for t in TARGETS if w.loc[t, "ci_low"] > 0]
        sig_neg = [t for t in TARGETS if w.loc[t, "ci_high"] < 0]
        best_t = max(TARGETS, key=lambda t: abs(w.loc[t, "rho"]))
        agree = sig_pos if expected > 0 else sig_neg if expected < 0 else []
        oppose = sig_neg if expected > 0 else sig_pos if expected < 0 else []
        strong = [t for t in agree if abs(w.loc[t, "rho"]) >= 0.10]
        if expected == 0:
            decision = "revise" if (sig_pos or sig_neg) else ("revise" if c["class"] == "speaker/mic effect" else "drop")
            reason = "no direction assumed; " + (f"within effect {'+' if sig_pos else '-'} on {', '.join(map(SHORT.get, sig_pos or sig_neg))}"
                                                  if sig_pos or sig_neg else c["class"])
        elif oppose and not agree:
            decision, reason = "revise", f"significant in the OPPOSITE direction within participants ({', '.join(map(SHORT.get, oppose))})"
        elif strong:
            decision, reason = "keep", f"expected direction within participants, |rho| >= 0.10 ({', '.join(map(SHORT.get, strong))})"
        elif agree:
            decision, reason = "weak", f"expected direction within participants but |rho| < 0.10 ({', '.join(map(SHORT.get, agree))})"
        elif c["class"] == "speaker/mic effect":
            decision, reason = "revise", "relation only between participants (measures the person)"
        else:
            decision, reason = "drop", "no relation with any target"
        b = bands.get(feat)
        rows.append({"signal": feat, "group": group, "expected_sign": expected, "decision": decision, "reason": reason,
                     "stage0_style_class": c["class"], "icc_speaker_share": c["icc_speaker_share"],
                     **{f"within_{SHORT[t]}": w.loc[t, "rho"] for t in TARGETS},
                     **{f"within_{SHORT[t]}_ci": f"[{w.loc[t, 'ci_low']:+.2f}, {w.loc[t, 'ci_high']:+.2f}]" for t in TARGETS},
                     "strongest_target": SHORT[best_t], "ebm_shape": b["shape"] if b else ""})
    return pd.DataFrame(rows)


# ============================================================================================
# 2. EBM optimal bands
# ============================================================================================
EBM_COLUMNS = list(SIGNALS) + ["speech_span_s", "word_count", "question_type"]


def _ebm(seed: int, bags: int = 8):
    from interpret.glassbox import ExplainableBoostingRegressor
    return ExplainableBoostingRegressor(interactions=0, outer_bags=bags, random_state=seed, n_jobs=-1)


def shape_and_band(ebm, col: str, x: pd.Series) -> dict:
    i = list(ebm.term_names_).index(col)
    data = ebm.explain_global().data(i)
    edges = np.asarray(data["names"], dtype=float)
    scores = np.asarray(data["scores"], dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    lo_x, hi_x = np.nanpercentile(x, [5, 95])
    m = (centres >= lo_x) & (centres <= hi_x)
    c, s = centres[m], scores[m]
    if len(s) < 3:
        return {"shape": "insufficient", "band_value": [float(lo_x), float(hi_x)], "range": 0.0}
    rng = float(s.max() - s.min())
    keep = s >= s.max() - BAND_TOL * rng
    lo, hi = float(c[keep].min()), float(c[keep].max())
    # Shape over the observed p5-p95 range.
    first, last = s[: max(1, len(s) // 4)].mean(), s[-max(1, len(s) // 4):].mean()
    if rng < 0.05:
        shape = "flat"
    elif not keep[0] and not keep[-1]:
        shape = "peaked (inverted U)"
    elif keep[0] and not keep[-1]:
        shape = "decreasing (lower is better)"
    elif keep[-1] and not keep[0]:
        shape = "increasing (higher is better)"
    else:
        shape = "U-shaped or flat-topped"
    return {"shape": shape, "band_value": [lo, hi], "range": rng, "first_quarter": float(first), "last_quarter": float(last),
            "observed_p5_p95": [float(lo_x), float(hi_x)], "curve": {"x": c.round(4).tolist(), "score": s.round(4).tolist()}}


def ebm_bands(train_u: pd.DataFrame, n_boot: int, seed: int) -> tuple[dict, dict]:
    X, y = train_u[EBM_COLUMNS], train_u["overall_performance"]
    main = _ebm(seed, bags=14).fit(X, y)
    bands = {c: shape_and_band(main, c, train_u[c]) for c in BAND_SIGNALS}
    imp = dict(zip(main.term_names_, main.term_importances()))
    parts = train_u["participant"].unique()
    rng = np.random.default_rng(seed)
    boot = {c: [] for c in BAND_SIGNALS}
    t0 = time.time()
    for b in range(n_boot):
        draw = rng.choice(parts, len(parts), replace=True)
        idx = np.concatenate([np.flatnonzero(train_u["participant"].to_numpy() == p) for p in draw])
        e = _ebm(seed + b + 1, bags=4).fit(X.iloc[idx], y.iloc[idx])
        for c in BAND_SIGNALS:
            boot[c].append(shape_and_band(e, c, train_u[c])["band_value"])
    print(f"  band bootstrap: {n_boot} fits in {time.time() - t0:.0f}s", flush=True)
    for c in BAND_SIGNALS:
        arr = np.array(boot[c]) if boot[c] else np.empty((0, 2))
        if len(arr):
            lo_ci = np.percentile(arr[:, 0], [2.5, 97.5])
            hi_ci = np.percentile(arr[:, 1], [2.5, 97.5])
            bands[c]["band_ci"] = {"lower_edge_95": lo_ci.round(3).tolist(), "upper_edge_95": hi_ci.round(3).tolist()}
            span = np.subtract(*bands[c]["observed_p5_p95"][::-1]) if "observed_p5_p95" in bands[c] else np.nan
            widths = [(lo_ci[1] - lo_ci[0]) / span, (hi_ci[1] - hi_ci[0]) / span] if span and np.isfinite(span) else [np.inf, np.inf]
            # A monotonic shape has no interior optimum: its "band" is just the end of the observed range (e.g. the
            # shortest answers), and narrow CIs there only reflect that the edge is pinned to p5/p95. Only peaked or
            # U-shaped curves can define an optimal band.
            interior = bands[c]["shape"].startswith(("peaked", "U-shaped"))
            bands[c]["band_identified"] = bool(interior and max(widths) < BAND_IDENTIFIED_MAX_CI)
            bands[c]["edge_ci_width_share"] = [round(float(w), 3) for w in widths]
        bands[c]["ebm_importance"] = float(imp[c])
    return bands, {"importances": imp, "model": main}


def plot_bands(model, train_u: pd.DataFrame, bands: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PLOTS.mkdir(parents=True, exist_ok=True)
    n = len(BAND_SIGNALS)
    fig, axes = plt.subplots((n + 3) // 4, 4, figsize=(18, 4 * ((n + 3) // 4)))
    for ax, col in zip(axes.ravel(), BAND_SIGNALS):
        b = bands[col]
        i = list(model.term_names_).index(col)
        data = model.explain_global().data(i)
        edges = np.asarray(data["names"], float)
        sc = np.asarray(data["scores"], float)
        lo, hi = np.nanpercentile(train_u[col], [1, 99])
        ax.stairs(sc, np.clip(edges, lo, hi), baseline=None, color="#4C72B0", lw=2)
        ax.axvspan(*b["band_value"], color="#55A868", alpha=0.2, label="band (within 10% of peak)")
        p5, p95 = b.get("observed_p5_p95", [lo, hi])
        ax.axvline(p5, color="grey", ls=":", lw=1)
        ax.axvline(p95, color="grey", ls=":", lw=1)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlim(lo, hi)
        ax.set_title(f"{col}\n{b['shape']}", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("EBM shape functions -> overall_performance (train, unique recordings). Dotted: p5/p95 of data.")
    fig.tight_layout()
    path = PLOTS / "ebm_bands_overall.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


# ============================================================================================
# 3. Concept bottleneck + 4. faithfulness + 5. grouped attribution (same folds as WavLM)
# ============================================================================================
def ensure_oof() -> pd.DataFrame:
    if not OOF_CACHE.exists():
        OOF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        blob = subprocess.run(["git", "show", f"{OOF_COMMIT}:results/repeated_cv_v1/oof_predictions.csv"],
                              cwd=ROOT, capture_output=True, check=True).stdout
        OOF_CACHE.write_bytes(blob)
    oof = pd.read_csv(OOF_CACHE, dtype={"id": str, "user_no": str})
    return oof[(oof["feature_set"] == "WavLM") & (oof["model"] == "Ridge")]


def ridge_pipeline():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("ridge", Ridge())])


def fit_ridge(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seed: int):
    from sklearn.model_selection import GridSearchCV, GroupKFold
    grid = GridSearchCV(ridge_pipeline(), {"ridge__alpha": np.logspace(-2, 4, 13)},
                        cv=GroupKFold(n_splits=5), scoring="neg_root_mean_squared_error")
    grid.fit(X, y, groups=groups)
    return grid.best_estimator_


def design(df: pd.DataFrame, with_duration: bool) -> pd.DataFrame:
    cols = list(SIGNALS) + (list(DURATION_SIGNALS) if with_duration else [])
    X = df[cols].astype(float).copy()
    for q in s0.QTYPE_ORDER[1:]:
        X[f"qtype_{q}"] = (df["question_type"] == q).astype(float)
    return X


def category_of(col: str) -> str:
    if col.startswith("qtype_"):
        return "question type"
    return {**SIGNALS, **DURATION_SIGNALS}[col][0]


def bottleneck(trainval: pd.DataFrame, oof: pd.DataFrame, ebm_repeats: int, seed: int):
    splits = json.loads(CV_SPLITS.read_text())["assignments"]
    groups = trainval["participant"].to_numpy()
    preds, contribs = [], []
    t0 = time.time()
    for with_dur in (False, True):
        X = design(trainval, with_dur)
        for target in TARGETS:
            y = trainval[target].to_numpy(float)
            for a in splits:
                va = trainval["participant"].isin(set(a["validation_user_no"])).to_numpy()
                tr = trainval["participant"].isin(set(a["train_user_no"])).to_numpy()
                ridge = fit_ridge(X[tr], y[tr], groups[tr], seed)
                p = ridge.predict(X[va])
                preds.append(pd.DataFrame({"id": trainval.loc[va, "id"].to_numpy(), "target": target, "model": "Ridge",
                                           "with_duration": with_dur, "repeat": a["repeat"], "fold": a["fold"],
                                           "actual": y[va], "prediction": p}))
                if a["repeat"] == 0:  # exact linear attribution on held-out clips: coef x standardised value
                    z = ridge[:-1].transform(X[va])
                    c = z * ridge[-1].coef_
                    cdf = pd.DataFrame(c, columns=X.columns).T.groupby(category_of).sum().T
                    cdf.insert(0, "id", trainval.loc[va, "id"].to_numpy())
                    contribs.append(cdf.assign(target=target, model="Ridge", with_duration=with_dur))
                if a["repeat"] < ebm_repeats:
                    ebm = _ebm(seed + a["repeat"] * 10 + a["fold"]).fit(X[tr], y[tr])
                    pe = ebm.predict(X[va])
                    preds.append(pd.DataFrame({"id": trainval.loc[va, "id"].to_numpy(), "target": target, "model": "EBM",
                                               "with_duration": with_dur, "repeat": a["repeat"], "fold": a["fold"],
                                               "actual": y[va], "prediction": pe}))
                    if a["repeat"] == 0:
                        terms = ebm.eval_terms(X[va])
                        cdf = pd.DataFrame(terms, columns=ebm.term_names_).T.groupby(category_of).sum().T
                        cdf.insert(0, "id", trainval.loc[va, "id"].to_numpy())
                        contribs.append(cdf.assign(target=target, model="EBM", with_duration=with_dur))
            print(f"  bottleneck with_duration={with_dur} {target}: {time.time() - t0:.0f}s", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    rows = []
    for (model, wd, target), g in preds.groupby(["model", "with_duration", "target"]):
        rows.append({"model": f"signals-{model}", "with_duration": wd, "target": target, "repeats": g["repeat"].nunique(),
                     "pooled_spearman": stats.spearmanr(g["actual"], g["prediction"]).statistic,
                     "pooled_pearson": stats.pearsonr(g["actual"], g["prediction"]).statistic})
    for target, g in oof.groupby("target"):
        rows.append({"model": "WavLM L7 + Ridge", "with_duration": None, "target": target, "repeats": g["repeat"].nunique(),
                     "pooled_spearman": stats.spearmanr(g["actual"], g["prediction"]).statistic,
                     "pooled_pearson": stats.pearsonr(g["actual"], g["prediction"]).statistic})
    res = pd.DataFrame(rows)
    wav = res[res.model == "WavLM L7 + Ridge"].set_index("target")["pooled_spearman"]
    res["share_of_wavlm"] = res["pooled_spearman"] / res["target"].map(wav)
    return res, preds, pd.concat(contribs, ignore_index=True)


def faithfulness(trainval: pd.DataFrame, oof: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cross-validated R² of WavLM's OOF prediction (mean over repeats) explained by delivery signals."""
    from sklearn.model_selection import GroupKFold, cross_val_predict
    mean_pred = oof.groupby(["target", "id"])["prediction"].mean()
    groups = trainval["participant"].to_numpy()
    variants = {
        "duration only": ["audio_duration_s"],
        "duration + loudness (confounds)": ["audio_duration_s", "intensity_mean_db"],
        "delivery signals (no duration)": list(SIGNALS),
        "delivery signals + duration": list(SIGNALS) + list(DURATION_SIGNALS),
        "delivery signals + duration + loudness": list(SIGNALS) + list(DURATION_SIGNALS) + ["intensity_mean_db"],
    }
    rows = []
    for target in TARGETS:
        y = mean_pred.loc[target].reindex(trainval["id"]).to_numpy()
        ok = np.isfinite(y)
        for name, cols in variants.items():
            X = trainval.loc[ok, cols].astype(float)
            cv = GroupKFold(n_splits=5)
            from sklearn.linear_model import RidgeCV
            from sklearn.pipeline import make_pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.preprocessing import StandardScaler
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
            p = cross_val_predict(model, X, y[ok], groups=groups[ok], cv=cv)
            r2_cv = 1 - np.sum((y[ok] - p) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
            r2_in = model.fit(X, y[ok]).score(X, y[ok])
            rows.append({"target": target, "explaining_variables": name, "n_vars": len(cols),
                         "r2_cv": r2_cv, "r2_in_sample": r2_in})
    return pd.DataFrame(rows)


# ============================================================================================
# Report
# ============================================================================================
def label_ceiling() -> pd.DataFrame:
    path = P.EXTRACTION_REPORT
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ") and any(t in line for t in TARGETS) and "Spearman" not in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) == 5:
                try:
                    rows.append({"target": parts[0], "copy_agreement_spearman": float(parts[2])})
                except ValueError:
                    pass
    return pd.DataFrame(rows)


def rater_noise(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Byte-identical copies of a recording were rated independently, so half the mean squared difference
    between their scores estimates the rater-noise variance of a single score. Compared with the variance of
    scores within a participant (all splits, unique recordings), it says how much of the answer-to-answer
    score variation the within-participant tests can possibly explain. CIs resample duplicate pairs."""
    rng = np.random.default_rng(seed)
    dup = df[df["duplicate_group_size"] > 1]
    first = dup[~dup["is_duplicate"]].set_index("duplicate_of")
    pairs = dup[dup["is_duplicate"]].join(first[TARGETS], on="duplicate_of", rsuffix="_first")
    u = df[~df["is_duplicate"]]
    rows = []
    for t in TARGETS:
        d2 = ((pairs[t] - pairs[f"{t}_first"]) ** 2).to_numpy()
        noise = d2.mean() / 2
        boot = np.array([d2[rng.integers(0, len(d2), len(d2))].mean() / 2 for _ in range(n_boot)])
        within = (u[t] - u.groupby("participant")[t].transform("mean")).pow(2).sum() / (len(u) - u["participant"].nunique())
        total = u[t].var()
        rows.append({"target": t, "n_pairs": len(d2), "rater_noise_var": noise,
                     "noise_var_ci": np.percentile(boot, [2.5, 97.5]).round(3).tolist(),
                     "within_participant_var": within, "total_var": total,
                     "noise_share_of_within": noise / within,
                     "noise_share_of_within_ci": (np.percentile(boot, [2.5, 97.5]) / within).round(2).tolist(),
                     "noise_share_of_total": noise / total})
    return pd.DataFrame(rows)


def write_report(n_train, decisions, corr, bands, imps, bott, faith, contrib, plot_path, args, noise=None) -> Path:
    L = []
    add = L.append
    add("# Stage 3: Validation and calibration on RecruitView\n")
    add(f"Descriptor tables: **{P.DESCRIPTOR_SOURCE}** (`{P.SEGMENTS.relative_to(P.ROOT).as_posix()}`).\n")
    add(f"Generated by `scripts/explainability/stage3_validate_recruitview.py`. Per-signal checks and EBM bands use the "
        f"**train split only** ({n_train} unique recordings; targets averaged over duplicate copies). The concept "
        "bottleneck and faithfulness checks use the **same 5x5 participant-grouped folds** over train+val as the WavLM "
        "baseline (`results/repeated_cv_v1/cv_splits.json`). Test is not used.\n")

    add("## 1. Headline\n")
    wav = bott[bott.model == "WavLM L7 + Ridge"].set_index("target")["pooled_spearman"]
    best = bott[bott.model != "WavLM L7 + Ridge"]
    nod = best[best.with_duration == False].groupby("target")["pooled_spearman"].max()  # noqa: E712
    wd = best[best.with_duration == True].groupby("target")["pooled_spearman"].max()  # noqa: E712
    ceil = label_ceiling().set_index("target")["copy_agreement_spearman"] if len(label_ceiling()) else pd.Series(dtype=float)
    add("| Target | WavLM L7 + Ridge | delivery signals (no duration) | + duration | share of WavLM (no dur / with dur) | label agreement between duplicate copies |")
    add("|---|---|---|---|---|---|")
    for t in TARGETS:
        add(f"| {t} | {wav[t]:.3f} | {nod[t]:.3f} | {wd[t]:.3f} | {nod[t] / wav[t]:.0%} / {wd[t] / wav[t]:.0%} | "
            f"{ceil.get(t, float('nan')):.2f} |")
    add("\nPooled out-of-fold Spearman over all repeats (best of Ridge / EBM). The last column is the Spearman between the "
        "scores RecruitView raters gave to byte-identical copies of the same recording (Stage 1): a rough ceiling for any "
        "audio-only model.\n")
    counts = decisions["decision"].value_counts().to_dict()
    add("Decisions: " + ", ".join(f"**{k}** {v}" for k, v in counts.items()) + " (details in §6).\n")

    if noise is not None:
        add("### How much of a participant's answer-to-answer score variation is rater noise?\n")
        add("RecruitView contains byte-identical copies of the same recording, rated independently (Stage 1). Half the "
            "mean squared score difference between copies estimates the noise variance of a single rating. If it is close "
            "to the within-participant score variance, the within-participant tests below have almost no real signal to "
            "detect, whatever the delivery signals do.\n")
        add("| Target | duplicate pairs | rater-noise variance (95% CI) | within-participant variance | noise / within (95% CI) | noise / total |")
        add("|---|---|---|---|---|---|")
        for _, r in noise.iterrows():
            add(f"| {r.target} | {r.n_pairs} | {r.rater_noise_var:.2f} {r.noise_var_ci} | {r.within_participant_var:.2f} | "
                f"{r.noise_share_of_within:.0%} {r.noise_share_of_within_ci} | {r.noise_share_of_total:.0%} |")
        add("\nThe duplicate pairs are few, so these are rough, but the message is clear: most of the variation between one "
            "person's answers in RecruitView is rating noise. RecruitView can show what raters reward **between people**; "
            "it has little power to show what changes a rating **from one answer to the next**. That per-answer validation "
            "has to come from MIT (Stage 7) and the in-house annotation (Stage 8).\n")

    add("## 2. Per-signal checks (train, unique recordings)\n")
    add("Same measures as Stage 0. **within_partial** = within participants, controlling for duration and question type "
        "(the delivery test). 95% participant-bootstrap CIs. Expected sign = the direction the feedback assumes "
        "(e.g. more mid-sentence pauses -> lower score).\n")
    add("| Signal | group | expected | within_partial conf | within_partial speak | within_partial overall | raw overall | between overall | ICC | decision |")
    add("|---|---|---|---|---|---|---|---|---|---|")

    def cell(feat, t, m):
        r = corr[(corr.feature == feat) & (corr.target == t) & (corr.measure == m)]
        return f"{r.rho.iloc[0]:+.2f} [{r.ci_low.iloc[0]:+.2f}, {r.ci_high.iloc[0]:+.2f}]" if len(r) else ""
    order = {"keep": 0, "weak": 1, "revise": 2, "drop": 3}
    for _, d in decisions.assign(_o=decisions.decision.map(order)).sort_values(["group", "_o", "signal"]).iterrows():
        key = "within" if d.signal in DURATION_BASED else "within_partial"
        add(f"| `{d.signal}` | {d.group} | {'+' if d.expected_sign > 0 else '-' if d.expected_sign < 0 else '?'} | "
            + " | ".join(cell(d.signal, t, key) for t in TARGETS)
            + f" | {cell(d.signal, 'overall_performance', 'raw')} | {cell(d.signal, 'overall_performance', 'between')} | "
            f"{d.icc_speaker_share:.2f} | **{d.decision}** |")
    add("\nFor conciseness signals (duration proxies) the within column is *within* (not partial: controlling duration "
        "for duration is meaningless).\n")

    add("## 3. Optimal bands (EBM shape functions, overall_performance)\n")
    add(f"`ExplainableBoostingRegressor(interactions=0)` on train with all delivery signals + duration + question type. "
        f"Band = values where the shape is within {BAND_TOL:.0%} of its range below the peak, inside the observed p5-p95. "
        f"Band-edge CIs from {args.band_bootstrap} participant-bootstrap refits.\n")
    add(f"A band counts as **identified** only if both edges' 95% CIs span < {BAND_IDENTIFIED_MAX_CI:.0%} of the observed "
        "p5-p95 range **and** the shape has an interior optimum (peaked or U-shaped). For monotonic shapes the 'band' is just the end of the observed range, so there is no optimum to calibrate; Stage 2 then keeps its provisional percentile band and uses the shape only as the direction.\n")
    add("| Signal | shape | band | lower edge 95% CI | upper edge 95% CI | observed p5-p95 | identified? | EBM importance |")
    add("|---|---|---|---|---|---|---|---|")
    for c, b in bands.items():
        ci = b.get("band_ci", {})
        add(f"| `{c}` | {b['shape']} | {b['band_value'][0]:.3g} - {b['band_value'][1]:.3g} | {ci.get('lower_edge_95', '')} | "
            f"{ci.get('upper_edge_95', '')} | {b['observed_p5_p95'][0]:.3g} - {b['observed_p5_p95'][1]:.3g} | "
            f"{'yes' if b.get('band_identified') else 'no'} | {b['ebm_importance']:.3f} |")
    add(f"\n![EBM bands]({plot_path.relative_to(OUT).as_posix()})\n")

    add("## 4. Concept bottleneck (same folds as WavLM)\n")
    add("| Model | with duration | target | repeats | pooled Spearman | pooled Pearson | share of WavLM |")
    add("|---|---|---|---|---|---|---|")
    for _, r in bott.sort_values(["target", "model", "with_duration"]).iterrows():
        add(f"| {r.model} | {'' if r.with_duration is None else r.with_duration} | {r.target} | {r.repeats} | "
            f"{r.pooled_spearman:.3f} | {r.pooled_pearson:.3f} | {r.share_of_wavlm:.0%} |")
    add("")

    add("## 5. Faithfulness: how much of WavLM's prediction do the signals explain?\n")
    add("Target = WavLM L7 + Ridge out-of-fold prediction (mean over 5 repeats) for each train+val clip. R² from a "
        "participant-grouped 5-fold Ridge (cross-validated), and in-sample for reference.\n")
    add("| Target | explaining variables | # vars | R² (CV) | R² (in-sample) |")
    add("|---|---|---|---|---|")
    for _, r in faith.iterrows():
        add(f"| {r.target} | {r.explaining_variables} | {r.n_vars} | {r.r2_cv:.3f} | {r.r2_in_sample:.3f} |")
    add("")

    add("## 6. Grouped attribution (which category moves the prediction)\n")
    add("Mean absolute per-clip contribution of each category to the bottleneck model's held-out prediction (repeat 0; "
        "Ridge = exact linear attribution on standardised inputs, EBM = additive term scores), in target units.\n")
    imp = contrib.drop(columns=["id"]).groupby(["model", "with_duration", "target"]).agg(lambda v: np.abs(v).mean())
    cats = [c for c in imp.columns]
    add("| model | with duration | target | " + " | ".join(cats) + " |")
    add("|---|---|---|" + "---|" * len(cats))
    for (m, wdur, t), r in imp.iterrows():
        add(f"| {m} | {wdur} | {t} | " + " | ".join(f"{r[c]:.3f}" for c in cats) + " |")
    add("\nPer-clip values: `tables/category_contributions.csv`.\n")

    add("## 7. Decisions\n")
    add("| Signal | group | decision | reason | EBM shape |")
    add("|---|---|---|---|---|")
    for _, d in decisions.assign(_o=decisions.decision.map(order)).sort_values(["group", "_o"]).iterrows():
        add(f"| `{d.signal}` | {d.group} | **{d.decision}** | {d.reason} | {d.ebm_shape} |")
    add("")
    path = OUT / "report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--band-bootstrap", type=int, default=20)
    parser.add_argument("--ebm-repeats", type=int, default=5, help="repeats of the 5x5 CV used for the EBM bottleneck")
    parser.add_argument("--min-effect", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    TABLES.mkdir(parents=True, exist_ok=True)
    df = load_all()
    train_u = unique_recordings(df[df["split"] == "train"])
    features = {**SIGNALS, **DURATION_SIGNALS, **DIAGNOSTIC_SIGNALS, **ASR_SIGNALS, **SER_SIGNALS, **CATEGORY_SCORES, **PROFILE_INDICES, **flag_signals(df)}
    print(f"Train unique recordings: {len(train_u)} ({train_u.participant.nunique()} participants)")

    print("1. Per-signal checks...")
    corr, cls = signal_checks(train_u, features, args.bootstrap, args.seed, args.min_effect)
    corr.to_csv(TABLES / "signal_checks.csv", index=False)

    print("2. EBM bands...")
    bands, ebm_info = ebm_bands(train_u, args.band_bootstrap, args.seed)
    plot_path = plot_bands(ebm_info["model"], train_u, bands)
    P.S3_BANDS.write_text(json.dumps(
        {"target": "overall_performance", "fitted_on": "train, unique recordings", "tolerance": BAND_TOL,
         "bands": {c: {k: v for k, v in b.items() if k != "curve"} for c, b in bands.items()},
         "curves": {c: b.get("curve") for c, b in bands.items()}}, indent=1), encoding="utf-8")

    decisions = decide(corr, cls, features, bands)
    decisions.to_csv(OUT / "decisions.csv", index=False)
    P.S3_DECISIONS.write_text(json.dumps(
        {r.signal: {"decision": r.decision, "reason": r.reason, "group": r.group} for r in decisions.itertuples()},
        indent=1), encoding="utf-8")

    print("3-5. Bottleneck / faithfulness / attribution (same folds as WavLM)...")
    oof = ensure_oof()
    trainval = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    assert set(oof["id"]) == set(trainval["id"]), "WavLM OOF clips differ from train+val descriptor clips"
    bott, preds, contrib = bottleneck(trainval, oof, args.ebm_repeats, args.seed)
    bott.to_csv(TABLES / "bottleneck.csv", index=False)
    contrib.to_csv(TABLES / "category_contributions.csv", index=False)
    faith = faithfulness(trainval, oof, args.seed)
    faith.to_csv(TABLES / "faithfulness.csv", index=False)

    noise = rater_noise(df, 2000, args.seed)
    noise.to_csv(TABLES / "rater_noise.csv", index=False)
    report = write_report(len(train_u), decisions, corr, bands, ebm_info["importances"], bott, faith, contrib, plot_path, args, noise)
    print(bott.round(3).to_string(index=False))
    print(faith.round(3).to_string(index=False))
    print(decisions[["signal", "decision", "reason"]].to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
