"""train_v2.py - visual-only training, round 3 (built from diagnostics.txt).

What the diagnostics showed, and what this script does about it
  * RecruitView labels are one general "impression" factor (rho 0.65-0.83 between them).
      -> composite target `RV Impression` (mean z-score; less noisy than any single label) and a
         `FacialExpression-specific` residual (what is left after removing the shared factor),
         to test whether the labels are separable at all.
  * Video length alone correlates -0.35 with the labels, as high as the best video model.
      -> every row reports a PARTIAL rho (controls for length; MIT: length + pre/post phase),
         and the ablation sets (nuisance_only / metadata_only / robust_plus_meta) stay.
  * Only ~27% of RecruitView label variance is between people; ~73% is within a person.
      -> `rho_within` (is the model right about which of a person's answers was better?),
         `rho_person` (are person averages right?) and `robust_plus_centered` features
         (each feature minus the person's mean; needs >=2 videos per person at use time).
  * MIT raters agree well on EngagingTone/Excited/Smiled/Friendly, poorly on Calm/NotStressed.
      -> `ceiling_r` and `pct_of_ceiling` per target; new targets Friendly, Overall,
         RecommendHiring; composites `MIT Animation` (engagement) and `MIT Composure`
         (calm / not awkward / not stressed / eye contact: the confidence-hesitation side).
  * Cross-sectionally Eye Contact is unpredictable (MIT has no gaze), but WITHIN a person head
      pitch tracked the pre/post change in eye contact.
      -> centred features, plus optional Py-Feat features for MIT (extract_features_mit_pyfeat.py)
         which add gaze and put both datasets in the same feature space.

Feature sets (Ridge on all; gradient boosting + Blend on the main ones)
  all | robust (drops length- and face-size-sensitive features) | robust_no_speech_proxy |
  speech_proxy_only | nuisance_only | metadata_only | robust_plus_meta | robust_plus_centered
  (+ MIT with Py-Feat: pyfeat_only | precomputed_only)

Usage:  python train_v2.py                 # everything
        python train_v2.py MIT Impression  # only targets whose name contains these words
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, r2_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import train as T                     # loaders + config (paths, task dictionaries, quality filter)
from diagnostics import icc1          # rater reliability

# =============================== CONFIG ======================================
N_SPLITS = 5
N_REPEATS = 3              # CV repeats; predictions are averaged across repeats
N_BOOT = 300               # bootstrap resamples (over people) for the 95% CIs
OUT_DIR = Path("results_v2")
MODEL_DIR = Path("models_v2")
PYFEAT_MIT = Path("features_mit_pyfeat.csv")
HGB_SETS = {"robust", "robust_plus_meta", "robust_plus_centered"}

LENGTH_SENSITIVE_SUFFIXES = ("_max", "_min", "_active_segments")
NUISANCE = {"RecruitView": ["duration_sec", "n_samples", "face_rate"], "MIT": ["n_frames"]}
PARTIAL_COVARS = {"RecruitView": ["duration_sec"], "MIT": ["n_frames", "meta__is_post"]}
SPEECH_PREFIXES = {"RecruitView": ("AU22", "AU25", "AU26", "AU27"),
                   "MIT": ("oLipH", "iLipH", "mouth_open_ratio", "AU22", "AU25", "AU26", "AU27")}
MIT_RAW_CHANNELS = ("inBrL", "otBrL", "inBrR", "otBrR", "EyeOL", "EyeOR",
                    "oLipH", "iLipH", "LipCDt", "brow_height")
MIT_RAW_SUFFIXES = ("_mean", "_std", "_slope", "_motion")

# extra single-label targets (rater ICC > 0.7 in diagnostics)
EXTRA_TASKS = {"RecruitView": {},
               "MIT": {"MIT Friendly": "Friendly", "MIT Overall": "Overall",
                       "MIT Recommend Hiring": "RecommendHiring"}}
# composite targets: name -> (new column, component label columns), averaged as z-scores
COMPOSITES = {
    "RecruitView": {"RV Impression (composite)": ("rv_impression",
                    ["confidence_score", "facial_expression", "extraversion", "overall_performance",
                     "interview_score", "answer_score", "speaking_skills"])},
    "MIT": {"MIT Animation (engagement composite)": ("mit_animation",
                ["EngagingTone", "Excited", "Smiled", "Friendly"]),
            "MIT Composure (confidence / low-hesitation composite)": ("mit_composure",
                ["Calm", "NotAwkward", "NotStressed", "EyeContact"])},
}
COMPOSITE_SPECS = {}       # filled by add_composites: composite column -> component columns
# =============================================================================


# ----------------------------- targets ---------------------------------------
def safe_name(target):
    return "".join(ch if ch.isalnum() else "_" for ch in target.lower()).strip("_")


def zscore(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / s.std(ddof=0)


def add_composites(df, dataset):
    attrs = dict(df.attrs)
    df = df.copy()
    tasks = {}
    for name, (col, comps) in COMPOSITES[dataset].items():
        comps = [c for c in comps if c in df.columns]
        if len(comps) >= 2:
            df[col] = np.mean([zscore(df[c]).to_numpy() for c in comps], axis=0)
            COMPOSITE_SPECS[col] = comps
            tasks[name] = col
    if dataset == "RecruitView" and "facial_expression" in df.columns:
        others = [c for c in ("confidence_score", "extraversion", "speaking_skills", "overall_performance")
                  if c in df.columns]
        ok = df[["facial_expression"] + others].notna().all(axis=1)
        X = np.c_[np.ones(ok.sum()), df.loc[ok, others].to_numpy(float)]
        beta = np.linalg.lstsq(X, df.loc[ok, "facial_expression"].to_numpy(float), rcond=None)[0]
        df["rv_facexp_specific"] = np.nan
        df.loc[ok, "rv_facexp_specific"] = df.loc[ok, "facial_expression"].to_numpy(float) - X @ beta
        tasks["RV FacialExpression-specific (residual)"] = "rv_facexp_specific"
    df.attrs = attrs
    return df, tasks


def get_tasks(dataset, df):
    tasks = dict(T.RV_TASKS if dataset == "RecruitView" else T.MIT_TASKS)
    tasks.update(EXTRA_TASKS[dataset])
    df, comp = add_composites(df, dataset)
    tasks.update(comp)
    return df, tasks


_RATERS = None


def label_ceiling(label_col):
    """sqrt of the reliability of the rater-averaged label = rough max achievable correlation."""
    global _RATERS
    if _RATERS is None and T.MIT_SCORES.exists():
        sc = pd.read_csv(T.MIT_SCORES)
        r = sc[sc[T.MIT_WORKER_COL].astype(str).str.upper() != "AGGR"].copy()
        r["vid"] = r[T.MIT_ID_COL].astype(str).str.lower()
        _RATERS = r
    if _RATERS is None:
        return np.nan
    cols = [c for c in COMPOSITE_SPECS.get(label_col, [label_col]) if c in _RATERS.columns]
    if not cols:
        return np.nan
    v = np.mean([zscore(_RATERS[c]).to_numpy() for c in cols], axis=0)
    _, mean_k, _ = icc1(v, _RATERS["vid"])
    return float(np.sqrt(max(mean_k, 0))) if np.isfinite(mean_k) else np.nan


def load_mit_all():
    m = T.load_mit()
    if PYFEAT_MIT.exists():
        pf = pd.read_csv(PYFEAT_MIT)
        pf["video_id"] = pf["video_id"].astype(str).str.lower()
        keep_nuis = [c for c in ("pf_duration_sec", "pf_face_rate") if c in pf.columns]
        pfc = [c for c in pf.columns if c.startswith("pf_") and c not in keep_nuis]
        attrs = dict(m.attrs)
        m = m.merge(pf[["video_id"] + pfc + keep_nuis], on="video_id", how="left")
        attrs["feature_cols"] = attrs["feature_cols"] + pfc
        m.attrs = attrs
        print(f"  merged {len(pfc)} Py-Feat features for {pf.video_id.nunique()} MIT videos")
    return m


# ----------------------------- models ----------------------------------------
def make_model(name):
    if name == "Ridge":
        pipe = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Ridge())
        return GridSearchCV(pipe, {"ridge__alpha": np.logspace(0, 4, 9)},
                            cv=GroupKFold(n_splits=3), scoring="neg_mean_squared_error")
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=150, max_depth=3,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=0)


def fit_model(name, X, y, groups):
    est = make_model(name)
    if name == "Ridge":
        est.fit(X, y, groups=groups)
    else:
        est.fit(X, y)
    return est


# ----------------------------- feature sets ----------------------------------
def robust_filter(dataset, cols):
    out = [c for c in cols if not c.endswith(LENGTH_SENSITIVE_SUFFIXES)]
    if dataset == "MIT":
        raw = {ch + s for ch in MIT_RAW_CHANNELS for s in MIT_RAW_SUFFIXES}
        out = [c for c in out if c not in raw]
    return out


def add_metadata(d, dataset):
    d = d.copy()
    meta = [c for c in NUISANCE[dataset] if c in d.columns and pd.api.types.is_numeric_dtype(d[c])]
    extra = {}
    if dataset == "RecruitView":
        for col in ("question_id", "duration", "video_quality"):
            if col in d.columns:
                dm = pd.get_dummies(d[col].astype(str), prefix=f"meta__{col}", dtype=float)
                extra.update({c: dm[c] for c in dm.columns})
    elif "phase" in d.columns:
        extra["meta__is_post"] = (d["phase"].astype(str).str.lower() == "post").astype(float)
    if extra:
        d = pd.concat([d, pd.DataFrame(extra, index=d.index)], axis=1)
    return d, meta + list(extra)


def add_centered(d, cols):
    """feature minus the person's mean over their videos (0 when a person has one video)."""
    g = d.groupby("group")
    cen = {f"{c}__c": d[c] - g[c].transform("mean") for c in cols}
    return pd.concat([d, pd.DataFrame(cen, index=d.index)], axis=1), list(cen)


def is_speech(dataset, c):
    base = c[3:] if c.startswith("pf_") else c
    return base.startswith(SPEECH_PREFIXES[dataset])


def build_sets(allf, meta_cols, nuis_cols, cen_cols, dataset):
    robust = robust_filter(dataset, allf)
    sets = {
        "all": allf,
        "robust": robust,
        "robust_no_speech_proxy": [c for c in robust if not is_speech(dataset, c)],
        "speech_proxy_only": [c for c in robust if is_speech(dataset, c)],
        "nuisance_only": nuis_cols,
        "metadata_only": meta_cols,
        "robust_plus_meta": robust + meta_cols,
        "robust_plus_centered": robust + cen_cols,
    }
    pf = [c for c in robust if c.startswith("pf_")]
    if pf and len(pf) < len(robust):
        sets["pyfeat_only"] = pf
        sets["precomputed_only"] = [c for c in robust if not c.startswith("pf_")]
    if len(robust) == len(allf):
        print("  (no length/scale-sensitive features were dropped - re-run the extractors)")
    return {k: v for k, v in sets.items() if v}


# ----------------------------- CV + metrics ----------------------------------
def oof_predict(X, y, groups, seed, models):
    strat = pd.qcut(y, 3, labels=False, duplicates="drop")
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof = {m: np.zeros(len(y)) for m in models}
    coefs = []
    for tr, te in cv.split(X, strat, groups):
        for m in models:
            est = fit_model(m, X[tr], y[tr], groups[tr])
            oof[m][te] = est.predict(X[te])
            if m == "Ridge":
                coefs.append(est.best_estimator_.named_steps["ridge"].coef_.copy())
    return oof, coefs


def metric_vec(y, pred):
    """[rho, auc_low, auc_high, acc3, bacc3]"""
    q1, q2 = np.quantile(y, [1 / 3, 2 / 3])
    ytrue = np.digitize(y, [q1, q2], right=True)
    yhat = np.digitize(pred, np.quantile(pred, [1 / 3, 2 / 3]), right=True)
    try:
        auc_low = roc_auc_score((y <= q1).astype(int), -pred)
        auc_high = roc_auc_score((y > q2).astype(int), pred)
    except ValueError:
        auc_low = auc_high = np.nan
    return np.array([spearmanr(y, pred)[0], auc_low, auc_high,
                     accuracy_score(ytrue, yhat), balanced_accuracy_score(ytrue, yhat)])


def cluster_bootstrap(y, pred, groups, rng):
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        stats.append(metric_vec(y[idx], pred[idx]))
    return np.nanpercentile(np.array(stats), [2.5, 97.5], axis=0)


def partial_spearman(y, pred, Z):
    """Spearman correlation of y and pred after removing the (rank-linear) effect of covariates Z."""
    if Z is None or Z.shape[1] == 0:
        return np.nan
    A = np.c_[np.ones(len(y)), np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])])]

    def resid(v):
        return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    return float(np.corrcoef(resid(rankdata(y)), resid(rankdata(pred)))[0, 1])


def within_rho(y, pred, groups):
    s = pd.DataFrame({"y": y, "p": pred, "g": groups})
    s = s[s.groupby("g").y.transform("size") >= 2]
    if len(s) < 10:
        return np.nan
    yc = s.y - s.groupby("g").y.transform("mean")
    pc = s.p - s.groupby("g").p.transform("mean")
    return float(spearmanr(yc, pc)[0])


def person_rho(y, pred, groups):
    a = pd.DataFrame({"y": y, "p": pred, "g": groups}).groupby("g").agg(
        y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    a = a[a.n >= 2]
    return float(spearmanr(a.y, a.p)[0]) if len(a) > 10 else np.nan


def top_ridge_features(coefs, feats, k=8):
    C = np.vstack(coefs)
    mean, cons = C.mean(0), (np.sign(C) == np.sign(C.mean(0))).mean(0)
    order = np.argsort(-np.abs(mean))[:k]
    return ", ".join(f"{feats[i]}({mean[i]:+.2f},{cons[i]:.0%})" for i in order)


# ----------------------------- one target ------------------------------------
def run_target(df, dataset, target, label_col):
    allf = T.feature_columns(df, label_col)
    d = df.dropna(subset=[label_col]).reset_index(drop=True)
    d, meta_cols = add_metadata(d, dataset)
    nuis = [c for c in NUISANCE[dataset] if c in d.columns]
    d, cen_cols = add_centered(d, robust_filter(dataset, allf))
    sets = build_sets(allf, meta_cols, nuis, cen_cols, dataset)
    y = d[label_col].to_numpy(float)
    groups = d["group"].to_numpy()
    cov = [c for c in PARTIAL_COVARS[dataset] if c in d.columns]
    Z = d[cov].to_numpy(float) if cov else None
    ceiling = label_ceiling(label_col) if dataset == "MIT" else np.nan
    rng = np.random.default_rng(0)
    rows, oof_cols = [], {}
    print(f"\n##### {target} ({dataset}, n={len(d)}, people={len(np.unique(groups))}"
          + (f", rater ceiling r={ceiling:.2f}" if np.isfinite(ceiling) else "") + ")")
    print("  feature sets:", {k: len(v) for k, v in sets.items()})

    for set_name, feats in sets.items():
        X = d[feats].to_numpy(float)
        models = ("Ridge", "HGB") if set_name in HGB_SETS else ("Ridge",)
        preds = {m: [] for m in models}
        coef_store = []
        for seed in range(N_REPEATS):
            oof, coefs = oof_predict(X, y, groups, seed, models)
            for m in models:
                preds[m].append(oof[m])
            coef_store += coefs
        avg = {m: np.vstack(preds[m]).mean(0) for m in models}
        if len(models) == 2:
            avg["Blend"] = (zscore(avg["Ridge"]).to_numpy() + zscore(avg["HGB"]).to_numpy()) / 2
        for m, pred in avg.items():
            oof_cols[f"{set_name}__{m}"] = pred
            mv = metric_vec(y, pred)
            ci = cluster_bootstrap(y, pred, groups, rng)
            rep_std = (float(np.std([spearmanr(y, p)[0] for p in np.vstack(preds[m])]))
                       if m in preds else np.nan)
            rows.append({
                "dataset": dataset, "target": target, "feature_set": set_name, "model": m,
                "n": len(y), "n_features": len(feats),
                "rho": mv[0], "rho_lo": ci[0, 0], "rho_hi": ci[1, 0], "rho_repeat_std": rep_std,
                "rho_partial": partial_spearman(y, pred, Z),
                "rho_within": within_rho(y, pred, groups),
                "rho_person": person_rho(y, pred, groups) if dataset == "RecruitView" else np.nan,
                "ceiling_r": ceiling, "pct_of_ceiling": mv[0] / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan,
                "pearson_r": pearsonr(y, pred)[0], "r2": r2_score(y, pred),
                "auc_low": mv[1], "auc_low_lo": ci[0, 1], "auc_low_hi": ci[1, 1],
                "auc_high": mv[2], "acc3": mv[3], "bacc3": mv[4],
                "ridge_top_features": top_ridge_features(coef_store, feats) if m == "Ridge" else "",
            })
        if set_name == "robust":                       # deployable, video-only final models
            MODEL_DIR.mkdir(exist_ok=True)
            for m in models:
                est = fit_model(m, X, y, groups)
                joblib.dump({"features": feats, "model": est, "y_mean": y.mean(), "y_std": y.std()},
                            MODEL_DIR / f"{safe_name(target)}__{m}.joblib")

    OUT_DIR.mkdir(exist_ok=True)
    keep = [c for c in ("video_id", "file_name", "participant", "user_no", "phase") if c in d.columns]
    oof_df = d[keep].copy()
    oof_df["group"], oof_df["y_true"] = groups, y
    for k, v in oof_cols.items():
        oof_df[f"pred_{k}"] = v
    oof_df.to_csv(OUT_DIR / f"oof_{safe_name(target)}.csv", index=False)

    res = pd.DataFrame(rows)
    print(res[["feature_set", "model", "n_features", "rho", "rho_lo", "rho_hi", "rho_partial",
               "rho_within", "rho_person", "auc_low", "bacc3"]]
          .to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    readout(res)
    return rows


def readout(res):
    g = res[res.model == "Ridge"].set_index("feature_set").rho
    order = ["all", "robust", "robust_no_speech_proxy", "speech_proxy_only", "nuisance_only",
             "metadata_only", "robust_plus_meta", "robust_plus_centered", "pyfeat_only", "precomputed_only"]
    print("  READ-OUT (Ridge rho): " + " | ".join(f"{k}={g[k]:.3f}" for k in order if k in g))
    if "metadata_only" in g and "robust_plus_meta" in g:
        print(f"  visual features add beyond metadata: {g['robust_plus_meta'] - g['metadata_only']:+.3f} rho")
    if "robust" in g and "robust_plus_centered" in g:
        print(f"  person-centred features change rho by {g['robust_plus_centered'] - g['robust']:+.3f}")
    if "speech_proxy_only" in g and "robust" in g and g["robust"] > 0:
        print(f"  mouth-opening features alone reach {g['speech_proxy_only'] / g['robust']:.0%} of the robust model's rho")
    b = res[(res.feature_set == "robust") & (res.model == "Ridge")]
    if len(b):
        r = b.iloc[0]
        print(f"  robust Ridge: rho={r.rho:.3f}, partial rho (length-controlled)={r.rho_partial:.3f}, "
              f"within-person rho={r.rho_within:.3f}")
    r = res[(res.model == "Ridge") & (res.feature_set == "robust_no_speech_proxy")]
    if len(r):
        print("  Ridge top features without the speech proxy (mean std. coef, sign consistency):")
        print("   ", r.iloc[0].ridge_top_features)


def main():
    filt = [a.lower() for a in sys.argv[1:]]
    loaders = []
    if T.RECRUITVIEW_FEATURES.exists() and T.RECRUITVIEW_META.exists():
        loaders.append(("RecruitView", T.load_recruitview()))
    if T.MIT_FEATURES.exists():
        loaders.append(("MIT", load_mit_all()))
    if not loaders:
        raise SystemExit("No feature files found. Run the extractors first.")

    all_rows = []
    for dataset, df in loaders:
        df, tasks = get_tasks(dataset, df)
        for target, col in tasks.items():
            if filt and not any(f in target.lower() for f in filt):
                continue
            if col not in df.columns:
                print(f"[skip] {target}: column '{col}' not found")
                continue
            all_rows += run_target(df, dataset, target, col)

    OUT_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "results_v2.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'results_v2.csv'}, per-target OOF predictions in {OUT_DIR}/, "
          f"models in {MODEL_DIR}/")
    best = (out[out.feature_set == "robust"].sort_values("rho", ascending=False)
            .groupby("target").head(1)[["target", "model", "rho", "rho_lo", "rho_hi", "rho_partial",
                                        "rho_within", "pct_of_ceiling", "auc_low"]])
    print("\nBest `robust` (video-only) model per target:")
    print(best.sort_values("rho", ascending=False).to_string(index=False, float_format=lambda v: f"{v:6.3f}"))


if __name__ == "__main__":
    main()
