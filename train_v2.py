"""train_v2.py - visual-only training on the TEAM SPLIT (RecruitView) / grouped CV (MIT).

PROTOCOLS
  RecruitView  "split" (default): the team's train/val/test split by user (see splits.py).
      * models are fitted on TRAIN users only (Ridge alpha chosen by an inner CV grouped by user)
      * VAL users choose between feature sets / models
      * TEST users are reported for the chosen model
      * predictions for TRAIN rows are out-of-fold (grouped CV inside train), for VAL/TEST rows they come
        from the model fitted on train. This is the standard stacking layout, so the fusion model can be
        trained on train, tuned on val and evaluated on test, exactly like the text and voice models.
      * predictions are written for ALL 2,011 responses (also the ones the quality filter excludes from
        training, flagged quality_ok=False) so the fusion table lines up row for row.
      "cv" (flag --cv or --cv-only): grouped 5-fold CV over all quality-ok videos, as in the previous version.
  MIT  grouped 5-fold CV by participant (there is no official split).

TARGETS  single ratings + composites (RecruitView composites are z-scored with TRAIN statistics only).
MODELS   Ridge, gradient boosting (HGB), Blend (mean of the two standardised predictions).
FEATURE SETS (what each tests)
  all | robust (no length/face-size-sensitive features) | robust_no_speech_proxy | speech_proxy_only |
  nuisance_only | metadata_only | robust_plus_meta | robust_plus_centered | (MIT) pyfeat_only / precomputed_only
METRICS  rho (Spearman) with 95% CI over people, rho_partial (length-controlled), rho_within, rho_person,
         AUC bottom-third (train-label cutoffs), acc3/bacc3, % of MIT rater ceiling.
         split names in the output: train_oof | val | test (quality-ok videos) | test_all (every test video)

Usage:  python train_v2.py                    everything (RecruitView on the team split, MIT by CV)
        python train_v2.py Impression MIT     only targets whose name contains these words
        python train_v2.py --cv               ALSO run grouped CV over all RecruitView data
        python train_v2.py --cv-only          RecruitView by CV only (ignore the team split)
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

import splits as S
import train as T                     # loaders + config (paths, task dictionaries, quality thresholds)
from diagnostics import icc1          # rater reliability

# =============================== CONFIG ======================================
N_SPLITS = 5
N_REPEATS = 3              # CV repeats (predictions are averaged across repeats)
N_BOOT = 300               # bootstrap resamples (over people) for the 95% CIs
MIN_EVAL = 30              # smallest subset we report metrics for
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

EXTRA_TASKS = {"RecruitView": {},
               "MIT": {"MIT Friendly": "Friendly", "MIT Overall": "Overall",
                       "MIT Recommend Hiring": "RecommendHiring"}}
# composite targets: name -> (new column, component label columns). MIT Composure was dropped: its
# components (Calm vs Not Awkward) respond to the same features in opposite directions and cancel.
COMPOSITES = {
    "RecruitView": {"RV Impression (composite)": ("rv_impression",
                    ["confidence_score", "facial_expression", "extraversion", "overall_performance",
                     "interview_score", "answer_score", "speaking_skills"])},
    "MIT": {"MIT Animation (engagement composite)": ("mit_animation",
                ["EngagingTone", "Excited", "Smiled", "Friendly"])},
}
COMPOSITE_SPECS = {}       # composite column -> component columns (filled by add_composites)
# =============================================================================


# ----------------------------- targets ---------------------------------------
def safe_name(target):
    return "".join(ch if ch.isalnum() else "_" for ch in target.lower()).strip("_")


def zscore(s, ref=None):
    """z-score using the mean/std of the rows selected by `ref` (default: all rows)."""
    s = pd.Series(s, dtype=float)
    r = s if ref is None else s[np.asarray(ref, bool)]
    return (s - r.mean()) / r.std(ddof=0)


def add_composites(df, dataset):
    attrs = dict(df.attrs)
    df = df.copy()
    ref = (df["split"] == "train").to_numpy() if "split" in df.columns else None   # train statistics only
    tasks = {}
    for name, (col, comps) in COMPOSITES[dataset].items():
        comps = [c for c in comps if c in df.columns]
        if len(comps) >= 2:
            df[col] = np.mean([zscore(df[c], ref).to_numpy() for c in comps], axis=0)
            COMPOSITE_SPECS[col] = comps
            tasks[name] = col
    if dataset == "RecruitView" and "facial_expression" in df.columns:
        others = [c for c in ("confidence_score", "extraversion", "speaking_skills", "overall_performance")
                  if c in df.columns]
        ok = df[["facial_expression"] + others].notna().all(axis=1).to_numpy()
        fit = ok & (ref if ref is not None else True)
        Xf = np.c_[np.ones(int(fit.sum())), df.loc[fit, others].to_numpy(float)]
        beta = np.linalg.lstsq(Xf, df.loc[fit, "facial_expression"].to_numpy(float), rcond=None)[0]
        df["rv_facexp_specific"] = np.nan
        df.loc[ok, "rv_facexp_specific"] = (df.loc[ok, "facial_expression"].to_numpy(float)
                                            - np.c_[np.ones(int(ok.sum())), df.loc[ok, others].to_numpy(float)] @ beta)
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
        est.fit(X, y, groups=groups)          # inner grouped CV picks alpha on the given rows only
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


def add_centered(d, cols, ok_mask=None):
    """feature minus the person's mean over their (quality-ok) videos; 0 when a person has one video.
    Persons never straddle splits, so this uses features of the same person only."""
    ok = np.ones(len(d), bool) if ok_mask is None else np.asarray(ok_mask, bool)
    means = d.loc[ok, ["group"] + list(cols)].groupby("group")[list(cols)].mean()
    cen = {f"{c}__c": d[c] - d["group"].map(means[c]) for c in cols}
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


# ----------------------------- metrics ---------------------------------------
def metric_vec(y, pred, cuts=None):
    """[rho, auc_low, auc_high, acc3, bacc3]. `cuts` = (q1, q2) label cutoffs (e.g. from TRAIN labels);
    default = tertiles of y itself. 3-class predictions use tertiles of the predicted score."""
    q1, q2 = cuts if cuts is not None else np.quantile(y, [1 / 3, 2 / 3])
    ytrue = np.digitize(y, [q1, q2], right=True)
    yhat = np.digitize(pred, np.quantile(pred, [1 / 3, 2 / 3]), right=True)
    try:
        auc_low = roc_auc_score((y <= q1).astype(int), -pred)
        auc_high = roc_auc_score((y > q2).astype(int), pred)
    except ValueError:
        auc_low = auc_high = np.nan
    return np.array([spearmanr(y, pred)[0], auc_low, auc_high,
                     accuracy_score(ytrue, yhat), balanced_accuracy_score(ytrue, yhat)])


def cluster_bootstrap(y, pred, groups, rng, cuts=None):
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    stats = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(ug, size=len(ug), replace=True)])
        stats.append(metric_vec(y[idx], pred[idx], cuts))
    return np.nanpercentile(np.array(stats), [2.5, 97.5], axis=0)


def partial_spearman(y, pred, Z):
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


def make_row(base, y, pred, groups, Z, cuts, rng, dataset, ceiling):
    mv = metric_vec(y, pred, cuts)
    ci = cluster_bootstrap(y, pred, groups, rng, cuts)
    row = dict(base)
    row.update({
        "n": len(y), "rho": mv[0], "rho_lo": ci[0, 0], "rho_hi": ci[1, 0],
        "rho_partial": partial_spearman(y, pred, Z),
        "rho_within": within_rho(y, pred, groups),
        "rho_person": person_rho(y, pred, groups) if dataset == "RecruitView" else np.nan,
        "ceiling_r": ceiling,
        "pct_of_ceiling": mv[0] / ceiling if np.isfinite(ceiling) and ceiling > 0 else np.nan,
        "pearson_r": pearsonr(y, pred)[0], "r2": r2_score(y, pred),
        "auc_low": mv[1], "auc_low_lo": ci[0, 1], "auc_low_hi": ci[1, 1],
        "auc_high": mv[2], "acc3": mv[3], "bacc3": mv[4],
    })
    return row


# ----------------------------- shared preparation ----------------------------
def prepare(df, dataset, label_col, ok_for_features):
    """Common set-up. `ok_for_features` = rows used to decide which feature columns are usable."""
    attrs = dict(df.attrs)
    ref = df[ok_for_features].copy()
    ref.attrs = attrs
    allf = T.feature_columns(ref, label_col)
    d = df.dropna(subset=[label_col]).reset_index(drop=True)
    d.attrs = attrs
    ok = d["quality_ok"].to_numpy(bool) if "quality_ok" in d.columns else np.ones(len(d), bool)
    d, meta_cols = add_metadata(d, dataset)
    nuis = [c for c in NUISANCE[dataset] if c in d.columns]
    d, cen_cols = add_centered(d, robust_filter(dataset, allf), ok)
    sets = build_sets(allf, meta_cols, nuis, cen_cols, dataset)
    cov = [c for c in PARTIAL_COVARS[dataset] if c in d.columns]
    Z = d[cov].to_numpy(float) if cov else None
    return d, sets, ok, Z


# ----------------------------- protocol 1: grouped CV ------------------------
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


def run_target_cv(df, dataset, target, label_col):
    if "quality_ok" in df.columns:
        attrs = dict(df.attrs)
        df = df[df["quality_ok"]].copy()
        df.attrs = attrs
    d, sets, ok, Z = prepare(df, dataset, label_col, np.ones(len(df), bool))
    y, groups = d[label_col].to_numpy(float), d["group"].to_numpy()
    ceiling = label_ceiling(label_col) if dataset == "MIT" else np.nan
    rng = np.random.default_rng(0)
    rows, oof_cols = [], {}
    print(f"\n##### {target} ({dataset}, grouped CV, n={len(d)}, people={len(np.unique(groups))}"
          + (f", rater ceiling r={ceiling:.2f}" if np.isfinite(ceiling) else "") + ")")
    print("  feature sets:", {k: len(v) for k, v in sets.items()})

    for set_name, feats in sets.items():
        X = d[feats].to_numpy(float)
        models = ("Ridge", "HGB") if set_name in HGB_SETS else ("Ridge",)
        preds, coef_store = {m: [] for m in models}, []
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
            base = {"dataset": dataset, "target": target, "protocol": "cv", "split": "all",
                    "feature_set": set_name, "model": m, "n_features": len(feats),
                    "ridge_top_features": top_ridge_features(coef_store, feats) if m == "Ridge" else ""}
            rows.append(make_row(base, y, pred, groups, Z, None, rng, dataset, ceiling))
        if set_name == "robust" and dataset == "MIT":
            save_final(feats, d[feats].to_numpy(float), y, groups, models, target, "all")

    OUT_DIR.mkdir(exist_ok=True)
    keep = [c for c in ("id", "video_id", "file_name", "participant", "user_no", "phase") if c in d.columns]
    oof_df = d[keep].copy()
    oof_df["group"], oof_df["y_true"] = groups, y
    for k, v in oof_cols.items():
        oof_df[f"pred_{k}"] = v
    oof_df.to_csv(OUT_DIR / f"oof_cv_{safe_name(target)}.csv", index=False)
    res = pd.DataFrame(rows)
    print_table(res, ["all"])
    readout(res, "all")
    return rows


# ----------------------------- protocol 2: team split ------------------------
def run_target_split(df, dataset, target, label_col, meta_all, mapping):
    ok_train = ((df["split"] == "train") & df["quality_ok"]).to_numpy()
    d, sets, ok, Z = prepare(df, dataset, label_col, ok_train)
    y, groups = d[label_col].to_numpy(float), d["group"].to_numpy()
    sp = d["split"].to_numpy()
    fit, val = (sp == "train") & ok, (sp == "val") & ok
    test, test_all = (sp == "test") & ok, sp == "test"
    if fit.sum() < 100 or test.sum() < MIN_EVAL:
        print(f"[skip] {target}: too few rows (train {fit.sum()}, test {test.sum()})")
        return []
    cuts = tuple(np.quantile(y[fit], [1 / 3, 2 / 3]))          # label cutoffs from TRAIN only
    rng = np.random.default_rng(0)
    rows, pred_cols = [], {}
    print(f"\n##### {target} ({dataset}, team split: train {fit.sum()} / val {val.sum()} / "
          f"test {test.sum()} (+{int((test_all & ~ok).sum())} test clips below the quality threshold))")
    print("  feature sets:", {k: len(v) for k, v in sets.items()})
    subsets = (("train_oof", fit), ("val", val), ("test", test), ("test_all", test_all))

    for set_name, feats in sets.items():
        X = d[feats].to_numpy(float)
        models = ("Ridge", "HGB") if set_name in HGB_SETS else ("Ridge",)
        oof_reps, coef_store = {m: [] for m in models}, []
        for seed in range(N_REPEATS):                          # OOF inside TRAIN only
            oof, coefs = oof_predict(X[fit], y[fit], groups[fit], seed, models)
            for m in models:
                oof_reps[m].append(oof[m])
            coef_store += coefs
        oof_avg = {m: np.vstack(oof_reps[m]).mean(0) for m in models}
        full = {}
        for m in models:                                       # one fit on all TRAIN rows -> predict everything
            full[m] = fit_model(m, X[fit], y[fit], groups[fit]).predict(X)
        if len(models) == 2:                                   # Blend standardised with TRAIN OOF statistics
            z = lambda m, v: (v - oof_avg[m].mean()) / oof_avg[m].std()
            oof_avg["Blend"] = (z("Ridge", oof_avg["Ridge"]) + z("HGB", oof_avg["HGB"])) / 2
            full["Blend"] = (z("Ridge", full["Ridge"]) + z("HGB", full["HGB"])) / 2
        for m in oof_avg:
            pred = full[m].copy()
            pred[fit] = oof_avg[m]                             # train rows: out-of-fold
            pred_cols[f"{set_name}__{m}"] = pred
            for split_name, mask in subsets:
                if mask.sum() < MIN_EVAL:
                    continue
                base = {"dataset": dataset, "target": target, "protocol": "split", "split": split_name,
                        "feature_set": set_name, "model": m, "n_features": len(feats),
                        "ridge_top_features": top_ridge_features(coef_store, feats) if m == "Ridge" else ""}
                rows.append(make_row(base, y[mask], pred[mask], groups[mask],
                                     Z[mask] if Z is not None else None, cuts, rng, dataset, np.nan))
        if set_name == "robust":
            trval = ((sp == "train") | (sp == "val")) & ok
            save_final(feats, X[trval], y[trval], groups[trval], models, target, "trainval")

    write_split_predictions(d, y, groups, sp, ok, pred_cols, target, meta_all, mapping)
    res = pd.DataFrame(rows)
    print_table(res, ["val", "test", "test_all"])
    readout(res, "val")
    readout(res, "test")
    return rows


def write_split_predictions(d, y, groups, sp, ok, pred_cols, target, meta_all, mapping):
    """One row per response in the metadata (all 2,011), aligned for fusion with the other modalities."""
    OUT_DIR.mkdir(exist_ok=True)
    keep = [c for c in ("id", "video_id") if c in d.columns]
    out = d[keep].copy()
    out["y_true"], out["quality_ok"] = y, ok
    for k, v in pred_cols.items():
        out[f"pred_{k}"] = v
    full = meta_all[["id", "file_name", "user_no"]].merge(out, on="id", how="left")
    full["split"] = full["user_no"].map(mapping)
    full["has_features"] = full["quality_ok"].notna()
    full["quality_ok"] = full["quality_ok"].fillna(False).astype(bool)
    front = ["id", "file_name", "user_no", "split", "has_features", "quality_ok", "y_true"]
    full = full[front + [c for c in full.columns if c not in front]]
    full.to_csv(OUT_DIR / f"preds_split_{safe_name(target)}.csv", index=False)
    n_missing = int((~full.has_features).sum())
    if n_missing:
        print(f"  note: {n_missing} of {len(full)} responses have no features/label for this target "
              f"(predictions left empty)")


def save_final(feats, X, y, groups, models, target, tag):
    MODEL_DIR.mkdir(exist_ok=True)
    for m in models:
        est = fit_model(m, X, y, groups)
        joblib.dump({"features": feats, "model": est, "y_mean": y.mean(), "y_std": y.std(), "fit_on": tag},
                    MODEL_DIR / f"{safe_name(target)}__{m}__{tag}.joblib")


# ----------------------------- reporting -------------------------------------
def print_table(res, split_names):
    cols = ["feature_set", "model", "n_features"]
    out = res[cols].drop_duplicates().reset_index(drop=True)
    for sname in split_names:
        r = res[res.split == sname].set_index(["feature_set", "model"])
        out[f"rho_{sname}"] = [r.rho.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
        if sname in ("test", "all"):
            out["ci_lo"] = [r.rho_lo.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
            out["ci_hi"] = [r.rho_hi.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
            out["partial"] = [r.rho_partial.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
            out["within"] = [r.rho_within.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
            out["auc_low"] = [r.auc_low.get((a, b), np.nan) for a, b in zip(out.feature_set, out.model)]
    print(out.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))


def readout(res, split_name):
    r0 = res[(res.split == split_name) & (res.model == "Ridge")].set_index("feature_set")
    if r0.empty:
        return
    g = r0.rho
    order = ["all", "robust", "robust_no_speech_proxy", "speech_proxy_only", "nuisance_only",
             "metadata_only", "robust_plus_meta", "robust_plus_centered", "pyfeat_only", "precomputed_only"]
    print(f"  READ-OUT [{split_name}] (Ridge rho): " + " | ".join(f"{k}={g[k]:.3f}" for k in order if k in g))
    if "metadata_only" in g and "robust_plus_meta" in g:
        print(f"    visual features add beyond metadata: {g['robust_plus_meta'] - g['metadata_only']:+.3f} rho")
    if "robust" in g and "robust_plus_centered" in g:
        print(f"    person-centred features change rho by {g['robust_plus_centered'] - g['robust']:+.3f}")
    if "speech_proxy_only" in g and "robust" in g:
        if g["robust"] > 0.05 and g["speech_proxy_only"] > 0:
            print(f"    mouth-opening features alone reach {g['speech_proxy_only'] / g['robust']:.0%} of the robust rho")
        else:
            print("    mouth-opening features alone carry no usable signal here")
    if "robust" in r0.index:
        r = r0.loc["robust"]
        print(f"    robust Ridge: rho={r.rho:.3f}, length-controlled rho={r.rho_partial:.3f}, "
              f"within-person rho={r.rho_within:.3f}")


def summarize(out):
    sp = out[out.protocol == "split"]
    if not sp.empty:
        val = sp[(sp.split == "val") & (sp.feature_set == "robust")]
        pick = val.sort_values("rho", ascending=False).groupby("target").head(1)[["target", "model", "rho"]]
        pick = pick.rename(columns={"rho": "rho_val"})
        t = sp[(sp.split == "test") & (sp.feature_set == "robust")]
        t = t[["target", "model", "rho", "rho_lo", "rho_hi", "rho_partial", "rho_within", "auc_low"]]
        ta = sp[(sp.split == "test_all") & (sp.feature_set == "robust")][["target", "model", "rho"]]
        ta = ta.rename(columns={"rho": "rho_test_all"})
        tab = pick.merge(t, on=["target", "model"]).merge(ta, on=["target", "model"], how="left")
        tab = tab.rename(columns={"rho": "rho_test"}).sort_values("rho_test", ascending=False)
        print("\nRecruitView, team split. Model chosen on VAL (robust features), reported on TEST:")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:6.3f}"))
    cv = out[(out.protocol == "cv") & (out.feature_set == "robust")]
    if not cv.empty:
        best = (cv.sort_values("rho", ascending=False).groupby("target").head(1)
                [["target", "model", "rho", "rho_lo", "rho_hi", "rho_partial", "rho_within",
                  "pct_of_ceiling", "auc_low"]])
        print("\nGrouped CV (MIT, or RecruitView with --cv). Best `robust` model per target "
              "(best-of-models, slightly optimistic):")
        print(best.sort_values("rho", ascending=False).to_string(index=False, float_format=lambda v: f"{v:6.3f}"))


def main():
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    filt = [a.lower() for a in args if not a.startswith("--")]
    rv_cv = "--cv" in flags or "--cv-only" in flags
    rv_split = "--cv-only" not in flags

    loaders = []
    if T.RECRUITVIEW_FEATURES.exists() and T.RECRUITVIEW_META.exists():
        rv = T.load_recruitview(quality_filter=False)          # keep every video; quality_ok marks the rest
        meta_all = pd.read_json(T.RECRUITVIEW_META, lines=True)
        mapping = None
        if rv_split:
            try:
                mapping = S.load_split()
                S.validate(mapping, meta_all)
                rv = S.attach_split(rv, mapping)
            except FileNotFoundError as e:
                print(f"WARNING: {e}\n  -> falling back to grouped CV for RecruitView")
                rv_split, rv_cv = False, True
        loaders.append(("RecruitView", rv))
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
            if dataset == "RecruitView":
                if rv_split:
                    all_rows += run_target_split(df, dataset, target, col, meta_all, mapping)
                if rv_cv:
                    all_rows += run_target_cv(df, dataset, target, col)
            else:
                all_rows += run_target_cv(df, dataset, target, col)

    OUT_DIR.mkdir(exist_ok=True)
    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_DIR / "results_v2.csv", index=False)
    print(f"\nSaved {OUT_DIR / 'results_v2.csv'}; fusion tables: {OUT_DIR}/preds_split_*.csv; "
          f"models: {MODEL_DIR}/")
    summarize(out)


if __name__ == "__main__":
    main()
