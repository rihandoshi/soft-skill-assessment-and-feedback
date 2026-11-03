"""Compare Ridge and an RBF support-vector regressor (SVR) on interpretable feature sets.

Feature sets:
  linguistic           text features from train_text_baselines.LING_COLS
  delivery             verbatim-ASR fillers/disfluency + acoustic syllable/articulation rate
                       (outputs/text/delivery/delivery_features.parquet; skipped if missing)
  linguistic_delivery  both

Hyper-parameters are chosen per target by participant-grouped 5-fold CV on train (CCC),
exactly like train_text_baselines.py. Validation metrics only unless --eval-test.

Writes outputs/text/predictions/<featureset>_<learner>.parquet (train rows out-of-fold) and
outputs/text/reports/feature_models_<split>.csv. Run summarize_results.py afterwards for the
combined table.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from text_common import OUT_DIR, SEED, TARGETS, ccc, cluster_bootstrap_ci, regression_metrics, write_json
from train_text_baselines import ALPHAS, LING_COLS, N_FOLDS, clip_bounds

DELIVERY_PATH = OUT_DIR / "delivery" / "delivery_features.parquet"

GRIDS = {
    "ridge": [{"alpha": a} for a in ALPHAS],
    "svr": [{"C": c, "epsilon": e, "gamma": g}
            for c, e, g in itertools.product([0.1, 0.3, 1.0, 3.0, 10.0], [0.1, 0.3],
                                             ["scale", 0.01, 0.03])],
}


def make_model(learner: str, params: dict):
    est = Ridge(**params) if learner == "ridge" else SVR(kernel="rbf", **params)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)


def fit_predict(learner, params, X_tr, y_tr, X_te):
    lo, hi = clip_bounds(y_tr, True)
    return make_model(learner, params).fit(X_tr, np.clip(y_tr, lo, hi)).predict(X_te)


def select_params(learner, X, y, folds):
    best = (-np.inf, None)
    for params in GRIDS[learner]:
        s = float(np.mean([ccc(y[va], fit_predict(learner, params, X[tr], y[tr], X[va]))
                           for tr, va in folds]))
        if s > best[0]:
            best = (s, params)
    return best[1], best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="+", default=["ridge", "svr"], choices=list(GRIDS))
    ap.add_argument("--eval-test", action="store_true",
                    help="also report test metrics; run once, after model selection on val")
    args = ap.parse_args()

    df = pd.read_parquet(OUT_DIR / "features" / "responses.parquet")
    sets = {"linguistic": LING_COLS}
    if DELIVERY_PATH.exists():
        dl = pd.read_parquet(DELIVERY_PATH)
        dcols = [c for c in dl.columns if c != "id" and pd.api.types.is_numeric_dtype(dl[c])]
        df = df.merge(dl[["id"] + dcols], on="id", how="left")
        sets["delivery"] = dcols
        sets["linguistic_delivery"] = LING_COLS + dcols
        print(f"delivery features: {len(dcols)} columns, "
              f"{int(df[dcols].notna().all(axis=1).sum())}/{len(df)} responses complete")
    else:
        print(f"no delivery features at {DELIVERY_PATH}; running linguistic only")

    tr = (df["split"] == "train").to_numpy()
    tr_idx = np.flatnonzero(tr)
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(tr.sum()), groups=df.loc[tr, "user_no"]))
    eval_splits = ["val"] + (["test"] if args.eval_test else [])

    rows, chosen = [], {}
    pdir = OUT_DIR / "predictions"
    for (fs, cols), learner in itertools.product(sets.items(), args.learners):
        name = f"{fs}_{learner}"
        X = df[cols].to_numpy(float)
        preds = df[["id", "user_no", "question_id", "split"]].copy()
        for t in TARGETS:
            y = df[t].to_numpy(float)
            params, cv = select_params(learner, X[tr], y[tr], folds)
            chosen[f"{name}/{t}"] = {"params": params, "cv_ccc": cv}
            p = np.empty(len(df))
            for f_tr, f_va in folds:   # out-of-fold train predictions
                p[tr_idx[f_va]] = fit_predict(learner, params, X[tr_idx[f_tr]], y[tr_idx[f_tr]],
                                              X[tr_idx[f_va]])
            p[~tr] = fit_predict(learner, params, X[tr], y[tr], X[~tr])
            preds[f"pred_{t}"] = p
            for s in eval_splits:
                m = (df["split"] == s).to_numpy()
                r = {"model": name, "target": t, "split": s, "cv_ccc_train": cv,
                     **regression_metrics(y[m], p[m])}
                r["spearman_ci95"] = cluster_bootstrap_ci(y[m], p[m], df.loc[m, "user_no"], "spearman")
                rows.append(r)
        preds.to_parquet(pdir / f"{name}.parquet", index=False)
        print(f"done: {name}")

    res = pd.DataFrame(rows)
    rdir = OUT_DIR / "reports"
    for s in eval_splits:
        res[res["split"] == s].to_csv(rdir / f"feature_models_{s}.csv", index=False)
    write_json(rdir / "feature_models_config.json", {"grids": GRIDS, "feature_sets": sets,
                                                     "chosen": chosen})
    for s in eval_splits:
        sub = res[res["split"] == s]
        for metric in ["spearman", "ccc", "r2"]:
            print(f"\n{s} {metric}")
            print(sub.pivot(index="model", columns="target", values=metric)
                  .reindex(columns=TARGETS).round(3).to_string())


if __name__ == "__main__":
    main()
