"""Train and compare text-branch regressors on the shared participant split.

Models (one Ridge per target; alpha chosen by participant-grouped 5-fold CV on train):
  mean            predict the training mean (reference floor)
  question_mean   predict the training mean of the response's question (question-effect floor)
  length_only     log word count only (checks whether a model beats "longer = better")
  duration_only   log clip seconds only (diagnostic; not a text feature)
  linguistic      interpretable linguistic features + qa_cosine
  embedding       frozen sentence embeddings of answer and question
  embedding_ling  linguistic features + answer embedding reduced to 32 PCs

Reports validation metrics only. The test split is evaluated only with --eval-test, which
should be run once, after the final model has been chosen on validation.

Writes:
  outputs/text/reports/baselines_val.csv (and _test.csv with --eval-test)
  outputs/text/predictions/<model>.parquet  train = out-of-fold, val/test = fit on full train
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from text_common import (OUT_DIR, SEED, TARGETS, ccc, cluster_bootstrap_ci, regression_metrics,
                         write_json)

ALPHAS = np.logspace(-2, 5, 15)
N_FOLDS = 5
LING_COLS = ["empty_transcript", "log_n_words", "n_sentences", "mean_sentence_words",
             "mean_word_chars", "mattr_50", "hedge_per100", "connective_per100", "number_per100",
             "fp_singular_per100", "fp_plural_per100", "question_overlap", "qa_cosine"]


N_PCA = 32   # embedding_ling: answer embedding reduced to 32 PCs (fit inside each fold)


def make_model(alpha: float, n_ling: int = 0):
    """n_ling > 0: the first n_ling columns are linguistic features and the remaining columns
    are an embedding, which is reduced with PCA so it does not swamp the linguistic features
    under a single Ridge penalty."""
    if n_ling:
        cols = ColumnTransformer([
            ("ling", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
             slice(0, n_ling)),
            ("emb", make_pipeline(PCA(N_PCA, random_state=SEED), StandardScaler()),
             slice(n_ling, None)),
        ])
        return make_pipeline(cols, Ridge(alpha=alpha))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def clip_bounds(y: np.ndarray, clip: bool) -> tuple[float, float]:
    """Labels are heavy-tailed (about -10..9 with IQR about -0.6..0.6). Winsorising the
    training targets at the 1st/99th percentile stops a few extreme labels from dominating
    the squared loss. Evaluation always uses the raw labels."""
    return tuple(np.percentile(y, [1, 99])) if clip else (-np.inf, np.inf)


def fit_predict(X_tr, y_tr, X_te, alpha, clip, n_ling=0):
    lo, hi = clip_bounds(y_tr, clip)
    if X_tr is None:
        return np.full(len(X_te), np.clip(y_tr, lo, hi).mean())
    if X_tr.dtype == object:   # question_mean: per-question training mean, global mean if unseen
        yc = pd.Series(np.clip(y_tr, lo, hi))
        q_mean = yc.groupby(X_tr.ravel()).mean()
        return pd.Series(X_te.ravel()).map(q_mean).fillna(yc.mean()).to_numpy(float)
    return make_model(alpha, n_ling).fit(X_tr, np.clip(y_tr, lo, hi)).predict(X_te)


def select_alpha(X, y, folds, clip, n_ling=0) -> tuple[float, float]:
    """Pick alpha by mean CCC over grouped folds (ties broken by lower alpha)."""
    best = (-np.inf, ALPHAS[0])
    for a in ALPHAS:
        scores = [ccc(y[va], fit_predict(X[tr], y[tr], X[va], a, clip, n_ling))
                  for tr, va in folds]
        best = max(best, (float(np.mean(scores)), a), key=lambda t: t[0])
    return best[1], best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-tag", default="all-mpnet-base-v2")
    ap.add_argument("--no-clip", action="store_true", help="train on raw (unwinsorised) targets")
    ap.add_argument("--eval-test", action="store_true",
                    help="also report test metrics; run once, after model selection on val")
    args = ap.parse_args()
    clip = not args.no_clip

    fdir = OUT_DIR / "features"
    df = pd.read_parquet(fdir / "responses.parquet")
    ans_emb = np.load(fdir / f"emb_{args.embed_tag}_answer.npy")
    emb = np.hstack([ans_emb, np.load(fdir / f"emb_{args.embed_tag}_question.npy")])
    ling = df[LING_COLS].to_numpy(float)
    inputs = {
        "mean": None,
        "question_mean": df[["question_id"]].to_numpy(object),
        "length_only": df[["log_n_words"]].to_numpy(float),
        # diagnostic only (clip timing is the audio branch's feature): is "length" really duration?
        "duration_only": np.log(df[["clip_seconds"]].to_numpy(float)),
        "linguistic": ling,
        "embedding": emb,
        "embedding_ling": np.hstack([ling, ans_emb]),
    }
    n_ling_of = {"embedding_ling": len(LING_COLS)}

    tr = (df["split"] == "train").to_numpy()
    groups = df.loc[tr, "user_no"].to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(tr.sum()), groups=groups))
    eval_splits = ["val"] + (["test"] if args.eval_test else [])

    rows, chosen = [], {}
    pdir = OUT_DIR / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    for name, X in inputs.items():
        nl = n_ling_of.get(name, 0)
        preds = df[["id", "user_no", "question_id", "split"]].copy()
        for t in TARGETS:
            y = df[t].to_numpy(float)
            X_tr = None if X is None else X[tr]
            if X is None or X.dtype == object:
                alpha, cv = np.nan, np.nan
            else:
                alpha, cv = select_alpha(X_tr, y[tr], folds, clip, nl)
            chosen[f"{name}/{t}"] = {"alpha": alpha, "cv_ccc": cv}

            p = np.empty(len(df))
            for f_tr, f_va in folds:   # out-of-fold train predictions (safe for later fusion)
                idx = np.flatnonzero(tr)
                p[idx[f_va]] = fit_predict(None if X is None else X_tr[f_tr], y[tr][f_tr],
                                           np.zeros(len(f_va)) if X is None else X_tr[f_va],
                                           alpha, clip, nl)
            rest = ~tr
            p[rest] = fit_predict(X_tr, y[tr], np.zeros(rest.sum()) if X is None else X[rest],
                                  alpha, clip, nl)
            preds[f"pred_{t}"] = p

            for s in eval_splits:
                m = (df["split"] == s).to_numpy()
                r = {"model": name, "target": t, "split": s, "alpha": alpha, "cv_ccc_train": cv,
                     **regression_metrics(y[m], p[m])}
                r["spearman_ci95"] = cluster_bootstrap_ci(y[m], p[m], df.loc[m, "user_no"], "spearman")
                r["ccc_ci95"] = cluster_bootstrap_ci(y[m], p[m], df.loc[m, "user_no"], "ccc")
                rows.append(r)
        preds.to_parquet(pdir / f"{name}.parquet", index=False)
        print(f"done: {name}")

    res = pd.DataFrame(rows)
    rdir = OUT_DIR / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    for s in eval_splits:
        res[res["split"] == s].to_csv(rdir / f"baselines_{s}.csv", index=False)
    write_json(rdir / "baselines_config.json", {"alphas_grid": ALPHAS.tolist(), "n_folds": N_FOLDS,
               "clip_train_targets_1_99": clip, "embed_tag": args.embed_tag,
               "linguistic_features": LING_COLS, "embedding_ling_pca": N_PCA, "chosen": chosen})

    for s in eval_splits:
        sub = res[res["split"] == s]
        print(f"\n=== {s.upper()} (n={int(sub['n'].iloc[0])}) ===")
        for metric in ["spearman", "ccc", "r2", "rmse"]:
            print(f"\n{metric}")
            print(sub.pivot(index="model", columns="target", values=metric)
                  .reindex(index=list(inputs), columns=TARGETS).round(3).to_string())


if __name__ == "__main__":
    main()
