"""Video-branch predictions on the shared participant split, for fusion, without touching their checkout.

The video teammate's train_v2.py cross-validates over all RecruitView videos, so its predictions
cannot be fused with the text/audio branches as they are. This runner loads their code unchanged
from PE/video_branch and reuses, as is:
  train.load_recruitview()       features_recruitview.csv + metadata merge and their quality filter
                                 (n_samples >= 8, face_rate >= 0.5)
  train.feature_columns()        visual feature columns only
  train_v2.robust_filter()       their `robust` set: drops length-sensitive *_max/_min/_active_segments
  train_v2.fit_model("Ridge")    their deployable model: median impute + scale + Ridge, alpha by
                                 GroupKFold(3) over participants
and fits it on our split: train = out-of-fold over the same GroupKFold(5, shuffle, seed 42) by
user_no folds as fusion/fuse.py, val/test = fit on all of train. Videos removed by their quality
filter get no prediction (NaN); fusion falls back to text + audio for them.

Output: outputs/video_branch_run/predictions.parquet (id, split, pred_<target>)
        outputs/video_branch_run/report.md (val metrics, feature count, coverage)
Run:    python fusion/run_video_branch.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PE = Path(__file__).resolve().parent.parent
VIDEO_REPO = PE / "video_branch"
TEXT_OUT = PE / "outputs" / "text"
WORK = PE / "outputs" / "video_branch_run"
sys.path.insert(0, str(PE / "text_branch"))
from text_common import SEED, TARGETS, cluster_bootstrap_ci, regression_metrics  # noqa: E402

VIDEO_TARGETS = TARGETS + ["facial_expression"]
N_FOLDS = 5


def load_their_modules():
    # train.py imports skrebate (ReliefF) at module level for its own classifier pipeline, which
    # this runner never calls; a placeholder avoids installing it just to import the loaders.
    if "skrebate" not in sys.modules:
        try:
            import skrebate  # noqa: F401
        except ImportError:
            sys.modules["skrebate"] = types.SimpleNamespace(ReliefF=None)
    sys.path.insert(0, str(VIDEO_REPO))
    import train as T
    import train_v2 as V2
    T.RECRUITVIEW_FEATURES = VIDEO_REPO / "features_recruitview.csv"
    T.RECRUITVIEW_META = PE / "RecruitView" / "metadata.jsonl"
    return T, V2


def load_video_features(ids: pd.Series):
    """Their robust visual features, row-aligned with `ids`; all-NaN rows for filtered-out videos."""
    T, V2 = load_their_modules()
    rv = T.load_recruitview()
    feats = V2.robust_filter("RecruitView", T.feature_columns(rv, label_col=""))
    rv["id"] = T._digits(rv["file_name"]).map(lambda n: f"{int(n):04d}")
    assert rv["id"].is_unique, "duplicate video ids after merge"
    return rv.set_index("id").reindex(ids)[feats].to_numpy(float), feats, V2


def main() -> None:
    df = pd.read_parquet(TEXT_OUT / "features" / "responses.parquet")[["id", "user_no", "split", *TARGETS]]
    face = pd.concat([pd.read_csv(PE / "split_dataset" / f"{s}.csv", dtype={"id": str})
                      for s in ("train", "val", "test")])[["id", "facial_expression"]]
    df = df.merge(face, on="id", how="left")
    X, feats, V2 = load_video_features(df["id"])
    has = ~np.isnan(X).all(axis=1)
    tr = (df["split"] == "train").to_numpy()
    groups = df["user_no"].to_numpy()
    # identical to fuse.py: same rows, same order, same splitter
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(tr.sum()), groups=groups[tr]))
    print(f"{len(feats)} robust visual features; video available for {has.sum()}/{len(df)} responses "
          f"(train {has[tr].sum()}/{tr.sum()})")

    tr_idx = np.flatnonzero(tr)
    out = df[["id", "split"]].copy()
    alphas = {}
    for t in VIDEO_TARGETS:
        y = df[t].to_numpy(float)
        p = np.full(len(df), np.nan)
        for f_tr, f_va in folds:
            fit_rows = tr_idx[f_tr][has[tr_idx[f_tr]]]
            pred_rows = tr_idx[f_va][has[tr_idx[f_va]]]
            est = V2.fit_model("Ridge", X[fit_rows], y[fit_rows], groups[fit_rows])
            p[pred_rows] = est.predict(X[pred_rows])
        fit_rows, pred_rows = np.flatnonzero(tr & has), np.flatnonzero(~tr & has)
        est = V2.fit_model("Ridge", X[fit_rows], y[fit_rows], groups[fit_rows])
        p[pred_rows] = est.predict(X[pred_rows])
        alphas[t] = float(est.best_params_["ridge__alpha"])
        out[f"pred_{t}"] = p
        print(f"done: {t} (alpha {alphas[t]:g})", flush=True)

    WORK.mkdir(parents=True, exist_ok=True)
    out.to_parquet(WORK / "predictions.parquet", index=False)

    va = (df["split"] == "val").to_numpy() & has
    md = ["# Video branch on the shared split (val)", "",
          f"Their `robust` Ridge, {len(feats)} features. Video available for {has.sum()}/{len(df)} "
          f"responses; val n = {va.sum()} of {(df['split'] == 'val').sum()}. Spearman / CCC with 95% "
          "participant-bootstrap CIs. Train predictions are out-of-fold (fusion folds).", "",
          "| target | Spearman | CCC | RMSE | alpha |", "|---|---|---|---|---|"]
    for t in VIDEO_TARGETS:
        y, p, g = df.loc[va, t].to_numpy(float), out.loc[va, f"pred_{t}"].to_numpy(float), groups[va]
        m = regression_metrics(y, p)
        s_lo, s_hi = cluster_bootstrap_ci(y, p, g, "spearman")
        c_lo, c_hi = cluster_bootstrap_ci(y, p, g, "ccc")
        md.append(f"| {t} | {m['spearman']:.3f} [{s_lo:+.2f}, {s_hi:+.2f}] | {m['ccc']:.3f} "
                  f"[{c_lo:+.2f}, {c_hi:+.2f}] | {m['rmse']:.3f} | {alphas[t]:g} |")
    (WORK / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
